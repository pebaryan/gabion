# qwen35 inference — evaluation protocol & optimization targets

**This is the authoritative measurement standard for all qwen35 inference work
in this repo.** Any agent (Codex, Hermes, future sessions) claiming a speedup,
a memory win, or a correctness-preserving change MUST produce evidence from
this protocol: the single report line from `tools/bench_qwen35_pipeline.py`,
run before and after the change, on this machine. Numbers measured any other
way are anecdotal and will be rejected.

Last updated: 2026-08-27 (port of the fast standalone runner into the adapter).

---

## 1. Fixed environment (do not deviate)

- Hardware: 2× NVIDIA RTX 5060 Ti 16GB (sm_120), 34GB RAM, Windows 11 (WDDM).
- Models:
  - 27B: `D:/aimodels/Qwen3.8-27B-IQ4_NL.gguf` (65 layers: 17 full-attn / 48 GDN; IQ4_NL layers, Q8_0 alpha/beta, Q5_K ssm_out; untied head)
  - 0.8B: `D:/aimodels/Qwen3.5-0.8B-IQ4_NL.gguf` (24 layers: 6 full / 18 GDN; tied head)
- Canonical run environment (bash/MSYS):
  ```
  unset CPU CUDA WEBGPU METAL CL AMD NV
  export DEV=CUDA CC="C:/Program Files/LLVM/bin/clang.exe" \
         PATH="C:/Program Files/LLVM/bin:$PATH" PYTHONPATH="D:/code/gabion"
  ```
- Python: `C:/Users/aryan/miniconda3/envs/p311/python.exe` (tinygrad 0.14.0).

### Device pinning (hard-won, tinygrad 0.14 gotchas)
- `HCQ_VISIBLE_DEVICES` is **ignored** by tinygrad (deprecated — `runtime/support/hcq.py`).
- `DEV=CUDA:1` does **not** select device 1 (the second colon-field is a *renderer name*).
- The only working mechanisms: explicit indexed device strings. The adapter takes
  `device="CUDA:0"/"CUDA:1"` (`from_gguf_shard(..., device=...)`) or `QWEN35_DEVICE=CUDA:<idx>`.
- **Process shape is part of the measurement**: ONE process holding BOTH shards
  (`device="CUDA:0"` + `device="CUDA:1"`, exactly like the standalone runner).
  Two separate processes, one per GPU, hit a WDDM per-context wall (~7.4GB each,
  measured 2026-08-27) and OOM during the 27B load. The bench tool already does
  the single-process shape — keep it that way.

## 2. Metrics & current targets

Primary metrics (everything is wall-clock, steady state, after warmup):

| metric | definition | 27B target (current best) | 0.8B target |
|---|---|---|---|
| **decode speed** | one generated token = one full forward (both shards). Report min / **median** / mean s/tok and tok/s | **≤ 0.40 s/tok (≥ 2.5 tok/s)** — runner 0.40, 2026-08-27 | **≥ 7 tok/s** (adapter measured 10–15) |
| **prefill speed** | a prompt of N tokens passed in one call (row-by-row T=1 jit inside the adapter). Report N / wall = tok/s | sequential ≥ **1.5 tok/s** (0.67 s/tok); chunked-direct 0.54 tok/s (T=16) | ≥ 7 tok/s |

Secondary metrics (report alongside, regressions are blocking):

| metric | target |
|---|---|
| VRAM peak per device (load, prefill, decode) | ≤ 15.5 GiB transient, ≤ 13 GiB steady per card (baseline 8.7 / 9.4 GiB) |
| load time (mmap → all weights resident) | ≤ 60 s (runner 42 s; adapter ~25–30 s per shard) |
| warmup / jit-capture time | report separately (one-time; current ~30–90 s per shard on first run, disk cache warms it) |

**A change "counts" only if:** median decode improves ≥ 5% OR prefill improves
≥ 10% at equal-or-better VRAM — with the correctness gate PASSING (below).
Regressions of any size in correctness gates or VRAM headroom are blocking,
even if speed improves.

## 3. Correctness gates (run FIRST; FAIL ⇒ stop, do not time)

Fixed prompt (tokenizer-stable ids, verified 2026-08-27):
`<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n`
(0.8B: no bos, 15 ids `[248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198]`;
27B: bos 248044 prefix, 16 ids).

Generated-token gates (greedy argmax, exact match required):
- **0.8B: `[248068, 198, 90700, 8340, 25, 271]`** (tinygrad-oracle reference)
- **27B: `[248068, 198, 760, 1156, 369, 9859, ...]`** (runner reference; first token 248068 is mandatory)

Parity rule when comparing two implementations (e.g. fused vs unfused path):
- logits max-abs diff ≤ 1e-3 relative for f32 paths; ≤ 1e-2 relative for f16
  paths (accumulation-order 1-ULP drift is benign — 20× below IQ4_NL noise);
- an argmax flip is allowed only on near-ties (top-2 within the documented
  delta) and must be called out; any systematic flip is a bug.

Unit tests: `tests/test_qwen35_text.py` (split-vs-split parity, decode state,
IQ4_XS dequant) must pass.

## 4. Measurement protocol (exact steps)

1. **Gate**: `--verify` on the bench (or the 0.8B oracle check). PASS required.
2. **Warmup**: 1 synthetic prefill (32 tokens) + 2 decode steps — discarded
   (this is where jit capture + kernel compile land). Report warmup time.
3. **Decode**: fresh state, prefill the gate prompt, then M = 10 single-token
   steps; per-step wall clock. Report min / median / mean s/tok. **Median is
   the headline number.**
4. **Prefill**: fresh state per length; N = 64 and N = 512 synthetic ids
   (`i % vocab`); one call each; N / wall = tok/s.
5. **Memory**: `tinygrad.helpers.GlobalCounters.mem_used_per_device` after
   load, after warmup, and max during decode.
6. **Report line** (paste verbatim into the commit/PR):
   `tools/bench_qwen35_pipeline.py --verify` prints
   `qwen35-bench v1 gguf=... load_s=... warmup_s=... decode_s_tok_min/median/mean=... decode_tok_s=... prefill64_tok_s=... prefill512_tok_s=... vram_load=... gate=PASS`.

Comparison rules: identical env, identical process shape, identical warmup.
Always run before AND after on the same boot state (no other GPU load — kill
stray python processes first; they hold CUDA contexts and distort VRAM).

## 5. Harness inventory

- **`tools/bench_qwen35_pipeline.py` — THE tool** (single process, both shards,
  gate + warmup + decode + prefill + memory, one report line).
- `tools/bench_qwen35_cuda.py` — shard-level steady-state only (legacy; for
  worker tuning, not for claims).
- `tests/test_qwen35_text.py` — fixture correctness.
- Reference implementation (canonical numbers): `D:/tmp/_q27_run.py` (standalone
  runner; 27B 0.40 s/tok, 0.8B 7.0–7.1 tok/s oracle-identical). History and
  dead ends: `docs/qwen35-inference-status.md` + `D:/tmp/q27-inference-status.md`.

## 6. Multi-agent research rules

1. **Evidence or it didn't happen**: every claim ships the before/after report
   line. No exceptions.
2. **Never modify** the gates, the gate prompt, or the reference numbers without
   a documented reason in the same commit.
3. **Forbidden measurement shortcuts**: `--nojit`, skipping warmup, f16 wire
   rounding inside the measured loop, reusing stale captures, timing with other
   GPU load present, timing per-shard and summing.
4. **Do not re-attempt without new information**: the raw-CUDA GDN scan
   (correct standalone, but drifts the 48-layer recurrence → token flips, and
   49 jit replays = 4× slower); custom-kernel scan codegen walls (coalesce
   unpack assert, "attempting multiple stores", DMC hang).
5. **Knowledge lives in the repo**: docs/ here are the shared brain. Other
   agents cannot see `D:/tmp` or Hermes skills — the 2026-08-27 failure
   (Codex rebuilt the slow path because the fast work was invisible). When you
   learn something, commit it to `docs/` in the same change.
6. **Ranked optimization levers** (each est. 10–20% on 0.40 s/tok, in order):
   - **#4 layer interleaving** (overlap the two shards' kernels across GPUs)
   - **#5 speculative decoding** (0.8B draft model at ≥ 7 tok/s, 27B verify)
   - **#6 fp8 KV cache / GPU-side dequant at load**
   - plus: quantify and document the two-process WDDM memory wall (the
     single-process dual-device shape is the current constraint).
7. Any change must keep the adapter API stable (`forward_shard_*`,
   `stream_state`, `from_gguf_shard`) — the mesh server/workers depend on it.
