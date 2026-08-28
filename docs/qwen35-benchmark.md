# qwen35 inference — evaluation protocol & optimization targets

**This is the authoritative measurement standard for all qwen35 inference work
in this repo.** Any agent (Codex, Hermes, future sessions) claiming a speedup,
a memory win, or a correctness-preserving change MUST produce evidence from
this protocol: the single report line from `tools/bench_qwen35_pipeline.py`,
run before and after the change, on this machine. Numbers measured any other
way are anecdotal and will be rejected.

Last updated: 2026-08-28 (u32-word quant loads accepted; see §2b).

### 2b. u32-word quant loads (2026-08-28, accepted — current default)

All fused dequant kernels (IQ4_NL GEMV/GEMM, Q5_K `ssm_out` GEMV/GEMM, and the
Q6_K output-head GEMV) now load their quant bytes as **uint32 words** (4 bytes
per load, shift-unpacked in-kernel) instead of scalar u8 loads. The unpack is
bit-exact versus the byte-load kernels (verified maxabs=0 on IQ4 f32/f16;
Q5/Q6 parity within reference tolerances), and a JIT-replay microbench
measured the IQ4 GEMV at **87 vs 232 µs** (2.65x) at 5120x5120. The quantized
weights are stored as native u32 tensors (a numpy view at load — zero extra
VRAM) with lazy u8 bitcast views serving the dequant fallbacks; an earlier
variant that bitcast per call materialized persistent copies inside the JIT
(+1.26 GiB on CUDA:1) and was replaced. `QWEN35_IQ4_U32=0` reverts the IQ4
GEMV to the byte-load kernel for diagnostics.

Same-boot baseline (pre-change defaults):
`qwen35-bench v1 path=native split=auto split_actual=33 q5_head=so q5_local=16 iq4_local=32 iq4_tile4=1 q6_head=fused q6_local=256 rstride=2 prefill_batch=4 gguf=Qwen3.8-27B-IQ4_NL.gguf load_s=72.7 warmup_s=10.7 decode_s_tok_min=0.2723 decode_s_tok_median=0.2783 decode_s_tok_mean=0.2778 decode_tok_s=3.59 prefill64_tok_s=22.252 prefill512_tok_s=23.319 vram_load=CUDA:9.00GiB,CUDA:1:9.82GiB,NPY:0.00GiB,PYTHON:0.00GiB gate=PASS`

After (new default report):
`qwen35-bench v1 path=native split=auto split_actual=33 q5_head=so q5_local=16 iq4_local=32 iq4_tile4=1 q6_head=fused q6_local=256 rstride=2 prefill_batch=4 gguf=Qwen3.8-27B-IQ4_NL.gguf load_s=31.9 warmup_s=6.5 decode_s_tok_min=0.1916 decode_s_tok_median=0.1928 decode_s_tok_mean=0.1927 decode_tok_s=5.19 prefill64_tok_s=29.395 prefill512_tok_s=30.942 vram_load=CUDA:8.82GiB,CUDA:1:9.67GiB,NPY:0.00GiB,PYTHON:0.00GiB gate=PASS`

That is **decode 0.2783 → 0.1928 s/tok (−31%, 3.59 → 5.19 tok/s)** and
**prefill 22.25/23.32 → 29.40/30.94 tok/s (+32%/+33%)** at lower VRAM
(9.00/9.82 → 8.82/9.67 GiB), gate PASS. An independent confirming clean run
measured **0.1932 s/tok (5.18 tok/s)** and **29.643 / 30.872 prefill tok/s**,
gate PASS, same VRAM.

Rejected in the same 2026-08-28 session (all gate-passing but slower):
- **Raw NVRTC warp-per-row decode GEMV** (`QWEN35_IQ4_RAW_GEMV=1`, kept as a
  diagnostic): 2x faster standalone, but raw `PROGRAM` UOps are not batched
  into the TinyJit CUDA graph, so at T=1 per-launch WDDM overhead regressed
  decode to **0.3316 s/tok** (vs 0.2783). JIT-replay microbench: raw 341 vs
  UOp 262 µs/GEMV. The batch-4 prefill tile is unaffected (amortized).
- **Prefill batch 8 with a raw batch-8 tile** (the tile builder is now
  generalized over T): **14.389 / 15.989 prefill tok/s** vs batch 4's
  22.25/23.32; decode unchanged. Batch 4 remains the default.

### 2a. Measured 27B comparison (2026-08-28)

On the fixed IQ4_NL model and clean two-GPU boot, the existing mesh-compatible
adapter path measured:

`qwen35-bench v1 path=adapter gguf=Qwen3.8-27B-IQ4_NL.gguf load_s=46.4 warmup_s=13.5 decode_s_tok_min=0.3985 decode_s_tok_median=0.4099 decode_s_tok_mean=0.4112 decode_tok_s=2.44 prefill64_tok_s=2.541 prefill512_tok_s=2.531 vram_load=CUDA:8.84GiB,CUDA:1:9.06GiB,NPY:0.00GiB,PYTHON:0.00GiB gate=PASS`

The native local dual-device path then measured:

`qwen35-bench v1 path=native gguf=Qwen3.8-27B-IQ4_NL.gguf load_s=27.5 warmup_s=6.5 decode_s_tok_min=0.3837 decode_s_tok_median=0.3843 decode_s_tok_mean=0.3850 decode_tok_s=2.60 prefill64_tok_s=5.717 prefill512_tok_s=5.790 vram_load=CUDA:8.84GiB,CUDA:1:9.07GiB,NPY:0.00GiB,PYTHON:0.00GiB gate=PASS`

This is a 6.2% median decode improvement and roughly 2.3x prefill improvement;
the correctness gate and VRAM headroom are unchanged. The native path uses a
destination queue synchronization by default (`QWEN35_PIPELINE_SYNC=dst`).

Before Q5 fusion, the graph-realization cadence was also swept on the same boot: `QWEN35_RSTRIDE=8`
measured **0.4098 s/tok** and `QWEN35_RSTRIDE=16` measured **0.4100 s/tok**;
both gates passed, but both regressed decode versus the then-default `4`, so
the default was retained at that stage.
The remaining `QWEN35_RSTRIDE=2` check measured **0.3467 s/tok** with the
fused Q6_K head and `gate=PASS`, tying the pre-Q5 default's **0.3462–0.3471
s/tok** range without a material gain. After Q5 fusion, two clean stride-2
runs measured **6.555 / 6.612** and **6.542 / 6.607 prefill tok/s** while
tying stride 4 on decode, so stride 2 is now the default.

The Q6_K output-head chunk count was checked as well: four chunks measured
**0.3982 s/tok** and 16 chunks **0.4310 s/tok**, versus the retained eight-chunk
path. A fused per-chunk argmax experiment passed the gate but measured only
**1.57 tok/s**, so the existing full-logit argmax remains faster.

The successful replacement is a compile-time-unrolled fused Q6_K GEMV for the
large greedy head: two clean runs measured **0.3462** and **0.3468 s/tok**
(**2.89 / 2.88 tok/s**), both with `gate=PASS` and VRAM **8.84 / 9.07 GiB**.
It is now the default (`QWEN35_FUSE_Q6_HEAD=1`); set it to `0` only to compare
against the chunked fallback.

Pre-Q5 default report:
`qwen35-bench v1 path=native split=auto q6_head=fused q6_local=256 gguf=Qwen3.8-27B-IQ4_NL.gguf load_s=27.1 warmup_s=6.3 decode_s_tok_min=0.3457 decode_s_tok_median=0.3466 decode_s_tok_mean=0.3471 decode_tok_s=2.88 prefill64_tok_s=5.815 prefill512_tok_s=5.872 vram_load=CUDA:8.84GiB,CUDA:1:9.07GiB,NPY:0.00GiB,PYTHON:0.00GiB gate=PASS`

The accepted next step is a shape-specific fused Q5_K GEMV for the 27B
GDN `ssm_out` matrices (`QWEN35_FUSE_Q5_PART=so`, now the default). Two clean
full-protocol runs at local width 256 measured **0.3179 / 0.3180 s/tok**;
the subsequent local-width sweep selected 64. At local width 64, two clean
runs measured **0.3135 / 0.3150 s/tok** (**3.19 / 3.17 tok/s**) and
**6.423 / 6.421 prefill64**, **6.479 / 6.479 prefill512 tok/s**; all
correctness gates passed. A one-warp local width of 32 was then faster: two
clean runs measured **0.3111 / 0.3113 s/tok** (**3.21 / 3.21 tok/s**) and
**6.475 / 6.465 prefill64**, **6.525 / 6.526 prefill512 tok/s**. The
width 16 improved it again: two clean full
runs measured **0.3092 / 0.3091 s/tok** (**3.23 / 3.24 tok/s**) and
  **6.515 / 6.506 prefill64**, **6.565 / 6.569 prefill512 tok/s**; all gates
passed. Width 8 fell back to **3.21 tok/s** and width 10 to **3.23 tok/s**
in gate-passing probes, so 16 remains selected. The corresponding default
report is recorded below.

The fused kernel width sweep was flat at this resolution: `QWEN35_Q6_LOCAL=128`
measured **0.3472 s/tok**, `=256` **0.3471 s/tok**, and `=512` **0.3490 s/tok**;
the additional 10-warp `=320` point measured **0.3476 s/tok**; all gates
passed, so 256 remains the default.

With the fused Q6_K head enabled, disabling the f16 GDN GEMV fusion
(`QWEN35_FUSE_F16=0`) measured **0.5403 s/tok** and **3.657/3.692 tok/s**
prefill, gate PASS. The fused-f16 default is therefore retained.

The post-Q6 synchronization check also retained destination sync:
`QWEN35_PIPELINE_SYNC=src` measured **0.3535 s/tok** and **5.680/5.722 tok/s**
prefill, gate PASS, versus the current `dst` result **0.3466 s/tok** and
**5.815/5.872 tok/s**.

A direct T>1 speculative-verification probe was not viable: after adding the
reference multi-row GDN scan, the full 16-token gate prompt returned final
argmax `0` instead of `248068`, and direct throughput was only **0.122 / 0.211
/ 0.355 / 0.521 tok/s** for T=2/4/8/16. The branch was reverted; supported
decode remains the correct T=1 path.

The shape-specific boundary matters: fusing Q5_K for both matrices, or for
`attn_qkv` alone, failed the first 27B correctness gate (`[198, 198, ...]`
versus the expected prefix), so those modes are disabled. A numerically exact
load-time f16 Q5_K cache passed the gate, but measured only **1.65 decode
tok/s** and **3.084 / 3.112 prefill tok/s** with VRAM **8.31 / 8.58 GiB**;
it was reverted as a clear regression.

The next accepted change is bounded native prefill. Each prompt is processed in
four-token on-device chunks; the last chunk returns only its final-row argmax,
while one-token generation continues to use the lean decode graph. The clean
batch-4 run measured **9.038 / 9.486 prefill tok/s** at 64/512 tokens versus
the prior **7.020 / 7.082**, with decode statistically unchanged at **0.2878
s/tok (3.48 tok/s)**. VRAM after warmup remained **9.00 / 9.82 GiB** and the
gate passed. Batch 8 also passed and fit, but regressed to **7.742 / 8.501
prefill tok/s** and increased warmup, so it is not the default.
The neighboring candidates were also gate-passing but slower: batch 2 measured
**7.278 / 7.404**, batch 3 **4.907 / 9.264**, batch 5 **2.907 / 8.260**, and
batch 6 **8.324 / 7.925** tok/s at 64/512. The uneven candidates pay for a
slow final partial chunk; batch 4 is the only consistently fast aligned point
in this local sweep.

Latest default report:
`qwen35-bench v1 path=native split=auto split_actual=33 q5_head=so q5_local=16 iq4_local=32 iq4_tile4=1 q6_head=fused q6_local=256 rstride=2 prefill_batch=4 gguf=Qwen3.8-27B-IQ4_NL.gguf load_s=33.6 warmup_s=6.9 decode_s_tok_min=0.2721 decode_s_tok_median=0.2748 decode_s_tok_mean=0.2752 decode_tok_s=3.64 prefill64_tok_s=22.130 prefill512_tok_s=23.364 vram_load=CUDA:9.00GiB,CUDA:1:9.82GiB,NPY:0.00GiB,PYTHON:0.00GiB gate=PASS`

The accepted batch-4 IQ4 path is now a raw NVRTC-compiled CUDA tile. One
256-thread block computes one output row, cooperatively reduces the input
dimension, and retains each dequantized weight across four token
accumulators. The compiled program is represented as a normal tinygrad
`PROGRAM` UOp, so TinyJit captures and replays it. On the canonical 27B file
it improved prefill from **9.038 / 9.486** to **22.130 / 23.364 tok/s** at
64/512 tokens (**+145% / +146%**) with the same **9.00 / 9.82 GiB** VRAM and
the exact six-token gate passing. An independent opt-in run measured **22.266
/ 23.104 tok/s**, confirming the result. Set `QWEN35_IQ4_RAW_TILE4=0` for the
prior portable UOp kernel.

Additional 2026-08-28 rejects after the batch-4 selection: removing the
destination queue sync passed the gate but fell to **3.40 decode tok/s** and
**8.717 / 9.105 prefill tok/s**; stride 1 measured **3.44** and
**8.889 / 9.324**. Q6 local widths 16/32 were **3.43 / 3.39 decode tok/s**;
64–512 remains better at 256. A Q5 `attn_qkv`-only resident f16 cache passed
but regressed to **3.24 decode tok/s** and raised VRAM to **10.17 / 10.98
GiB**. Separating batched-IQ4 width did not help: 16 was **9.064 / 9.448** and
64 **7.225 / 7.530 prefill tok/s**. These routes remain out of the defaults.

The actual auto boundary is layer **33** (the benchmark now reports
`split_actual`).  A same-boot adjacent split sweep did not beat it by the
acceptance rule. Auto/33 measured **3.26 decode tok/s** and **8.783 / 9.173
prefill tok/s**; split 34 measured **3.35** and **8.644 / 8.947** with
**9.26 / 9.57 GiB**; split 32 measured **3.35** and **9.024 / 9.469** with
**8.75 / 10.08 GiB**; split 31 measured **3.38** and **9.059 / 9.464** with
**8.52 / 10.30 GiB**. All gates passed. The small gains at 31/32 are below
the threshold, increase peak VRAM, and merely tie the retained clean prefill
best, so byte-balanced auto/33 remains the default.

Speculative decoding was rechecked after correct batch-4 target prefill became
available. Both local Qwen3.5 drafts (0.8B IQ4_NL and 9B-distilled IQ4_NL)
matched just **2 of 12** greedy target tokens on the fixed France prompt. The
9B draft also cannot be resident alongside the 27B target under WDDM: target
prefill fails on CUDA:1 at **14.60 GiB** used. The available drafts therefore
cannot repay verification or fit the current deployment; speculation remains
out of scope unless a higher-agreement, lower-memory draft is supplied.

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
| **prefill speed** | a prompt of N tokens passed in one call using bounded native chunks. Report N / wall = tok/s and the chunk size | raw IQ4 four-token tiles: **22.130 / 23.364 tok/s** at 64/512 | ≥ 7 tok/s |

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
   (`i % vocab`); one call each; native uses `QWEN35_PREFILL_BATCH=4` by
   default; report N / wall = tok/s and the batch size.
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
  Add `--native` to benchmark the local dual-device `Qwen35TextPipeline` path:
  the shard boundary stays as a synchronized direct tensor transfer instead of
  crossing the mesh-facing NumPy boundary. The default mode remains the
  adapter/websocket-compatible path for comparison.
  `QWEN35_PIPELINE_SYNC=dst` is the validated fast default for the native path;
  use `both` only as a conservative diagnostic fallback.
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
