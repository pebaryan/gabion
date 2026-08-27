# Qwen35 (Qwen3.8-27B / Qwen3.5-0.8B) inference — existing work, measurements & verdicts

**READ THIS BEFORE TOUCHING qwen35 inference code.** There is a proven, fast
standalone implementation (`D:/tmp/_q27_run.py`, not in this repo) whose
techniques and numbers this adapter should adopt. This doc exists because the
fast work lived in scratch files and was invisible to agents working in this
repo — that is the failure this document fixes.

## The fast reference: `D:/tmp/_q27_run.py` (standalone, 2× RTX 5060 Ti 16GB)

Measured (2026-08-27, verified): **27B decode 0.40s/tok (2.5 tok/s)**, 0.8B
7.0-7.1 tok/s oracle-identical. Load 42s, VRAM baseline 8.7/9.4GB.

Verification values (any change must preserve):
- 0.8B oracle (tinygrad's own qwen35 Transformer, `D:/tmp/_q27_oracle.py`):
  `[248068, 198, 90700, 8340, 25, 271]`.
- 27B: first token `248068`; decode `[198, 760, 1156, 369, 9859, 264, 4145]`.

### The techniques that make it fast (the whole point)

1. **Persistent u8 weights in VRAM** — quantized bytes are loaded once as u8
   tensors; the f16 weight is NEVER materialized whole. token_embd/lm_head use
   `KQuant` (Q4_K/Q6_K) with on-device gather+dequant.
2. **Fused IQ4_NL dequant-GEMV custom kernel** (`build_iq4nl_gemv` in the
   runner, j-RANGE reduce form, local=256) replaces dequant+matmul for ALL
   matmul sites in decode (single row). Reads u8 once, no f16 churn. 75µs at
   27B ffn scale vs 133µs for the bare f16 matmul. Both f32 and f16 paths
   fused (`--fuse-f16`, auto-ON for GGUFs ≥8GiB; the f16 path rounds weights
   to f16 inside the kernel to match the dequant's `cast(float16)`).
3. **The GDN recurrent scan runs on the GPU in f32** (state f16, math f32) —
   NOT numpy. The current `qwen35_text.py` adapter does the scan with numpy
   einsum on the CPU and round-trips every projection through Tensor→numpy:
   that is the ~35× regression (measured 14.2s/token shard-0 vs 0.40s/token).
4. T=1 decode through one TinyJit; the jit GRAPHs ~1760 kernels/step into 6
   CUDA-graph batches (the "batched N" kernels in DEBUG=2).

### What NOT to re-attempt (measured dead ends, 2026-08-27)

- Custom-kernel GDN scan (dot/update/out): codegen walls in tinygrad 0.14
  (coalesce unpack assert, "attempting multiple stores", DMC pipeline hang).
- Raw-CUDA GDN scan via forward segments: kernel is correct standalone but the
  f32 accumulation order drifts through the 48-layer f16 state and flips the
  27B's first decode token (198→220), and 49 TinyJit replays cost ~30ms each
  → 1.8s/tok. Reverted; the in-graph tinygrad scan stands.
- Raw-CUDA fused-matmul cut: moot — fused matmuls already run at bench speed
  inside the graphs.
- Upstream tinygrad bump: tinygrad-master is byte-identical to 0.14.0.

### Quantization facts (verified, don't re-derive)

- IQ4_NL block: f16 scale FIRST (bytes 0-2), then 16 nibble bytes. Nibble
  mapping is CONSECUTIVE: byte j low nibble = weight j, high nibble = weight
  j+16 (the runner's `cat(lo, hi)`). Interleaved passes self-consistent tests
  but fails real data.
- GGUF tensor dims are (in, out) — the runner reverses.
- The dequant rounds to f16: `(KVAL * sc).cast(float16)` — fused kernels must
  replicate that rounding.
- Block layouts for all types: `D:/tmp/_quants.c` (llama.cpp reference) and
  the parity harnesses in `D:/tmp/_q27_deq_parity2.py`.

## The raw-CUDA machinery (reusable)

`D:/tmp/_q27_rawscan.py` — compile .cu→PTX with clang alone (no llc):
`clang -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_120
-S -o k.ptx k.cu` (clang 22; sm_90 fallback). With -nocudainc, define the CUDA
shims manually (`__global__` attribute, `struct __half { __fp16 x; }`, the
`__nvvm_read_ptx_sreg_tid_x` builtin as `int`). Launch via
`tinygrad.runtime.autogen.cuda` ctypes. 5060 Ti = sm_120. Pointer:
`t._buffer()._buf.value`.

## Port status (2026-08-27, DONE)

`gabion/user_models/qwen35_text.py` has been **ported to the fast design** — the
naive per-layer-dequant + numpy-scan path is gone. What's in the adapter now:

- Persistent u8 weights (`_QWeight` IQ4_NL/Q5_K, `_KQuant` Q4_K/Q6_K for
  emb/head) + the fused IQ4_NL GEMV custom kernel (`_mm` → `_QWeight.gemv`,
  f32 always fused; f16 fused when `QWEN35_FUSE_F16=auto` and GGUF ≥ 8GiB).
- GDN scan + full attention run **in-graph on the GPU** (T=1 step, state via
  `uop.store` inside the TinyJit) — no numpy einsum, no per-op ping-pong.
- Rows processed one token at a time through one TinyJit per entry point
  (`ids`/`hid`/`log`); prefill and decode both use the fast T=1 path.
- API/mesh plumbing unchanged (`from_gguf_shard` byte-balanced shards,
  `new_state`/`stream_state`/`release_stream`, the three `forward_shard_*`
  methods, `sample_batch`/`loss`). One active GPU state per adapter (the mesh
  pipeline is serialized per prompt; do not interleave streams).
- Tied lm_head works (each shard builds the emb matrix locally from the mmap).
- Two latent bugs fixed while porting: `_raw_blocks` treated Q8_0 as a
  256-block type (it's 32), and the ported full-attn layer initially missed
  the residual+FFN wrapper.

Verified: fixture tests 3/3 (split-vs-split parity, state, IQ4_XS dequant);
**0.8B oracle-identical `[248068, 198, 90700, 8340, 25, 271]` at ~10-15 tok/s
per full step** (fused f32, f16 matmuls unfused — same as the runner's
oracle-safe config). 27B mesh-shape verification + bench numbers: see below.

**Measurement standard: `docs/qwen35-benchmark.md` + `tools/bench_qwen35_pipeline.py`**
(single process, both shards via `device="CUDA:0"/"CUDA:1"`; correctness gate,
decode median s/tok, prefill tok/s at 64/512, VRAM). Adapter accepts an
explicit `device=` (or `QWEN35_DEVICE=CUDA:<idx>`) — tinygrad 0.14 ignores
`HCQ_VISIBLE_DEVICES` and `DEV=CUDA:1` for device selection.

## Web deployment: single-process dual-device mesh worker (2026-08-27, RUNNING)

The two-process mesh (one worker per shard) **cannot run the 27B on this
system**: each process OOMs at ~7.4GB (`MemoryError: Allocation of 256 B
failed on CUDA. Used: 7.29 GB`) — a WDDM per-context wall, while one process
can allocate ~15GB per GPU. The web deployment therefore uses
**`D:/tmp/_q35_dual_worker.py`**: ONE process builds both shard adapters
(`from_gguf_shard(..., device="CUDA:0")` / `device="CUDA:1")`) and registers
as BOTH pebble workers (`q35-s0`, `q35-s1`). The server still orchestrates
shard0 → hidden → shard1 → logits over websockets.

- Measured: GPU0 10.4 GiB / GPU1 9.6 GiB in one process, load ~60s.
- Warm web decode ≈ **1.56 tok/s (0.64 s/tok)** at 30 tokens — the mesh adds
  a per-token websocket round-trip + f16 hidden encode/decode on top of the
  runner's 0.40 s/tok. 27B oracle prefix `[248068, 198, 760, 1156, 369, 9859]`
  matches through the full pipeline.
- **Pinned-pool bug found by this shape (fixed in the adapter):** each weight
  upload allocates a tinygrad pinned host staging buffer (`_copyin` →
  `BufferSpec(host=True)`) that is only released on stream sync. The port
  dropped the runner's per-weight `Device[dev].synchronize()` calls, so at 27B
  scale the WDDM pinned pool (~5.5GB) exhausted mid-load
  (`cuMemHostAlloc: CUDA Error 2`). Fix: sync after every layer + emb/head in
  `_load_gpu` (the runner syncs after every weight group).
- Launch: mesh server (`--max-rounds 0`) + `_q35_dual_worker.py` + a local
  chat page (`D:/tmp/_q35_chat_web.py`, proxies POST /infer same-origin).
  `server.py` caps `max_tokens` at 2048 (was 128 — too small for real answers).

## Recommendation for this adapter (resolved by the port)

- ✅ Adopt the runner's design: persistent u8 weights + the fused IQ4_NL GEMV
  custom kernel for decode rows; GDN scan and hidden-state math on the GPU in
  f32; no per-op numpy round-trips.
- ✅ The mesh pipeline (hidden across the websocket, state keyed by
  stream_id) is the right shape — the layer-local dequant was the problem.
- Contact: the runner + all harnesses are in `D:/tmp/`; the Hermes skill
  `gabion-tinygrad-inference` is the living knowledge base.
