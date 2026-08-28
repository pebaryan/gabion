"""Qwen3.8/Qwen3.5 hybrid text inference for CUDA pebble workers.

NOTE: read ``docs/qwen35-inference-status.md`` FIRST — this adapter is the
ported version of the proven standalone runner (``D:/tmp/_q27_run.py``).
The previous per-layer-dequant + numpy-scan design measured ~35x slower
(14.2s/tok shard-0 vs 0.40s/tok) and was replaced.  Techniques ported:

- weights stay QUANTIZED as persistent VRAM u8 tensors (IQ4_NL / Q5_K);
  IQ4_NL single-row matmuls go through a fused dequant-GEMV custom kernel
  (``_QWeight.gemv``) so no f16 matrix is ever materialized for decode.
  The 27B ``ssm_out`` Q5_K GEMV uses the same fused treatment by default.
- the GatedDeltaNet recurrent scan runs IN-GRAPH on the GPU (T=1 scan step,
  state tensors updated via ``uop.store`` inside the TinyJit graph).
- the large Q6_K output head uses a fused dequant-GEMV for greedy argmax;
  the chunked dequant path remains available for full-logit callers.
- decode rows are processed one token at a time through one TinyJit per entry
  point; native prefill uses a bounded batch of four tokens and returns only
  the final row's argmax. There is no per-op Tensor<->numpy ping-pong.
- token_embd / output weights use KQuant (u8 + on-device dequant) so the
  ~2.5GB f16 copies never live in VRAM.

API / mesh plumbing is unchanged: shard ranges, ``new_state`` /
``stream_state`` / ``release_stream``, ``forward_shard_ids_to_hidden`` /
``forward_shard_hidden_to_hidden`` / ``forward_shard_hidden_to_logits``.
The adapter keeps ONE active GPU state (conv/rec/KV are per-layer tensors
mutated in-graph) — the mesh pipeline is serialized per prompt, so use one
stream at a time.  State is zeroed automatically in-graph whenever the
stream position is 0 (reset), matching the runner's semantics.

Qwen3.8-27B uses the ``qwen35`` GGUF layout: most layers are GatedDeltaNet
linear-attention layers, every fourth layer is full attention, and the final
layer is also full attention.
"""
from __future__ import annotations

import math
import mmap
import os
from pathlib import Path
from typing import Any

import numpy as np

# The raw batch-4 IQ4 kernel is compiled with NVRTC. tinygrad loads its CUDA
# compiler support lazily, so point it at the standard Windows CUDA toolkit
# before importing tinygrad. ``auto`` enables it when that runtime is present.
_iq4_raw_mode = os.environ.get("QWEN35_IQ4_RAW_TILE4", "auto").lower()
_IQ4_RAW_TILE4_ENABLED = _iq4_raw_mode != "0"
if os.name == "nt" and _IQ4_RAW_TILE4_ENABLED:
    _cuda_root = Path(os.environ.get("CUDA_PATH", ""))
    if not _cuda_root.is_dir():
        _cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9")
    _nvrtc = _cuda_root / "bin" / "nvrtc64_120_0.dll"
    if _nvrtc.is_file():
        os.environ["CUDA_PATH"] = str(_cuda_root)
        os.environ.setdefault("NVRTC_PATH", str(_nvrtc))
    elif _iq4_raw_mode == "auto":
        _IQ4_RAW_TILE4_ENABLED = False
elif os.name != "nt" and _iq4_raw_mode == "auto":
    _IQ4_RAW_TILE4_ENABLED = False

try:
    from tinygrad import Tensor, dtypes, TinyJit
    from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo, ProgramInfo
    from tinygrad.llm.model import precompute_freqs_cis, apply_rope
except ImportError:  # keep the module importable in pure-meta contexts
    Tensor = dtypes = TinyJit = UOp = None  # type: ignore
    Ops = AxisType = KernelInfo = ProgramInfo = None  # type: ignore
    precompute_freqs_cis = apply_rope = None  # type: ignore


_BLOCK_BYTES = {
    "Q2_K": 84,
    "Q3_K": 110,
    "Q4_K": 144,
    "Q5_K": 176,
    "Q6_K": 210,
    "Q8_0": 34,
    "IQ4_NL": 18,
    "IQ4_XS": 136,
}
_GPU_BLOCK_TYPES = {"Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4_NL", "IQ4_XS"}


# ---------------------------------------------------------------------------
# Fused IQ4_NL dequant-GEMV custom kernel (ported from D:/tmp/_q27_run.py,
# parity-verified: 1.7e-6 f32, 1 ULP f16 vs the dequant+matmul reference).
#   out[o] = sum_{b=0..B-1} sum_{j=0..15} (KVAL[qs[o*B+b,j]&0xF]*x[b*32+j]
#                                          + KVAL[qs[o*B+b,j]>>4]*x[b*32+16+j]) * sc[o*B+b]
#   ggml layout: low nibble -> weight j, high nibble -> weight j+16.
#   M=1 only (decode). Ranges: (g, t) ended; REDUCE ranges (b, j) NOT in the
#   end() (waitlist-deadlock lesson). num_axes=0 reduce (scalar loads).
# ---------------------------------------------------------------------------
_IQ4NL_FXN: dict = {}
_IQ4NL_GEMM_FXN: dict = {}
_IQ4NL_RAW_TILE4_PROGRAM: dict = {}
_IQ4NL_RAW_GEMV_PROGRAM: dict = {}
_Q5K_GEMV_FXN: dict = {}
_Q5K_GEMM_FXN: dict = {}
_Q6K_GEMV_FXN: dict = {}


def build_iq4nl_gemv(out_dim: int, in_dim: int, name: str, local: int = 256) -> callable:
    B = in_dim // 32
    assert out_dim % local == 0 and in_dim % 32 == 0

    def fxn(out: UOp, x: UOp, qs: UOp, sc: UOp, ks: UOp) -> UOp:
        g = UOp.range(out_dim // local, 0)               # GLOBAL grid
        t = UOp.range(local, 1, AxisType.LOCAL)          # threads per block
        oo = g * local + t                               # output index
        b = UOp.range(B, 2, AxisType.REDUCE)             # 32-weight blocks
        j = UOp.range(16, 3, AxisType.REDUCE)            # nibble bytes
        s = sc.index(oo * B + b).load()
        byte = qs.index(oo * B + b, j).load()
        vlo = ks.index((byte & UOp.const(0xF, dtypes.uint8)).cast(dtypes.int32)).load()
        vhi = ks.index((byte >> UOp.const(4, dtypes.uint8)).cast(dtypes.int32)).load()
        xlo = x.index(b * 32 + j).load()
        xhi = x.index(b * 32 + 16 + j).load()
        if x.dtype == dtypes.float16:
            # Match the f16 dequant path exactly: weights are KVAL*sc ROUNDED
            # to f16 first; products then computed in f32 like the matmul.
            wlo = (vlo * s).cast(dtypes.float16).float()
            whi = (vhi * s).cast(dtypes.float16).float()
            contrib = wlo * xlo.float() + whi * xhi.float()
        else:
            contrib = (vlo * xlo + vhi * xhi) * s
        acc = contrib.reduce(b, j, arg=(Ops.ADD, 0))
        return out.index(oo).store(acc).end(g, t).sink(arg=KernelInfo(name=name, opts_to_apply=()))
    return fxn


def build_iq4nl_gemv_u32(out_dim: int, in_dim: int, name: str, local: int = 32) -> callable:
    """u32-load variant of the fused IQ4_NL GEMV: quant bytes are loaded as
    uint32 words (4 bytes per load) and unpacked with shifts in-kernel.
    Bit-exact vs the byte-load kernel (f32 and f16 paths) and measured 2.65x
    faster per replayed GEMV (87 vs 232 us at 5120x5120, 2026-08-28)."""
    B = in_dim // 32
    assert out_dim % local == 0 and in_dim % 32 == 0

    def fxn(out: UOp, x: UOp, qs32T: UOp, scT: UOp, ks: UOp) -> UOp:
        g = UOp.range(out_dim // local, 0)
        t = UOp.range(local, 1, AxisType.LOCAL)
        oo = g * local + t
        b = UOp.range(B, 2, AxisType.REDUCE)
        jw = UOp.range(4, 3, AxisType.REDUCE)
        # Transposed [b, w, oo] weights: adjacent threads read adjacent words.
        s = scT.index(b, oo).load()
        word = qs32T.index(b * 4 + jw, oo).load()
        terms = []
        for bi in range(4):
            byte = (word >> UOp.const(8 * bi, dtypes.uint32)) & UOp.const(0xFF, dtypes.uint32)
            vlo = ks.index((byte & UOp.const(0xF, dtypes.uint32)).cast(dtypes.int32)).load()
            vhi = ks.index((byte >> UOp.const(4, dtypes.uint32)).cast(dtypes.int32)).load()
            j = jw * 4 + bi
            xlo = x.index(b * 32 + j).load()
            xhi = x.index(b * 32 + 16 + j).load()
            if x.dtype == dtypes.float16:
                wlo = (vlo * s).cast(dtypes.float16).float()
                whi = (vhi * s).cast(dtypes.float16).float()
                terms.append(wlo * xlo.float() + whi * xhi.float())
            else:
                terms.append((vlo * xlo + vhi * xhi) * s)
        acc = terms[0]
        for term in terms[1:]:
            acc = acc + term
        return out.index(oo).store(acc.reduce(b, jw, arg=(Ops.ADD, 0))).end(g, t).sink(
            arg=KernelInfo(name=name, opts_to_apply=()))
    return fxn


def iq4nl_gemv(x: Tensor, u8: Tensor, sc: Tensor, ks: Tensor, out_dim: int, in_dim: int, dev: str,
               u32t: Tensor | None = None) -> Tensor:
    """Fused dequant-GEMV for one row x (in_dim,) -> (1, 1, out_dim)."""
    # Diagnostic only (QWEN35_IQ4_RAW_GEMV=1): the raw warp-per-row GEMV is
    # faster standalone but raw PROGRAMs are not batched into the TinyJit CUDA
    # graph, so at T=1 the per-launch WDDM overhead regresses decode (0.3316
    # vs 0.2783 s/tok, measured 2026-08-28). The UOp kernel remains default.
    raw_mode = os.environ.get("QWEN35_IQ4_RAW_GEMV", "0").lower()
    if (raw_mode == "1"
            and str(dev).startswith("CUDA") and out_dim % 4 == 0 and in_dim % 32 == 0):
        return iq4nl_raw_gemv(x, u8, sc, ks, out_dim, in_dim, dev)
    was_f16 = x.dtype == dtypes.float16
    # Uniform one-warp scheduling is the measured full-pipeline winner on the
    # canonical 27B dual-5060-Ti setup.  ``auto`` remains available for shape
    # diagnostics and other Qwen35 variants.
    local_env = os.environ.get("QWEN35_IQ4_LOCAL", "32").lower()
    if local_env != "auto":
        local = int(local_env)
    elif x.dtype == dtypes.float16:
        # The 27B GDN projections have a different occupancy sweet spot from
        # the full-attention/FFN f32 path.  Keep the old 256 fallback for
        # shapes from other Qwen35 variants.
        local = {
            (5120, 17408): 32,  # ffn_down when a variant keeps the input f16
            (6144, 5120): 32,   # GDN gate
            (1024, 5120): 32,   # GDN alpha/beta
            (12288, 5120): 64,  # full-attention q in an f16 caller
            (5120, 6144): 32,
        }.get((out_dim, in_dim), 256)
    else:
        local = {
            (17408, 5120): 32,  # FFN gate/up
            (5120, 17408): 64,  # FFN down
            (6144, 5120): 64,   # full-attention gate
            (12288, 5120): 128, # full-attention q
            (1024, 5120): 32,   # full-attention k/v
            (5120, 6144): 32,   # full-attention output
        }.get((out_dim, in_dim), 256)
    key = (out_dim, in_dim, x.dtype, local)
    fxn = _IQ4NL_FXN.get(key)
    if fxn is None:
        fxn = build_iq4nl_gemv_u32(out_dim, in_dim, f"iq4nl_u32T_{out_dim}_{in_dim}_l{local}", local=local)
        _IQ4NL_FXN[key] = fxn
    out = Tensor.empty(out_dim, device=dev)
    res = out.custom_kernel(x.reshape(in_dim), u32t, sc, ks, fxn=fxn)[0].reshape(1, 1, out_dim)
    return res.cast(dtypes.float16) if was_f16 else res


def build_iq4nl_gemm(out_dim: int, in_dim: int, tokens: int, name: str, local: int = 32) -> callable:
    """Build a fused IQ4_NL matrix multiply for a small prefill batch."""
    B = in_dim // 32
    assert out_dim % local == 0 and in_dim % 32 == 0

    def fxn(out: UOp, x: UOp, qs32T: UOp, scT: UOp, ks: UOp) -> UOp:
        g = UOp.range(tokens * out_dim // local, 0)
        t = UOp.range(local, 1, AxisType.LOCAL)
        flat = g * local + t
        tok, oo = flat // out_dim, flat % out_dim
        b = UOp.range(B, 2, AxisType.REDUCE)
        jw = UOp.range(4, 3, AxisType.REDUCE)
        # Transposed [b, w, oo] weights + u32 loads (bit-exact vs byte loads).
        s = scT.index(b, oo).load()
        word = qs32T.index(b * 4 + jw, oo).load()
        terms = []
        for bi in range(4):
            byte = (word >> UOp.const(8 * bi, dtypes.uint32)) & UOp.const(0xFF, dtypes.uint32)
            vlo = ks.index((byte & UOp.const(0xF, dtypes.uint32)).cast(dtypes.int32)).load()
            vhi = ks.index((byte >> UOp.const(4, dtypes.uint32)).cast(dtypes.int32)).load()
            j = jw * 4 + bi
            xlo = x.index(tok * in_dim + b * 32 + j).load()
            xhi = x.index(tok * in_dim + b * 32 + 16 + j).load()
            if x.dtype == dtypes.float16:
                wlo = (vlo * s).cast(dtypes.float16)
                whi = (vhi * s).cast(dtypes.float16)
                terms.append((wlo * xlo).cast(dtypes.float32) + (whi * xhi).cast(dtypes.float32))
            else:
                terms.append((vlo * xlo + vhi * xhi) * s)
        contrib = terms[0]
        for term in terms[1:]:
            contrib = contrib + term
        return out.index(flat).store(contrib.reduce(b, jw, arg=(Ops.ADD, 0))).end(g, t).sink(
            arg=KernelInfo(name=name, opts_to_apply=()))
    return fxn


def iq4nl_gemm(x: Tensor, w: "_QWeight", tokens: int) -> Tensor:
    """Fused IQ4_NL GEMM for (1, tokens, in_dim) prefill input."""
    out_dim, in_dim = w.shape
    if (2 <= tokens <= 16 and x.dtype == dtypes.float32 and str(w.dev).startswith("CUDA")
            and _IQ4_RAW_TILE4_ENABLED):
        return iq4nl_raw_tile4(x, w, tokens)
    local = int(os.environ.get("QWEN35_IQ4_LOCAL", "32"))
    key = (out_dim, in_dim, tokens, x.dtype, local)
    fxn = _IQ4NL_GEMM_FXN.get(key)
    if fxn is None:
        fxn = build_iq4nl_gemm(out_dim, in_dim, tokens,
                               f"iq4nl_gemm_{out_dim}_{in_dim}_t{tokens}_l{local}", local=local)
        _IQ4NL_GEMM_FXN[key] = fxn
    out = Tensor.empty(tokens * out_dim, device=w.dev)
    res = out.custom_kernel(x.reshape(tokens * in_dim), w.u832T, w.scT, w.ks, fxn=fxn)[0]
    res = res.reshape(1, tokens, out_dim)
    return res.cast(dtypes.float16) if x.dtype == dtypes.float16 else res


def _build_iq4nl_raw_tile_program(dev: str, out_dim: int, in_dim: int, tokens: int) -> UOp:
    """Compile a replayable raw CUDA batch-T IQ4 projection program.

    One 256-thread block computes one output row; each dequantized weight is
    retained across all T token accumulators, so the u8 weight traffic is
    amortized T-fold (the prefill win: prefill is weight-bandwidth-bound)."""
    from tinygrad import Device
    from tinygrad.helpers import Target
    from tinygrad.renderer import Estimates
    from tinygrad.renderer.cstyle import CUDARenderer

    cuda = Device[dev]
    compile_target = Target(device="CUDA", renderer="CLANG", arch=cuda.arch)
    T = tokens
    assert out_dim % 128 == 0, f"raw tile needs out_dim % 128 == 0, got {out_dim}"
    accs = "".join(f"  float a{t} = 0.0f;\n" for t in range(T))
    fmas = "".join(f"        a{t} += wl * x[{t}*IN_DIM + k] + wh * x[{t}*IN_DIM + k + 16];\n"
                   for t in range(T))
    outs = "".join(f"  out[{t}*OUT_DIM + oo] = a{t};\n" for t in range(T))
    # One thread per output row so adjacent threads read adjacent words of the
    # transposed [b, w, oo] layout (coalesced), while each dequantized weight
    # is still reused across all T token accumulators.
    source = (r'''
extern "C" __global__ void __launch_bounds__(128) iq4_raw_tile(
    float *out, const float *x, const unsigned int *qsT,
    const float *scT, const float *ks) {
  __shared__ float ksm[16];
  if (threadIdx.x < 16) ksm[threadIdx.x] = ks[threadIdx.x];
  __syncthreads();
  const int oo = blockIdx.x * 128 + threadIdx.x;
  const int B = IN_DIM / 32;
ACCS
  for (int b = 0; b < B; ++b) {
    const float s = scT[b * OUT_DIM + oo];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const unsigned int word = qsT[(b * 4 + w) * OUT_DIM + oo];
#pragma unroll
      for (int bi = 0; bi < 4; ++bi) {
        const unsigned int v = (word >> (8 * bi)) & 0xFFu;
        const float wl = ksm[v & 15u] * s, wh = ksm[v >> 4] * s;
        const int k = b * 32 + w * 4 + bi;
FMAS
      }
    }
  }
OUTS
}
'''.replace("ACCS", accs).replace("FMAS", fmas).replace("OUTS", outs)
      .replace("IN_DIM", str(in_dim)).replace("OUT_DIM", str(out_dim)))
    binary = CUDARenderer(compile_target).compiler.compile_cached(source)
    params = (
        UOp.param(0, dtypes.float32, (1, T, out_dim), device=dev),
        UOp.param(1, dtypes.float32, (1, T, in_dim), device=dev),
        UOp.param(2, dtypes.uint32, (in_dim // 32 * 4, out_dim), device=dev),
        UOp.param(3, dtypes.float32, (in_dim // 32, out_dim), device=dev),
        UOp.param(4, dtypes.float32, (16,), device=dev),
    )
    sink = UOp(Ops.SINK, arg=KernelInfo(name="iq4_raw_tile", estimates=Estimates()))
    linear = UOp(Ops.LINEAR, src=params)
    actual_target = Target(device="CUDA", arch=cuda.arch)
    info = ProgramInfo("iq4_raw_tile", (out_dim // 128, 1, 1), (128, 1, 1), (),
                       (0, 1, 2, 3, 4), (0,), (1, 2, 3, 4), actual_target)
    return UOp(Ops.PROGRAM, src=(sink, linear, UOp(Ops.SOURCE, arg=source),
                                UOp(Ops.BINARY, dtypes.uchar, arg=binary)), arg=info)


def iq4nl_raw_tile4(x: Tensor, w: "_QWeight", tokens: int = 4) -> Tensor:
    """Raw CUDA tiled IQ4 projection for a small float32 prefill batch."""
    out_dim, in_dim = w.shape
    key = (w.dev, out_dim, in_dim, tokens)
    program = _IQ4NL_RAW_TILE4_PROGRAM.get(key)
    if program is None:
        program = _build_iq4nl_raw_tile_program(w.dev, out_dim, in_dim, tokens)
        _IQ4NL_RAW_TILE4_PROGRAM[key] = program
    out = Tensor.empty(1, tokens, out_dim, device=w.dev)
    call = program.call(out.uop, x.reshape(1, tokens, in_dim).uop, w.u832T.uop, w.scT.uop, w.ks.uop)
    return Tensor(out.uop.after(call))


def _build_iq4nl_raw_gemv_program(dev: str, out_dim: int, in_dim: int, f16: bool) -> UOp:
    """Compile a replayable raw CUDA warp-per-row IQ4 decode GEMV.

    One warp reduces one output row with vectorized 16-byte block loads
    (uint4 = one whole IQ4_NL block of quant bytes), the KVALUES table in
    shared memory, and a warp-shuffle reduction.  The f16 variant replicates
    the reference rounding exactly: weight = f16(KVAL*sc), product in f32.
    """
    from tinygrad import Device
    from tinygrad.helpers import Target
    from tinygrad.renderer import Estimates
    from tinygrad.renderer.cstyle import CUDARenderer

    cuda = Device[dev]
    compile_target = Target(device="CUDA", renderer="CLANG", arch=cuda.arch)
    if f16:
        body = r'''
      const float wl = h2f(f2h(ksm[v & 15] * s)), wh = h2f(f2h(ksm[v >> 4] * s));
      acc += wl * h2f(x[k + j]) + wh * h2f(x[k + 16 + j]);
'''
    else:
        body = r'''
      acc += (ksm[v & 15] * x[k + j] + ksm[v >> 4] * x[k + 16 + j]) * s;
'''
    source = (r'''
__device__ __forceinline__ float h2f(unsigned short h) {
  float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ unsigned short f2h(float f) {
  unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
extern "C" __global__ void __launch_bounds__(128) iq4_raw_gemv(
    float *out, const XTYPE *x, const unsigned char *qs,
    const float *sc, const float *ks) {
  __shared__ float ksm[16];
  if (threadIdx.x < 16) ksm[threadIdx.x] = ks[threadIdx.x];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int o = blockIdx.x * 4 + warp;
  const int B = IN_DIM / 32;
  float acc = 0.0f;
  for (int b = lane; b < B; b += 32) {
    const uint4 q = ((const uint4 *)qs)[o * B + b];
    const float s = sc[o * B + b];
    const unsigned char *pb = (const unsigned char *)&q;
    const int k = b * 32;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const unsigned char v = pb[j];
BODY
    }
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) out[o] = acc;
}
'''.replace("BODY", body)
      .replace("XTYPE", "unsigned short" if f16 else "float")
      .replace("IN_DIM", str(in_dim)))
    binary = CUDARenderer(compile_target).compiler.compile_cached(source)
    params = (
        UOp.param(0, dtypes.float32, (out_dim,), device=dev),
        UOp.param(1, dtypes.float16 if f16 else dtypes.float32, (in_dim,), device=dev),
        UOp.param(2, dtypes.uint8, (out_dim * in_dim // 32, 16), device=dev),
        UOp.param(3, dtypes.float32, (out_dim * in_dim // 32,), device=dev),
        UOp.param(4, dtypes.float32, (16,), device=dev),
    )
    sink = UOp(Ops.SINK, arg=KernelInfo(name="iq4_raw_gemv", estimates=Estimates()))
    linear = UOp(Ops.LINEAR, src=params)
    actual_target = Target(device="CUDA", arch=cuda.arch)
    info = ProgramInfo("iq4_raw_gemv", (out_dim // 4, 1, 1), (128, 1, 1), (),
                       (0, 1, 2, 3, 4), (0,), (1, 2, 3, 4), actual_target)
    return UOp(Ops.PROGRAM, src=(sink, linear, UOp(Ops.SOURCE, arg=source),
                                UOp(Ops.BINARY, dtypes.uchar, arg=binary)), arg=info)


def iq4nl_raw_gemv(x: Tensor, u8: Tensor, sc: Tensor, ks: Tensor,
                   out_dim: int, in_dim: int, dev: str) -> Tensor:
    """Raw CUDA warp-per-row IQ4 GEMV for one decode row (f32 or f16 input)."""
    f16 = x.dtype == dtypes.float16
    key = (dev, out_dim, in_dim, f16)
    program = _IQ4NL_RAW_GEMV_PROGRAM.get(key)
    if program is None:
        program = _build_iq4nl_raw_gemv_program(dev, out_dim, in_dim, f16)
        _IQ4NL_RAW_GEMV_PROGRAM[key] = program
    out = Tensor.empty(out_dim, device=dev)
    call = program.call(out.uop, x.reshape(in_dim).uop, u8.uop, sc.uop, ks.uop)
    res = Tensor(out.uop.after(call)).reshape(1, 1, out_dim)
    return res.cast(dtypes.float16) if f16 else res


def build_q5k_gemv(out_dim: int, in_dim: int, name: str, local: int = 256) -> callable:
    """Build a fused Q5_K dequant-GEMV for one decode row."""
    B = in_dim // 256
    assert out_dim % local == 0 and in_dim % 256 == 0

    def fxn(out: UOp, x: UOp, qs32: UOp, qh32: UOp, sc: UOp, mn: UOp, d: UOp, dmin: UOp) -> UOp:
        g = UOp.range(out_dim // local, 0)
        t = UOp.range(local, 1, AxisType.LOCAL)
        oo = g * local + t
        b = UOp.range(B, 2, AxisType.REDUCE)
        jw = UOp.range(8, 3, AxisType.REDUCE)
        terms = []
        for sub in range(8):
            # u32 quant loads + shift unpack (bit-exact vs byte loads)
            qsw = qs32.index(oo * B + b, (sub // 2) * 8 + jw).load()
            qhw = qh32.index(oo * B + b, jw).load()
            for bi in range(4):
                byte = (qsw >> UOp.const(8 * bi, dtypes.uint32)) & UOp.const(0xFF, dtypes.uint32)
                nib = (byte >> UOp.const((sub & 1) * 4, dtypes.uint32)) & UOp.const(0xF, dtypes.uint32)
                hbyte = (qhw >> UOp.const(8 * bi, dtypes.uint32)) & UOp.const(0xFF, dtypes.uint32)
                high = (hbyte >> UOp.const(sub, dtypes.uint32)) & UOp.const(1, dtypes.uint32)
                qv = nib | (high << UOp.const(4, dtypes.uint32))
                xv = x.index(b * 256 + sub * 32 + jw * 4 + bi).load()
                wv = (d.index(oo * B + b).load() * sc.index(oo * B + b, sub).load() * qv.cast(dtypes.float32)
                      - dmin.index(oo * B + b).load() * mn.index(oo * B + b, sub).load()).cast(dtypes.float16)
                # The reference path is (f16 x f16) -> f16 product, then an f32
                # reduction.  Keeping that order avoids unnecessary numerical
                # drift in the recurrent stack.  For f32 callers retain the
                # natural f32 product used by the standalone GEMV test.
                terms.append((wv * xv).cast(dtypes.float32) if x.dtype == dtypes.float16
                             else wv.float() * xv.cast(dtypes.float32))
        acc = terms[0]
        for term in terms[1:]:
            acc = acc + term
        return out.index(oo).store(acc.reduce(b, jw, arg=(Ops.ADD, 0))).end(g, t).sink(
            arg=KernelInfo(name=name, opts_to_apply=()))
    return fxn


def q5k_gemv(x: Tensor, w: "_QWeight") -> Tensor:
    was_f16 = x.dtype == dtypes.float16
    out_dim, in_dim = w.shape
    local = int(os.environ.get("QWEN35_Q5_LOCAL", "16"))
    key = (out_dim, in_dim, x.dtype, local)
    fxn = _Q5K_GEMV_FXN.get(key)
    if fxn is None:
        fxn = build_q5k_gemv(out_dim, in_dim, f"q5k_{out_dim}_{in_dim}_l{local}", local=local)
        _Q5K_GEMV_FXN[key] = fxn
    out = Tensor.empty(out_dim, device=w.dev)
    res = out.custom_kernel(x.reshape(in_dim), w.qs32, w.qh32,
                            w.sc, w.mn, w.d, w.dmin, fxn=fxn)[0]
    res = res.reshape(1, 1, out_dim)
    return res.cast(dtypes.float16) if was_f16 else res


def build_q5k_gemm(out_dim: int, in_dim: int, tokens: int, name: str, local: int = 16) -> callable:
    """Build a fused Q5_K matrix multiply for a small prefill batch."""
    B = in_dim // 256
    assert out_dim % local == 0 and in_dim % 256 == 0

    def fxn(out: UOp, x: UOp, qs32: UOp, qh32: UOp, sc: UOp, mn: UOp, d: UOp, dmin: UOp) -> UOp:
        g = UOp.range(tokens * out_dim // local, 0)
        t = UOp.range(local, 1, AxisType.LOCAL)
        flat = g * local + t
        tok, oo = flat // out_dim, flat % out_dim
        b = UOp.range(B, 2, AxisType.REDUCE)
        jw = UOp.range(8, 3, AxisType.REDUCE)
        terms = []
        for sub in range(8):
            qsw = qs32.index(oo * B + b, (sub // 2) * 8 + jw).load()
            qhw = qh32.index(oo * B + b, jw).load()
            for bi in range(4):
                byte = (qsw >> UOp.const(8 * bi, dtypes.uint32)) & UOp.const(0xFF, dtypes.uint32)
                nib = (byte >> UOp.const((sub & 1) * 4, dtypes.uint32)) & UOp.const(0xF, dtypes.uint32)
                hbyte = (qhw >> UOp.const(8 * bi, dtypes.uint32)) & UOp.const(0xFF, dtypes.uint32)
                high = (hbyte >> UOp.const(sub, dtypes.uint32)) & UOp.const(1, dtypes.uint32)
                qv = nib | (high << UOp.const(4, dtypes.uint32))
                xv = x.index(tok * in_dim + b * 256 + sub * 32 + jw * 4 + bi).load()
                wv = (d.index(oo * B + b).load() * sc.index(oo * B + b, sub).load() * qv.cast(dtypes.float32)
                      - dmin.index(oo * B + b).load() * mn.index(oo * B + b, sub).load()).cast(dtypes.float16)
                terms.append((wv * xv).cast(dtypes.float32) if x.dtype == dtypes.float16
                             else wv.float() * xv.cast(dtypes.float32))
        acc = terms[0]
        for term in terms[1:]:
            acc = acc + term
        return out.index(flat).store(acc.reduce(b, jw, arg=(Ops.ADD, 0))).end(g, t).sink(
            arg=KernelInfo(name=name, opts_to_apply=()))
    return fxn


def q5k_gemm(x: Tensor, w: "_QWeight", tokens: int) -> Tensor:
    """Fused Q5_K GEMM for (1, tokens, in_dim) prefill input."""
    out_dim, in_dim = w.shape
    local = int(os.environ.get("QWEN35_Q5_LOCAL", "16"))
    key = (out_dim, in_dim, tokens, x.dtype, local)
    fxn = _Q5K_GEMM_FXN.get(key)
    if fxn is None:
        fxn = build_q5k_gemm(out_dim, in_dim, tokens,
                             f"q5k_gemm_{out_dim}_{in_dim}_t{tokens}_l{local}", local=local)
        _Q5K_GEMM_FXN[key] = fxn
    out = Tensor.empty(tokens * out_dim, device=w.dev)
    res = out.custom_kernel(x.reshape(tokens * in_dim), w.qs32, w.qh32,
                            w.sc, w.mn, w.d, w.dmin, fxn=fxn)[0]
    res = res.reshape(1, tokens, out_dim)
    return res.cast(dtypes.float16) if x.dtype == dtypes.float16 else res


def build_q6k_gemv(out_dim: int, in_dim: int, name: str, local: int = 256) -> callable:
    """Build the fused Q6_K output-head GEMV used by single-token greedy decode."""
    B = in_dim // 256
    assert out_dim % local == 0 and in_dim % 256 == 0

    def fxn(out: UOp, x: UOp, ql32: UOp, qh32: UOp, sc: UOp, d: UOp) -> UOp:
        g = UOp.range(out_dim // local, 0)
        t = UOp.range(local, 1, AxisType.LOCAL)
        oo = g * local + t
        b = UOp.range(B, 2, AxisType.REDUCE)
        jw = UOp.range(4, 3, AxisType.REDUCE)
        terms = []
        # Unroll h/group/half-of-the-32-byte scale selection.  The first
        # dynamic version passed Python parity but produced wrong CUDA values;
        # fixed offsets also keep this kernel friendly to tinygrad's indexer.
        # Quant bytes are loaded as u32 words and shift-unpacked (bit-exact).
        # Weights are in the transposed [b, w, oo] layout so that adjacent
        # threads (adjacent oo) read adjacent words -> coalesced loads.
        for h in range(2):
            for group in range(4):
                for half in range(2):
                    qlw = ql32.index(b * 32 + h * 16 + (group & 1) * 8 + half * 4 + jw, oo).load()
                    qhw = qh32.index(b * 16 + h * 8 + half * 4 + jw, oo).load()
                    for bi in range(4):
                        qlb = (qlw >> UOp.const(8 * bi, dtypes.uint32)) & UOp.const(0xFF, dtypes.uint32)
                        qhb = (qhw >> UOp.const(8 * bi, dtypes.uint32)) & UOp.const(0xFF, dtypes.uint32)
                        qv = ((qlb >> UOp.const((group // 2) * 4, dtypes.uint32)) & UOp.const(0xF, dtypes.uint32)) \
                            | (((qhb >> UOp.const(group * 2, dtypes.uint32)) & UOp.const(3, dtypes.uint32)) << UOp.const(4, dtypes.uint32))
                        scale = sc.index(b * 16 + h * 8 + group * 2 + half, oo).load()
                        xv = x.index(b * 256 + h * 128 + group * 32 + half * 16 + jw * 4 + bi).load().cast(dtypes.float32)
                        terms.append((qv.cast(dtypes.float32) - UOp.const(32.0, dtypes.float32)) * scale * d.index(b, oo).load() * xv)
        contrib = terms[0]
        for term in terms[1:]:
            contrib = contrib + term
        acc = contrib.reduce(b, jw, arg=(Ops.ADD, 0))
        return out.index(oo).store(acc).end(g, t).sink(arg=KernelInfo(name=name, opts_to_apply=()))
    return fxn


def q6k_gemv(x: Tensor, ql: Tensor, qh: Tensor, sc: Tensor, d: Tensor,
             out_dim: int, in_dim: int, dev: str) -> Tensor:
    local = int(os.environ.get("QWEN35_Q6_LOCAL", "256"))
    key = (out_dim, in_dim, x.dtype, local)
    fxn = _Q6K_GEMV_FXN.get(key)
    if fxn is None:
        fxn = build_q6k_gemv(out_dim, in_dim, f"q6k_{out_dim}_{in_dim}_l{local}", local=local)
        _Q6K_GEMV_FXN[key] = fxn
    out = Tensor.empty(out_dim, device=dev)
    return out.custom_kernel(x.reshape(in_dim), ql, qh,
                             sc, d, fxn=fxn)[0].reshape(1, 1, out_dim)


def _rms(x: Tensor, w: Tensor | None, eps: float) -> Tensor:
    x = x * (x.float().pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    return x * w if w is not None else x


# ---------------------------------------------------------------------------
# Persistent GPU weights
# ---------------------------------------------------------------------------

class _F16W:
    """CPU-dequantized-once f16/f32 tensor weight — uniform .dequant() interface."""

    def __init__(self, t: Tensor):
        self.t = t

    def dequant(self) -> Tensor:
        return self.t

    def __getitem__(self, idx):
        return self.t[idx]


class _QWeight:
    """IQ4_NL / Q5_K kept as persistent u8 VRAM; dequantized on-device per matmul.

    ``blk`` is the raw block bytes reshaped (nb, block_bytes) in GGUF order.
    ``shape`` is the numpy (out, in) shape.
    """

    def __init__(self, dev: str, gtype: str, blk: np.ndarray, shape: tuple):
        self.dev, self.gtype, self.shape = dev, gtype, shape
        n = int(np.prod(shape))
        if gtype == "IQ4_NL":
            nb = n // 32
            out_dim, in_dim = shape
            Bq = in_dim // 32
            raw = np.ascontiguousarray(blk[:, 2:18])                       # (nb, 16) nibbles
            sc = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)   # (nb,)
            # Transposed [b, w, oo] layout, output-row index fastest-varying.
            # Both the decode GEMV and the batched prefill kernels give one
            # thread one output row, so row-major put adjacent threads B*16
            # bytes apart (4 useful bytes per 32B transaction).  Transposing
            # makes those loads coalesced: measured 1.5-3.1x per GEMV across
            # the dominant 27B shapes, bit-exact.  Row-major views are rebuilt
            # lazily by __getattr__ for the dequant fallbacks so VRAM is flat.
            self.u832T = Tensor(
                np.ascontiguousarray(raw.view("<u4").reshape(out_dim, Bq, 4).transpose(1, 2, 0)).reshape(Bq * 4, out_dim),
                device=dev).realize()
            self.scT = Tensor(
                np.ascontiguousarray(sc.reshape(out_dim, Bq).transpose(1, 0)).reshape(Bq, out_dim),
                device=dev).realize()
            from tools.export_model import _IQ4_KVALUES
            self.ks = Tensor(np.array(_IQ4_KVALUES, dtype=np.float32).reshape(-1), device=dev).realize()
        elif gtype == "Q5_K":
            nb = n // 256
            d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
            dmin = blk[:, 2:4].copy().view("<f2")[:, 0].astype(np.float32)
            scales = blk[:, 4:16]
            sc = np.empty((nb, 8), dtype=np.float32)
            mn = np.empty((nb, 8), dtype=np.float32)
            for i in range(4):
                sc[:, i] = scales[:, i] & 63
                mn[:, i] = scales[:, i + 4] & 63
            sc[:, 4] = (scales[:, 8] & 0xF) | ((scales[:, 0] >> 6) << 4)
            mn[:, 4] = (scales[:, 8] >> 4) | ((scales[:, 4] >> 6) << 4)
            sc[:, 5] = (scales[:, 9] & 0xF) | ((scales[:, 1] >> 6) << 4)
            mn[:, 5] = (scales[:, 9] >> 4) | ((scales[:, 5] >> 6) << 4)
            sc[:, 6] = (scales[:, 10] & 0xF) | ((scales[:, 2] >> 6) << 4)
            mn[:, 6] = (scales[:, 10] >> 4) | ((scales[:, 6] >> 6) << 4)
            sc[:, 7] = (scales[:, 11] & 0xF) | ((scales[:, 3] >> 6) << 4)
            mn[:, 7] = (scales[:, 11] >> 4) | ((scales[:, 7] >> 6) << 4)
            # Q5_K stays ROW-MAJOR.  Transposing it the way IQ4_NL/Q6_K were
            # (see docs/qwen35-benchmark.md 2d) was measured twice and is a
            # net loss here: -0.9% decode for +3% prefill.  These kernels run
            # at local width 16 (half a warp), so coalescing buys less, while
            # row-major keeps a row's eight sub-blocks contiguous for the
            # sub-block loop.  Decode is the headline metric, so row-major
            # wins for Q5_K even though the same change is a large win for
            # the wider IQ4_NL and Q6_K kernels.
            # uint32 is ALWAYS the realized primary; u8 is the lazy bitcast
            # view for the dequant fallbacks.  This is not just a preference:
            # when qs32/qh32 were an unrealized bitcast view (the old path for
            # every shape except ssm_out), feeding them to the fused GEMV
            # inside a TinyJit produced correct standalone values but
            # degenerate output on replay — a stale captured buffer.  That,
            # not numerics, is why attn_qkv fusion "failed the gate": its
            # kernel is accurate to 7.5e-07 (better than ssm_out's 2.7e-06).
            qs_np = np.ascontiguousarray(blk[:, 48:176])
            qh_np = np.ascontiguousarray(blk[:, 16:48])
            self.qs32 = Tensor(qs_np.view("<u4"), device=dev).realize()
            self.qh32 = Tensor(qh_np.view("<u4"), device=dev).realize()
            self.qs = self.qs32.bitcast(dtypes.uint8)
            self.qh = self.qh32.bitcast(dtypes.uint8)
            self.d = Tensor(d, device=dev).realize()
            self.dmin = Tensor(dmin, device=dev).realize()
            self.sc = Tensor(sc, device=dev).realize()
            self.mn = Tensor(mn, device=dev).realize()
        else:
            raise ValueError(f"_QWeight: unsupported on-device type {gtype}")

    def __getattr__(self, name: str):
        # IQ4_NL keeps only the coalesced transposed layout; the row-major
        # views feed the dequant fallbacks and are rebuilt on first use.
        gt = self.__dict__.get("gtype")
        if name in ("u8", "u832", "sc") and gt == "IQ4_NL":
            self._build_rowmajor()
            return self.__dict__[name]
        raise AttributeError(name)

    def _build_rowmajor(self) -> None:
        out_dim, in_dim = self.shape
        Bq = in_dim // 32
        u832 = self.u832T.reshape(Bq, 4, out_dim).permute(2, 0, 1).reshape(out_dim * Bq, 4).contiguous().realize()
        self.__dict__["u832"] = u832
        self.__dict__["u8"] = u832.bitcast(dtypes.uint8)
        self.__dict__["sc"] = self.scT.reshape(Bq, out_dim).permute(1, 0).reshape(out_dim * Bq).contiguous().realize()

    def dequant(self) -> Tensor:
        n = int(np.prod(self.shape))
        if self.gtype == "IQ4_NL":
            nb = n // 32
            lo = self.u8 & 0x0F
            hi = (self.u8 >> 4) & 0x0F
            nib = Tensor.cat(lo, hi, dim=1).reshape(-1).cast(dtypes.int32)  # (n,)
            v = self.ks.gather(0, nib).reshape(nb, 32).float() * self.sc.reshape(nb, 1)
            return v.reshape(self.shape).cast(dtypes.float16)
        if self.gtype == "Q5_K":
            nb = n // 256
            qs, qh = self.qs, self.qh
            qs_lo = qs & 0x0F
            qs_hi = (qs >> 4) & 0x0F
            nib = Tensor.stack(qs_lo.reshape(nb, 4, 32), qs_hi.reshape(nb, 4, 32), dim=2).reshape(nb, 8, 32)
            hi_bits = ((qh.reshape(nb, 1, 32) >> Tensor.arange(8).to(self.dev).reshape(1, 8, 1)) & 1).reshape(nb, 8, 32)
            val = (nib | (hi_bits << 4)).float()
            dl = self.d.reshape(nb, 1, 1) * self.sc.reshape(nb, 8, 1)
            dm = self.dmin.reshape(nb, 1, 1) * self.mn.reshape(nb, 8, 1)
            return (dl * val - dm).reshape(self.shape).cast(dtypes.float16)
        raise AssertionError

    def dequant_f16(self) -> Tensor:
        """Dequantize directly in f16 for memory-constrained batched prefill."""
        n = int(np.prod(self.shape))
        if self.gtype == "IQ4_NL":
            nb = n // 32
            lo = self.u8 & 0x0F
            hi = (self.u8 >> 4) & 0x0F
            # The table and per-block scale are cast before multiplication so
            # the large intermediate never becomes an f32 full weight matrix.
            ks16 = self.ks.cast(dtypes.float16)
            vlo = ks16.gather(0, lo.reshape(-1).cast(dtypes.int32)).reshape(nb, 16)
            vlo.realize()
            vhi = ks16.gather(0, hi.reshape(-1).cast(dtypes.int32)).reshape(nb, 16)
            vhi.realize()
            v = Tensor.cat(vlo, vhi, dim=1)
            return (v * self.sc.cast(dtypes.float16).reshape(nb, 1)).reshape(self.shape)
        if self.gtype == "Q5_K":
            nb = n // 256
            qs, qh = self.qs, self.qh
            qs_lo = qs & 0x0F
            qs_hi = (qs >> 4) & 0x0F
            nib = Tensor.stack(qs_lo.reshape(nb, 4, 32), qs_hi.reshape(nb, 4, 32), dim=2).reshape(nb, 8, 32)
            hi_bits = ((qh.reshape(nb, 1, 32) >> Tensor.arange(8).to(self.dev).reshape(1, 8, 1)) & 1)
            val = (nib | (hi_bits << 4)).cast(dtypes.float16)
            dl = self.d.cast(dtypes.float16).reshape(nb, 1, 1) * self.sc.cast(dtypes.float16).reshape(nb, 8, 1)
            dm = self.dmin.cast(dtypes.float16).reshape(nb, 1, 1) * self.mn.cast(dtypes.float16).reshape(nb, 8, 1)
            return (dl * val - dm).reshape(self.shape)
        raise AssertionError

    def gemv(self, x: Tensor) -> Tensor:
        """Fused dequant-GEMV (M=1 only) for IQ4_NL and selected Q5_K."""
        out_dim, in_dim = self.shape
        if self.gtype == "IQ4_NL":
            return iq4nl_gemv(x, None, self.scT, self.ks, out_dim, in_dim, self.dev, u32t=self.u832T)
        assert self.gtype == "Q5_K"
        return q5k_gemv(x, self)


class _KQuant:
    """Q4_K / Q6_K weight kept as u8 in VRAM; dequant on-device per call.

    Used for token_embd (gather rows) and lm_head (fused dequant-matmul) so
    the f16 copies (~2.5GB each on the 27B) never live in VRAM.
    Block layout (numpy (vocab, D), D % 256 == 0): row v = B blocks of 256.
    """

    def __init__(self, dev: str, gtype: str, blk: np.ndarray, shape: tuple):
        vocab, D = shape
        assert D % 256 == 0, f"D={D} not a multiple of 256"
        self.dev, self.gtype, self.vocab, self.D = dev, gtype, vocab, D
        self.B = D // 256
        nb = vocab * self.B
        if gtype == "Q4_K":
            self.d = Tensor(blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32), device=dev).realize()
            self.dmin = Tensor(blk[:, 2:4].copy().view("<f2")[:, 0].astype(np.float32), device=dev).realize()
            self.qs = Tensor(blk[:, 16:], device=dev, dtype=dtypes.uint8).realize()       # (nb,128)
            self.scales = Tensor(blk[:, 4:16], device=dev, dtype=dtypes.uint8).realize()  # (nb,12)
        elif gtype == "Q6_K":
            # Transposed [b, w, oo] layout with the output-row index varying
            # fastest.  The fused head GEMV gives one thread one output row,
            # so row-major left adjacent threads B*128 bytes apart and every
            # 32B transaction returned 4 useful bytes.  Transposing makes the
            # loads fully coalesced: measured 11.04 -> 3.23 ms on the 27B head
            # (~117 -> ~400 GB/s), bit-exact (maxabs 0, same argmax).
            # Only this layout is stored; row-major views are rebuilt lazily
            # by __getattr__ for the diagnostic dequant fallbacks so VRAM does
            # not regress.
            ql_np = np.ascontiguousarray(blk[:, 0:128]).view("<u4").reshape(vocab, self.B, 32)
            qh_np = np.ascontiguousarray(blk[:, 128:192]).view("<u4").reshape(vocab, self.B, 16)
            sc_np = blk[:, 192:208].astype(np.int8).astype(np.float32).reshape(vocab, self.B, 16)
            d_np = blk[:, 208:210].copy().view("<f2")[:, 0].astype(np.float32).reshape(vocab, self.B)
            self.ql32T = Tensor(np.ascontiguousarray(ql_np.transpose(1, 2, 0)).reshape(self.B * 32, vocab), device=dev).realize()
            self.qh32T = Tensor(np.ascontiguousarray(qh_np.transpose(1, 2, 0)).reshape(self.B * 16, vocab), device=dev).realize()
            self.scT = Tensor(np.ascontiguousarray(sc_np.transpose(1, 2, 0)).reshape(self.B * 16, vocab), device=dev).realize()
            self.dT = Tensor(np.ascontiguousarray(d_np.transpose(1, 0)).reshape(self.B, vocab), device=dev).realize()
        else:
            raise ValueError(f"_KQuant: unsupported type {gtype}")

    def __getattr__(self, name: str):
        # Q6_K keeps only the coalesced transposed layout.  The row-major
        # views (used by the chunked-dequant fallbacks and by emb() when a
        # model ties a Q6_K head to the embedding) are rebuilt on first use.
        if name in ("ql", "qh", "sc", "d", "ql32", "qh32") and self.__dict__.get("gtype") == "Q6_K":
            self._build_rowmajor()
            return self.__dict__[name]
        raise AttributeError(name)

    def _build_rowmajor(self) -> None:
        B, vocab = self.B, self.vocab
        ql32 = self.ql32T.reshape(B, 32, vocab).permute(2, 0, 1).reshape(vocab * B, 32).contiguous().realize()
        qh32 = self.qh32T.reshape(B, 16, vocab).permute(2, 0, 1).reshape(vocab * B, 16).contiguous().realize()
        self.__dict__["ql32"] = ql32
        self.__dict__["qh32"] = qh32
        self.__dict__["ql"] = ql32.bitcast(dtypes.uint8)
        self.__dict__["qh"] = qh32.bitcast(dtypes.uint8)
        self.__dict__["sc"] = self.scT.reshape(B, 16, vocab).permute(2, 0, 1).reshape(vocab * B, 16).contiguous().realize()
        self.__dict__["d"] = self.dT.reshape(B, vocab).permute(1, 0).reshape(vocab * B).contiguous().realize()

    def _q4k_vals(self, qs, scales, d, dmin, nb2: int) -> Tensor:
        sc = Tensor.stack(
            scales[:, 0] & 63, scales[:, 1] & 63, scales[:, 2] & 63, scales[:, 3] & 63,
            (scales[:, 8] & 0xF) | ((scales[:, 0] >> 6) << 4),
            (scales[:, 9] & 0xF) | ((scales[:, 1] >> 6) << 4),
            (scales[:, 10] & 0xF) | ((scales[:, 2] >> 6) << 4),
            (scales[:, 11] & 0xF) | ((scales[:, 3] >> 6) << 4), dim=1).float()  # (nb2,8)
        mn = Tensor.stack(
            scales[:, 4] & 63, scales[:, 5] & 63, scales[:, 6] & 63, scales[:, 7] & 63,
            (scales[:, 8] >> 4) | ((scales[:, 4] >> 6) << 4),
            (scales[:, 9] >> 4) | ((scales[:, 5] >> 6) << 4),
            (scales[:, 10] >> 4) | ((scales[:, 6] >> 6) << 4),
            (scales[:, 11] >> 4) | ((scales[:, 7] >> 6) << 4), dim=1).float()  # (nb2,8)
        qs_lo = qs & 0x0F
        qs_hi = (qs >> 4) & 0x0F
        nib = Tensor.stack(qs_lo.reshape(nb2, 4, 32), qs_hi.reshape(nb2, 4, 32), dim=2).reshape(nb2, 8, 32).float()
        dl = d.reshape(nb2, 1, 1) * sc.reshape(nb2, 8, 1)
        dm = dmin.reshape(nb2, 1, 1) * mn.reshape(nb2, 8, 1)
        return (dl * nib - dm).reshape(nb2, 256)

    def _q6k_vals(self, ql, qh, sc, d, nb2: int) -> Tensor:
        parts = []
        for half, (qlh, qhh, sb) in enumerate([(ql[:, 0:64].reshape(nb2, 2, 32), qh[:, 0:32], 0),
                                               (ql[:, 64:128].reshape(nb2, 2, 32), qh[:, 32:64], 8)]):
            q1 = ((qlh[:, 0] & 0x0F) | ((qhh & 0x03) << 4)).float() - 32
            q2 = ((qlh[:, 1] & 0x0F) | (((qhh >> 2) & 0x03) << 4)).float() - 32
            q3 = ((qlh[:, 0] >> 4) | (((qhh >> 4) & 0x03) << 4)).float() - 32
            q4 = ((qlh[:, 1] >> 4) | (((qhh >> 6) & 0x03) << 4)).float() - 32
            for i, qv in enumerate((q1, q2, q3, q4)):
                sv = Tensor.cat(sc[:, sb + 2 * i].reshape(nb2, 1).expand(nb2, 16),
                                sc[:, sb + 2 * i + 1].reshape(nb2, 1).expand(nb2, 16), dim=1).reshape(nb2, 32)
                parts.append(d.reshape(nb2, 1) * sv * qv)
        return Tensor.cat(*parts, dim=1).reshape(nb2, 256)

    def _gather_rows(self, t: Tensor, idx: Tensor, row_len: int, nb2: int) -> Tensor:
        flat = t.reshape(-1)
        off = (idx.reshape(nb2, 1) * row_len + Tensor.arange(row_len).to(self.dev).reshape(1, row_len)).reshape(-1)
        return flat.gather(0, off.cast(dtypes.int32)).reshape(nb2, row_len)

    def emb(self, ids: Tensor) -> Tensor:
        """ids (1, T) int32 -> (1, T, D) f16."""
        T = ids.shape[1]
        nb2 = T * self.B
        idx = (ids.unsqueeze(-1) * self.B + Tensor.arange(self.B).to(self.dev).reshape(1, 1, self.B)).reshape(-1).cast(dtypes.int32)
        if self.gtype == "Q4_K":
            vals = self._q4k_vals(self._gather_rows(self.qs, idx, 128, nb2),
                                  self._gather_rows(self.scales, idx, 12, nb2),
                                  self.d.gather(0, idx), self.dmin.gather(0, idx), nb2)
        else:
            vals = self._q6k_vals(self._gather_rows(self.ql, idx, 128, nb2),
                                  self._gather_rows(self.qh, idx, 64, nb2),
                                  self._gather_rows(self.sc, idx, 16, nb2),
                                  self.d.gather(0, idx), nb2)
        return vals.reshape(1, T, self.D).cast(dtypes.float16)

    def fused_out(self, x: Tensor) -> Tensor:
        """x (1, D) -> (1, vocab) or (1, T, D) -> (1, T, vocab). Dequant fuses into the reduction."""
        nb = self.vocab * self.B
        if self.gtype == "Q4_K":
            wb = self._q4k_vals(self.qs, self.scales, self.d, self.dmin, nb)
        else:
            wb = self._q6k_vals(self.ql, self.qh, self.sc, self.d, nb)
        wb = wb.reshape(1, 1, self.vocab, self.B, 256)  # f16
        if x.ndim == 2:
            xr = x.reshape(1, 1, 1, self.B, 256)
            return (xr * wb).sum((-1, -2))[0]           # (1, vocab) f32
        T = x.shape[1]
        xr = x.reshape(1, T, 1, self.B, 256)
        return (xr * wb).sum((-1, -2))                  # (1, T, vocab) f32

    def fused_out_chunked(self, x: Tensor, n_chunks: int = 8) -> Tensor:
        """x (1, T, D) -> (1, T, vocab). Dequant the lm_head in row chunks so the f16
        transient is bounded, then tinygrad's optimized matmul per chunk."""
        vs = self.vocab // n_chunks
        B, D = self.B, self.D
        outs = []
        for c in range(n_chunks):
            b0, b1 = c * vs * B, (c + 1) * vs * B
            nb2 = vs * B
            if self.gtype == "Q4_K":
                wc = self._q4k_vals(self.qs[b0:b1], self.scales[b0:b1], self.d[b0:b1], self.dmin[b0:b1], nb2)
            else:
                wc = self._q6k_vals(self.ql[b0:b1], self.qh[b0:b1], self.sc[b0:b1], self.d[b0:b1], nb2)
            outs.append(x @ wc.reshape(vs, D).cast(dtypes.float16).T)   # (1, T, vs)
        return Tensor.cat(*outs, dim=-1)                                # (1, T, vocab) f32


# ---------------------------------------------------------------------------
# Layer implementations (T=1 decode path; the adapter processes rows one at a
# time, so the GDN scan is the single-step in-graph recurrence)
# ---------------------------------------------------------------------------

class _FullAttnLayer:
    def __init__(self, a: "Qwen35TextAdapter", w: dict, li: int):
        self.a, self.w, self.li, self.dev = a, w, li, a.dev
        r = a
        self.cache = Tensor.zeros(2, 1, r.n_kv_heads, r.seq_len, r.head_dim, dtype=dtypes.half, device=self.dev).clone()
        self.freqs = precompute_freqs_cis(r.rope_dim, r.seq_len, r.rope_base, device=self.dev)
        self.col_idx = Tensor.arange(r.seq_len).to(self.dev)

    def _attn(self, x: Tensor, start_pos) -> Tensor:
        r = self.a
        B, T, _ = x.shape
        qg = r._mm(x, self.w["q"]).reshape(B, T, r.n_heads, 2, r.head_dim)
        q = qg[:, :, :, 0, :]
        gate = qg[:, :, :, 1, :].reshape(B, T, r.n_heads * r.head_dim)
        k = r._mm(x, self.w["k"])
        v = r._mm(x, self.w["v"])
        q = q.transpose(1, 2)
        k = k.reshape(B, T, r.n_kv_heads, r.head_dim).transpose(1, 2)
        v = v.reshape(B, T, r.n_kv_heads, r.head_dim).transpose(1, 2)
        q = _rms(q, self.w["qn"], r.norm_eps)
        k = _rms(k, self.w["kn"], r.norm_eps)
        freqs = self.freqs[start_pos:start_pos + T]
        q = apply_rope(q[..., :r.rope_dim], freqs).cat(q[..., r.rope_dim:], dim=-1)
        k = apply_rope(k[..., :r.rope_dim], freqs).cat(k[..., r.rope_dim:], dim=-1)
        store = self.cache[:, :, :, start_pos:start_pos + T, :].uop.store(Tensor.stack(k, v).cast(dtypes.half).uop)
        kv = Tensor(self.cache.uop.after(store))
        kk = kv[0, :, :, 0:start_pos + T, :]
        vv = kv[1, :, :, 0:start_pos + T, :]
        mask = None
        if T != 1:
            rows = Tensor.arange(T).to(self.dev).reshape(1, 1, T, 1)
            cols = self.col_idx[0:start_pos + T].reshape(1, 1, 1, start_pos + T)
            mask = (cols > rows + start_pos).where(float("-inf"), 0.0)
        attn = q.scaled_dot_product_attention(kk, vv, attn_mask=mask, enable_gqa=True)
        attn = attn.transpose(1, 2).reshape(B, T, -1)
        return r._mm(attn * gate.sigmoid(), self.w["o"])

    def _ffn(self, x: Tensor) -> Tensor:
        r = self.a
        w = self.w
        if x.shape[1] > 1:
            # Batched prefill otherwise keeps the two large gate/up dequant
            # graphs live until the down projection is scheduled.  Realize
            # each projection at the memory boundary; decode (T=1) keeps the
            # fused graph path unchanged.
            gate = r._mm(x, w["ffn_gate"]).realize()
            up = r._mm(x, w["ffn_up"]).realize()
            return r._mm(gate.silu() * up, w["ffn_down"]).realize()
        return r._mm(r._mm(x, w["ffn_gate"]).silu() * r._mm(x, w["ffn_up"]), w["ffn_down"])

    def __call__(self, x: Tensor, start_pos) -> Tensor:
        r = self.a
        h = x + self._attn(_rms(x, self.w["norm"], r.norm_eps), start_pos)
        return (h + self._ffn(_rms(h, self.w["ffn_norm"], r.norm_eps))).contiguous()


class _GDNLayer:
    def __init__(self, a: "Qwen35TextAdapter", w: dict, li: int):
        self.a, self.w, self.li, self.dev = a, w, li, a.dev
        r = a
        self.conv_state = Tensor.zeros(1, r.d_conv - 1, r.conv_dim, device=self.dev).clone()
        self.rec_state = Tensor.zeros(1, r.num_v_heads, r.head_v_dim, r.head_k_dim, device=self.dev).clone()

    def _attn_pre(self, x: Tensor, start_pos, need_state: bool = True):
        r = self.a
        B, T, _ = x.shape
        initial = Tensor(start_pos, device=self.dev).eq(0)
        x = x.half()  # GDN projections run in f16 (matches the oracle); scan in f32
        out_gate = r._mm(x, self.w["gate"]).reshape(B, T, r.num_v_heads, r.head_v_dim)
        beta = r._mm(x, self.w["beta"]).sigmoid().reshape(B, T, r.num_v_heads)
        log_alpha = (r._mm(x, self.w["alpha"]).float() + self.w["dt"]).softplus()
        log_alpha = (log_alpha.reshape(B, T, r.num_v_heads, -1) * self.w["a"].reshape(r.num_v_heads, -1))
        alpha = log_alpha.transpose(1, 2).exp().unsqueeze(-1)  # (B, V, T, 1, 1)

        qkv = r._mm(x, self.w["qkv"])  # (B, T, conv_dim)
        conv_state = initial.where(0, self.conv_state)
        win = Tensor.cat(conv_state, qkv, dim=1)  # (B, d_conv-1+T, conv_dim)
        conv_out = sum((win[:, i:i + T] * self.w["conv"][:, i]) for i in range(self.w["conv"].shape[1])).silu()
        q, k, v = conv_out.split([r.q_dim, r.q_dim, r.conv_dim - 2 * r.q_dim], dim=-1)
        q = q.reshape(B, T, r.num_k_heads, r.head_k_dim).normalize(dim=-1, eps=1e-6).repeat(1, 1, r.num_v_rep, 1)
        k = k.reshape(B, T, r.num_k_heads, r.head_k_dim).normalize(dim=-1, eps=1e-6).repeat(1, 1, r.num_v_rep, 1)
        v = v.reshape(B, T, r.num_v_heads, r.head_v_dim)
        q, k, v = (z.transpose(1, 2).float() for z in (q, k, v))
        q = q * (r.head_k_dim ** -0.5)
        q = q.unsqueeze(-2)  # (B, V, T, 1, K)
        k = k.unsqueeze(-2)  # K on last axis: delta*k_t is a true outer product
        v = v.unsqueeze(-1)  # (B, V, T, Vd, 1)
        beta = beta.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)  # (B, V, T, 1, 1)

        conv_win = win[:, T:T + self.w["conv"].shape[1] - 1].cast(self.conv_state.dtype)
        if need_state:
            state = Tensor(self.rec_state.uop.after(self.conv_state.uop.store(conv_win.uop)))
            state = initial.where(0, state.float())
            return q, k, v, alpha, beta, out_gate, state, conv_win
        return q, k, v, alpha, beta, out_gate, conv_win

    def _attn_post(self, core: Tensor, out_gate: Tensor, start_pos) -> Tensor:
        r = self.a
        B, T = core.shape[0], core.shape[1]
        z = (_rms(core.reshape(B, T, r.num_v_heads, r.head_v_dim), self.w["sn"], r.norm_eps) * out_gate.silu()).cast(dtypes.half)
        return r._mm(z.reshape(B, T, -1), self.w["so"])

    def _attn(self, x: Tensor, start_pos) -> Tensor:
        """GDN recurrence for one decode row or a batched prefill chunk."""
        r = self.a
        q, k, v, alpha, beta, out_gate, state, _ = self._attn_pre(x, start_pos)
        V, Vd, K = r.num_v_heads, r.head_v_dim, r.head_k_dim
        if x.shape[1] == 1:
            st = state.reshape(V, Vd, K).contiguous()
            ah = alpha[:, :, 0].reshape(V, 1, 1).contiguous()
            bh = beta[:, :, 0].reshape(V, 1).contiguous()
            kh = k[:, :, 0].reshape(V, 1, K).contiguous()
            qh = q[:, :, 0].reshape(V, 1, K).contiguous()
            vh = v[:, :, 0].reshape(V, Vd).contiguous()
            s1 = st * ah
            delta = (vh - (s1 * kh).sum(-1)) * bh
            st_new = s1 + delta.reshape(V, Vd, 1) * kh
            core_t = (st_new * qh).sum(-1)  # (V, Vd)
            store = self.rec_state.uop.store(st_new.cast(self.rec_state.dtype).reshape(1, V, Vd, K).contiguous().uop)
            core = Tensor(core_t.reshape(1, 1, V, Vd).contiguous().uop.after(store))
        else:
            # The recurrence remains sequential in token order, while its
            # projections and surrounding FFN are evaluated as a batch.
            outs = []
            for t in range(x.shape[1]):
                s1 = state * alpha[:, :, t]
                delta = (v[:, :, t] - (s1 * k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]
                state = s1 + delta * k[:, :, t]
                outs.append((state * q[:, :, t]).sum(-1))
            store = self.rec_state.uop.store(state.cast(self.rec_state.dtype).uop)
            core = Tensor(outs[0].stack(*outs[1:], dim=1).contiguous().uop.after(store))
        return self._attn_post(core, out_gate, start_pos)

    def _ffn(self, x: Tensor) -> Tensor:
        r = self.a
        w = self.w
        if x.shape[1] > 1:
            gate = r._mm(x, w["ffn_gate"]).realize()
            up = r._mm(x, w["ffn_up"]).realize()
            return r._mm(gate.silu() * up, w["ffn_down"]).realize()
        return r._mm(r._mm(x, w["ffn_gate"]).silu() * r._mm(x, w["ffn_up"]), w["ffn_down"])

    def __call__(self, x: Tensor, start_pos) -> Tensor:
        r = self.a
        h = x + self._attn(_rms(x, self.w["norm"], r.norm_eps), start_pos)
        return (h + self._ffn(_rms(h, self.w["ffn_norm"], r.norm_eps))).contiguous()


class Qwen35TextAdapter:
    """Layer-sharded Qwen3.5-family inference adapter.

    Weights are loaded once into persistent VRAM (quantized types stay u8 and
    dequant on-device per matmul; IQ4_NL single-row matmuls use the fused
    GEMV custom kernel; the large Q6_K greedy head uses a fused GEMV too).
    Decode rows are processed one token at a time through one TinyJit, while
    native prefill uses bounded four-token chunks; both paths stay on-device
    and avoid per-op numpy ping-pong.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        n_layers: int,
        d_ff: int,
        layer_types: list[str],
        head_dim: int,
        rope_base: float,
        rope_dim: int,
        norm_eps: float,
        ssm: dict[str, int],
        tie_weights: bool,
        seq_len: int,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.n_layers = int(n_layers)
        self.d_ff = int(d_ff)
        self.layer_types = list(layer_types)
        self.head_dim = int(head_dim)
        self.rope_base = float(rope_base)
        self.rope_dim = int(rope_dim)
        self.norm_eps = float(norm_eps)
        self.tie_weights = bool(tie_weights)
        self.seq_len = int(seq_len)

        self.d_conv = int(ssm["conv_kernel"])
        self.d_state = int(ssm["state_size"])
        self.nk = int(ssm["num_k_heads"])
        self.hk = int(ssm["head_k_dim"])
        self.nv = int(ssm["num_v_heads"])
        self.hv = int(ssm["head_v_dim"])
        self.kd = self.nk * self.hk
        self.vd = self.nv * self.hv
        self.conv_dim = int(ssm.get("conv_dim", 2 * self.kd + self.vd))

        # derived GDN dims (matches D:/tmp/_q27_run.py)
        self.head_k_dim, self.num_k_heads = self.hk, self.nk
        self.head_v_dim, self.num_v_heads = self.hv, self.nv
        self.q_dim = self.kd
        self.num_v_rep = self.num_v_heads // self.num_k_heads

        self.layer_start = 0
        self.layer_end = self.n_layers
        self.shard_idx = 0
        self.num_shards = 1
        self.owns_output_norm = True
        self._is_shard = False
        self._keep_q4 = True
        self._gguf_buf: mmap.mmap | bytes | None = None
        self._gguf_by_name: dict[str, tuple[tuple, str, int]] = {}
        self._gguf_data_base = 0
        self._gguf_size = 0
        self._stream_states: dict[str, dict[str, Any]] = {}

        # GPU runtime state (set by _load_gpu)
        self.dev: str | None = None
        self._w: dict[int, dict[str, Any]] = {}
        self._mat_cache: dict[str, Any] = {}
        self._blks: list = []
        self._tok_emb = None
        self._out_norm = None
        self._out_w = None
        self._jits: dict[str, Any] = {}
        self._v_sp = None
        self._fuse_f16 = False
        self._rstride = 2

    @staticmethod
    def _layer_is_full(names: set[str], i: int) -> bool:
        # Qwen3.8 has a full-attention tail layer (64) in addition to the
        # interval-selected layers.  Tensor presence is authoritative.
        return f"blk.{i}.attn_q.weight" in names

    @classmethod
    def from_gguf_shard(
        cls,
        gguf_path: str | Path,
        shard_idx: int = 0,
        num_shards: int = 2,
        keep_q4: bool | None = None,
        device: str | None = None,
    ) -> "Qwen35TextAdapter":
        from tools.export_model import parse_gguf

        p = Path(gguf_path)
        meta, infos, data_base = parse_gguf(p)
        by_name = {n: (dims, gtype, off) for n, dims, gtype, off in infos}
        arch = str(meta.get("general.architecture", "qwen35"))
        if arch != "qwen35":
            raise ValueError(f"Qwen35TextAdapter requires qwen35 GGUF, got {arch!r}")
        names = set(by_name)
        L = int(meta["qwen35.block_count"])
        D = int(meta["qwen35.embedding_length"])
        H = int(meta["qwen35.attention.head_count"])
        KV = int(meta.get("qwen35.attention.head_count_kv") or H)
        HD = int(meta.get("qwen35.attention.key_length") or D // H)
        dff = int(meta.get("qwen35.feed_forward_length") or 4 * D)
        rope_base = float(meta.get("qwen35.rope.freq_base") or 10_000_000.0)
        rope_dim = int(meta.get("qwen35.rope.dimension_count") or HD)
        eps = float(meta.get("qwen35.attention.layer_norm_rms_epsilon") or 1e-6)
        ctx = int(meta.get("qwen35.context_length") or 2048)
        d_conv = int(meta["qwen35.ssm.conv_kernel"])
        d_state = int(meta["qwen35.ssm.state_size"])
        dt_rank = int(meta["qwen35.ssm.time_step_rank"])
        n_group = int(meta["qwen35.ssm.group_count"])
        d_inner = int(meta["qwen35.ssm.inner_size"])
        ssm = {
            "conv_kernel": d_conv,
            "state_size": d_state,
            "dt_rank": dt_rank,
            "group_count": n_group,
            "head_k_dim": d_state,
            "num_k_heads": n_group,
            "head_v_dim": d_inner // dt_rank,
            "num_v_heads": dt_rank,
            "inner_size": d_inner,
            "conv_dim": 2 * d_state * n_group + d_inner,
        }
        layers = ["full" if cls._layer_is_full(names, i) else "linear" for i in range(L)]
        if any(f"blk.{i}.attn_qkv.weight" not in names for i, kind in enumerate(layers) if kind == "linear"):
            raise ValueError("qwen35 GGUF has a linear layer without attn_qkv.weight")
        gpu_types = _GPU_BLOCK_TYPES | {"F16", "F32", "BF16"}
        unsupported = sorted({typ for n, _dims, typ, _off in infos if n.startswith("blk.") and typ not in gpu_types})
        if unsupported:
            raise ValueError(
                "Qwen35 CUDA adapter does not have GPU dequant kernels for "
                f"{', '.join(unsupported)}; use IQ4_NL or IQ4_XS"
            )
        n_vocab = int(by_name["token_embd.weight"][0][1])
        tie = "output.weight" not in by_name

        inst = cls(
            vocab_size=n_vocab,
            d_model=D,
            n_heads=H,
            n_kv_heads=KV,
            n_layers=L,
            d_ff=dff,
            layer_types=layers,
            head_dim=HD,
            rope_base=rope_base,
            rope_dim=rope_dim,
            norm_eps=eps,
            ssm=ssm,
            tie_weights=tie,
            seq_len=min(ctx, int(os.environ.get("QWEN35_MAX_CONTEXT", "2048"))),
        )
        shard_idx = int(shard_idx)
        num_shards = max(1, int(num_shards))
        if shard_idx < 0 or shard_idx >= num_shards:
            raise ValueError(f"invalid shard {shard_idx}/{num_shards}")

        # Balance by quantized bytes, not layer count.  This keeps the larger
        # full-attention layers and the final tail from making one card idle.
        def tensor_bytes(i: int) -> int:
            total = 0
            for n, (dims, typ, _off) in by_name.items():
                if not n.startswith(f"blk.{i}."):
                    continue
                count = int(np.prod(dims[::-1]))
                if typ in _BLOCK_BYTES:
                    block_count = count // (32 if typ in ("IQ4_NL", "Q8_0") else 256)
                    total += block_count * _BLOCK_BYTES[typ]
                else:
                    total += count * (2 if typ in ("F16", "BF16") else 4)
            return total

        weights = [tensor_bytes(i) for i in range(L)]
        total = sum(weights)
        forced_split = os.environ.get("QWEN35_SPLIT")
        if forced_split is not None and num_shards == 2:
            split = int(forced_split)
            if not 1 <= split < L:
                raise ValueError(f"QWEN35_SPLIT must be in [1, {L - 1}], got {split}")
            boundaries = [0, split, L]
        else:
            boundaries = [0]
            for s in range(1, num_shards):
                target = total * s / num_shards
                acc = 0
                cut = boundaries[-1]
                while cut < L and acc + weights[cut] <= target:
                    acc += weights[cut]
                    cut += 1
                prev = max(boundaries[-1] + 1, min(L - (num_shards - s), cut))
                nxt = min(L - (num_shards - s), cut + 1)
                prev_bytes = sum(weights[boundaries[-1]:prev])
                next_bytes = sum(weights[boundaries[-1]:nxt])
                chosen = prev if abs(prev_bytes - target) <= abs(next_bytes - target) else nxt
                boundaries.append(max(boundaries[-1] + 1, min(L - (num_shards - s), chosen)))
            boundaries.append(L)
        for j in range(1, len(boundaries)):
            boundaries[j] = max(boundaries[j], boundaries[j - 1] + 1)
        boundaries[-1] = L
        if len(boundaries) != num_shards + 1:
            raise AssertionError("invalid shard boundaries")

        inst.layer_start = boundaries[shard_idx]
        inst.layer_end = boundaries[shard_idx + 1]
        inst.shard_idx = shard_idx
        inst.num_shards = num_shards
        inst.owns_output_norm = shard_idx == num_shards - 1
        inst._is_shard = True
        inst._keep_q4 = True if keep_q4 is None else bool(keep_q4)
        if not inst._keep_q4:
            raise ValueError("Qwen35TextAdapter requires quantized mmap mode; set QWEN35_KEEP_Q4=1")
        with p.open("rb") as f:
            inst._gguf_buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        inst._gguf_by_name = by_name
        inst._gguf_data_base = data_base
        inst._gguf_size = p.stat().st_size
        inst._load_gpu(device)
        return inst

    @classmethod
    def from_gguf(cls, gguf_path: str | Path, device: str | None = None) -> "Qwen35TextAdapter":
        return cls.from_gguf_shard(gguf_path, 0, 1, device=device)

    # ------------------------------------------------------------------
    # GPU loading
    # ------------------------------------------------------------------

    def _load_gpu(self, device: str | None = None) -> None:
        if Tensor is None:
            raise RuntimeError("tinygrad is not importable")
        from tinygrad import Device

        # tinygrad 0.14 cannot select a device index via the DEV env var
        # (the second colon-field is the RENDERER name).  Use an explicit
        # indexed device string (device= arg or QWEN35_DEVICE=CUDA:1), which
        # is the mechanism the standalone runner uses for its second card.
        self.dev = device or os.environ.get("QWEN35_DEVICE") or str(Device.DEFAULT)
        # Resweep after the u32-word kernel change (2026-08-28): stride 1
        # edged out stride 2 on two clean runs (0.1917/0.1919 vs 0.1928/0.1932
        # s/tok) now that per-kernel cost is lower, so it is the new default.
        self._rstride = int(os.environ.get("QWEN35_RSTRIDE", "1"))
        fuse = os.environ.get("QWEN35_FUSE_F16", "auto")
        self._fuse_f16 = {"1": True, "0": False}.get(fuse, self._gguf_size >= 8 * 2**30)
        self._v_sp = UOp.variable("start_pos", 0, self.seq_len - 1)

        for li in range(self.layer_start, self.layer_end):
            self._w[li] = self._load_layer_weights(li)
            # Drain tinygrad's pinned host staging (pending_copyin) per layer: each
            # weight upload allocates a pinned buffer for the async HtoD copy, and
            # without a sync they accumulate until the WDDM pinned pool is exhausted
            # (~5.5GB) — the 27B two-shard load OOMs without these syncs (the standalone
            # runner syncs after every weight group).
            Device[self.dev].synchronize()
        if self.layer_start == 0:
            self._tok_emb = self._load_emb()
            Device[self.dev].synchronize()
        if self.owns_output_norm:
            self._out_norm = self._small_gpu("output_norm.weight")
            self._out_w = self._load_head()
            Device[self.dev].synchronize()

        self._blks = []
        for li in range(self.layer_start, self.layer_end):
            w = self._w[li]
            if self.layer_types[li] == "full":
                self._blks.append(_FullAttnLayer(self, w, li))
            else:
                self._blks.append(_GDNLayer(self, w, li))
        Device[self.dev].synchronize()

    def _small_gpu(self, name: str) -> Tensor:
        from tools.export_model import _tensor_data

        dims, typ, off = self._gguf_by_name[name]
        arr = np.asarray(_tensor_data(self._gguf_buf, name, dims, typ, off, self._gguf_data_base), dtype=np.float32)
        return Tensor(arr.astype(np.float16), device=self.dev).realize()

    def _matrix_gpu(self, name: str):
        """Persistent GPU weight for a layer matrix. IQ4_NL/Q5_K stay u8 (fused/on-device
        dequant); other quantized types dequant to an f16 tensor once; F32/F16/BF16 stay
        as-is."""
        from tools.export_model import _tensor_data

        dims, typ, off = self._gguf_by_name[name]
        shape = tuple(int(d) for d in dims[::-1])  # numpy (out, in)
        if typ in ("IQ4_NL", "Q5_K"):
            raw, _, _ = self._raw_blocks(dims, typ, off)
            return _QWeight(self.dev, typ, np.ascontiguousarray(raw).reshape(-1, _BLOCK_BYTES[typ]), shape)
        if typ in _BLOCK_BYTES:
            return _F16W(self._gpu_dequant_k(dims, typ, off))
        arr = np.asarray(_tensor_data(self._gguf_buf, name, dims, typ, off, self._gguf_data_base), dtype=np.float32)
        return _F16W(Tensor(arr.astype(np.float16), device=self.dev).realize())

    def _load_matrix_weight(self, name: str):
        """Persistent weight for a (vocab-like, D) matrix: KQuant if Q4_K/Q6_K, else an
        f16 tensor.  Cached by name so a tied lm_head reuses the embedding object."""
        cached = self._mat_cache.get(name)
        if cached is not None:
            return cached
        from tools.export_model import _tensor_data

        dims, typ, off = self._gguf_by_name[name]
        shape = tuple(int(d) for d in dims[::-1])  # numpy (rows, D)
        if typ in ("Q4_K", "Q6_K"):
            raw, _, _ = self._raw_blocks(dims, typ, off)
            w = _KQuant(self.dev, typ, np.ascontiguousarray(raw).reshape(-1, _BLOCK_BYTES[typ]), shape)
        elif typ in _BLOCK_BYTES:
            w = _F16W(self._gpu_dequant_k(dims, typ, off))
        else:
            arr = np.asarray(_tensor_data(self._gguf_buf, name, dims, typ, off, self._gguf_data_base), dtype=np.float32)
            w = _F16W(Tensor(arr.astype(np.float16), device=self.dev).realize())
        self._mat_cache[name] = w
        return w

    def _load_emb(self):
        return self._load_matrix_weight("token_embd.weight")

    def _load_head(self):
        # Tied heads: each shard builds the embedding matrix locally from the
        # mmap (shard processes have no access to each other's VRAM).
        return self._load_matrix_weight("token_embd.weight" if self.tie_weights else "output.weight")

    def _load_layer_weights(self, li: int) -> dict:
        p = f"blk.{li}."
        w: dict[str, Any] = {}
        if self.layer_types[li] == "full":
            for n, key in (
                (p + "attn_q.weight", "q"), (p + "attn_k.weight", "k"), (p + "attn_v.weight", "v"),
                (p + "attn_output.weight", "o"), (p + "ffn_gate.weight", "ffn_gate"),
                (p + "ffn_up.weight", "ffn_up"), (p + "ffn_down.weight", "ffn_down"),
            ):
                w[key] = self._matrix_gpu(n)
            for n, key in (
                (p + "attn_norm.weight", "norm"), (p + "post_attention_norm.weight", "ffn_norm"),
                (p + "attn_q_norm.weight", "qn"), (p + "attn_k_norm.weight", "kn"),
            ):
                w[key] = self._small_gpu(n)
        else:
            for n, key in (
                (p + "attn_qkv.weight", "qkv"), (p + "attn_gate.weight", "gate"),
                (p + "ssm_alpha.weight", "alpha"), (p + "ssm_beta.weight", "beta"),
                (p + "ssm_out.weight", "so"), (p + "ffn_gate.weight", "ffn_gate"),
                (p + "ffn_up.weight", "ffn_up"), (p + "ffn_down.weight", "ffn_down"),
            ):
                w[key] = self._matrix_gpu(n)
            for n, key in (
                (p + "attn_norm.weight", "norm"), (p + "post_attention_norm.weight", "ffn_norm"),
                (p + "ssm_norm.weight", "sn"), (p + "ssm_a", "a"), (p + "ssm_dt.bias", "dt"),
                (p + "ssm_conv1d.weight", "conv"),
            ):
                w[key] = self._small_gpu(n)
        return w

    # ------------------------------------------------------------------
    # compute plumbing
    # ------------------------------------------------------------------

    def _mm(self, x: Tensor, w) -> Tensor:
        """Matmul x @ w.T — fused IQ4_NL/Q5_K GEMV for single-row decode.

        f32 path (FullAttn projections + FFNs): parity 5e-7, oracle-safe.
        f16 path (GDN projections): fused only for big models (QWEN35_FUSE_F16=auto
        -> GGUF >= 8GiB); the f16 matmul differs by 1 f16 ULP on ~0.1% of outputs
        (accumulation order) — 20x smaller than the IQ4_NL quantization error."""
        # The 27B ssm_out shape is gate-verified and materially faster.  Q5
        # attn_qkv remains on the reference path because fusing it fails the
        # model gate and does not improve throughput.
        q5_part = os.environ.get("QWEN35_FUSE_Q5_PART", "so").lower()
        # "so" is always eligible (it's the accepted default); qkv/v are
        # additional opt-in diagnostics that must not disable "so".
        q5_fused = (isinstance(w, _QWeight) and w.gtype == "Q5_K"
                    and ((q5_part in {"1", "so", "ssm_out", "all"} and w.shape == (5120, 6144))
                         or ((q5_part in {"qkv", "all"} or os.environ.get("QWEN35_FUSE_Q5_QKV", "0") == "1")
                             and w.shape == (10240, 5120))
                         or (os.environ.get("QWEN35_FUSE_Q5_V", "1") == "1" and w.shape == (1024, 5120))))
        if (isinstance(w, _QWeight) and str(self.dev).startswith("CUDA")
                and x.numel() == w.shape[1]
                and (w.gtype == "IQ4_NL" or (w.gtype == "Q5_K" and q5_fused))
                and (x.dtype == dtypes.float32 or (x.dtype == dtypes.float16 and self._fuse_f16))):
            return w.gemv(x)
        if isinstance(w, _QWeight) and x.ndim == 3 and x.shape[1] > 1:
            # T>1 prefill cannot use the single-row fused GEMV; use the
            # memory-bounded f16 dequant form instead of an f32 full matrix.
            if w.gtype == "IQ4_NL":
                return iq4nl_gemm(x, w, x.shape[1])
            # The shape-specific ssm_out kernel is gate-verified.  QKV's
            # recurrent accumulation is more sensitive to fused dequant
            # rounding, so retain the reference batched matmul there.
            if w.shape == (5120, 6144):
                return q5k_gemm(x, w, x.shape[1])
            return x @ w.dequant_f16().T
        return x @ w.dequant().T

    def _forward_ids(self, ids_t: Tensor, start_pos) -> Tensor:
        if isinstance(self._tok_emb, _KQuant):
            x = self._tok_emb.emb(ids_t).float()
        else:
            x = self._tok_emb[ids_t].float()
        return self._forward_hid(x, start_pos)

    def _forward_hid(self, x: Tensor, start_pos) -> Tensor:
        for i, blk in enumerate(self._blks):
            x = blk(x, start_pos)
            if (i + 1) % self._rstride == 0 or i == len(self._blks) - 1:
                x = x.realize()
        return x

    def _forward_logits(self, x: Tensor, start_pos) -> Tensor:
        x = self._forward_hid(x, start_pos)
        z = _rms(x, self._out_norm, self.norm_eps)
        if isinstance(self._out_w, _KQuant):
            if self._out_w.vocab * self._out_w.D > 500_000_000:
                logits = self._out_w.fused_out_chunked(z)
            else:
                logits = self._out_w.fused_out(z)
        else:
            logits = z @ self._out_w.dequant().T
        return logits.reshape(1, -1)

    def _forward_argmax_last(self, x: Tensor, start_pos) -> Tensor:
        """Batched-prefill graph returning only the final row's argmax."""
        x = self._forward_hid(x, start_pos)[:, -1:]
        z = _rms(x, self._out_norm, self.norm_eps)
        if (os.environ.get("QWEN35_FUSE_Q6_HEAD", "1") == "1"
                and isinstance(self._out_w, _KQuant) and self._out_w.gtype == "Q6_K"
                and self._out_w.vocab * self._out_w.D > 500_000_000
                and self._out_w.vocab % 256 == 0):
            logits = q6k_gemv(z, self._out_w.ql32T, self._out_w.qh32T, self._out_w.scT,
                               self._out_w.dT, self._out_w.vocab, self._out_w.D, self.dev)
        elif isinstance(self._out_w, _KQuant):
            logits = (self._out_w.fused_out_chunked(z) if self._out_w.vocab * self._out_w.D > 500_000_000
                      else self._out_w.fused_out(z))
        else:
            logits = z @ self._out_w.dequant().T
        return logits.reshape(1, -1).argmax(-1)

    def _forward_argmax(self, x: Tensor, start_pos) -> Tensor:
        """Final greedy token with the vocabulary reduction inside the JIT graph."""
        if (os.environ.get("QWEN35_FUSE_Q6_HEAD", "1") == "1"
                and isinstance(self._out_w, _KQuant) and self._out_w.gtype == "Q6_K"
                and self._out_w.vocab * self._out_w.D > 500_000_000
                and self._out_w.vocab % 256 == 0):
            x = self._forward_hid(x, start_pos)
            z = _rms(x, self._out_norm, self.norm_eps)
            logits = q6k_gemv(z, self._out_w.ql32T, self._out_w.qh32T, self._out_w.scT,
                               self._out_w.dT, self._out_w.vocab, self._out_w.D, self.dev)
            return logits.argmax(-1)
        return self._forward_logits(x, start_pos).argmax(-1)

    def _step(self, key: str, inp: Tensor, pos: int) -> Tensor:
        jit = self._jits.get(key)
        if jit is None:
            if key.startswith("ids"):
                fn = self._forward_ids
            elif key.startswith("hid"):
                fn = self._forward_hid
            elif key == "log":
                fn = self._forward_logits
            elif key == "argmax":
                fn = self._forward_argmax
            elif key.startswith("argmax_last"):
                fn = self._forward_argmax_last
            else:
                raise KeyError(key)
            jit = TinyJit(fn)
            self._jits[key] = jit
        return jit(inp, self._v_sp.bind(pos))

    # ------------------------------------------------------------------
    # public pipeline API (mesh plumbing — unchanged)
    # ------------------------------------------------------------------

    def new_state(self, max_len: int | None = None) -> dict[str, Any]:
        return {"pos": 0, "max_len": max(1, int(max_len or self.seq_len))}

    def forward_shard_ids_to_hidden(self, x_ids: Any, state: dict[str, Any] | None = None) -> np.ndarray:
        if self.layer_start != 0:
            raise AssertionError("ids must enter shard 0")
        state = state or self.new_state()
        ids = np.asarray(x_ids, dtype=np.int64).reshape(-1).tolist()
        if not ids:
            return np.empty((0, self.d_model), dtype=np.float32)
        outs = []
        for tid in ids:
            t = Tensor([[tid]], dtype=dtypes.int32, device=self.dev)
            out = self._step("ids", t, int(state["pos"]))
            outs.append(np.asarray(out.numpy(), dtype=np.float32).reshape(-1))
            state["pos"] += 1
        return np.stack(outs, axis=0)

    def forward_shard_hidden_to_hidden(self, hidden: Any, state: dict[str, Any] | None = None) -> np.ndarray:
        state = state or self.new_state()
        hidden = np.asarray(hidden, dtype=np.float32).reshape(-1, self.d_model)
        if hidden.shape[0] == 0:
            return np.empty((0, self.d_model), dtype=np.float32)
        outs = []
        for row in hidden:
            x = Tensor(row.reshape(1, 1, -1), device=self.dev)
            out = self._step("hid", x, int(state["pos"]))
            outs.append(np.asarray(out.numpy(), dtype=np.float32).reshape(-1))
            state["pos"] += 1
        return np.stack(outs, axis=0)

    def forward_shard_hidden_to_logits(self, hidden: Any, state: dict[str, Any] | None = None) -> np.ndarray:
        if not self.owns_output_norm:
            raise AssertionError("logits must be produced by the final shard")
        state = state or self.new_state()
        hidden = np.asarray(hidden, dtype=np.float32).reshape(-1, self.d_model)
        if hidden.shape[0] == 0:
            raise ValueError("forward_shard_hidden_to_logits requires at least one row")
        last = None
        for row in hidden:
            x = Tensor(row.reshape(1, 1, -1), device=self.dev)
            last = self._step("log", x, int(state["pos"]))
            state["pos"] += 1
        return np.asarray(last.numpy(), dtype=np.float32).reshape(1, -1)

    def stream_state(self, stream_id: str, reset: bool = False) -> dict[str, Any]:
        if reset or stream_id not in self._stream_states:
            self._stream_states[stream_id] = self.new_state()
        return self._stream_states[stream_id]

    def release_stream(self, stream_id: str) -> None:
        self._stream_states.pop(stream_id, None)

    def sample_batch(self, batch_size: int, seed: int):
        rng = np.random.default_rng(seed)
        x = Tensor(rng.integers(0, self.vocab_size, size=(batch_size, 32), dtype=np.int32))
        return x[:, :-1], x[:, 1:]

    def loss(self, logits, y):
        return logits.sparse_categorical_crossentropy(y)

    # ------------------------------------------------------------------
    # dequant helpers (kept for the load-time fallback and the IQ4_XS parity test)
    # ------------------------------------------------------------------

    def _raw_blocks(self, dims: tuple, typ: str, off: int) -> tuple[np.ndarray, int, int]:
        n_in, n_out = int(dims[0]), int(dims[1])
        block = _BLOCK_BYTES[typ]
        if typ in ("IQ4_NL", "Q8_0"):
            if n_in % 32:
                raise ValueError(f"{typ} tensor has non-32 input width: {n_in}")
        elif n_in % 256:
            raise ValueError(f"{typ} tensor has non-256 input width: {n_in}")
        nb = n_in // (32 if typ in ("IQ4_NL", "Q8_0") else 256)
        raw = np.frombuffer(self._gguf_buf, dtype=np.uint8, count=n_out * nb * block, offset=self._gguf_data_base + off)
        return raw.reshape(n_out, nb, block), n_in, n_out

    def _gpu_dequant_k(self, dims: tuple, typ: str, off: int):
        from tinygrad import Tensor as _Tensor
        from tinygrad import dtypes

        # Pin every construction to this shard's device: the old one-device-
        # per-process world relied on the default; with device="CUDA:1" that
        # would leak device-0 buffers into the layer graph (mixed-device assert).
        Tensor = lambda *a, **k: _Tensor(*a, device=self.dev, **k)

        blk, n_in, n_out = self._raw_blocks(dims, typ, off)
        if typ == "IQ4_NL":
            from tools.export_model import _IQ4_KVALUES

            qs = Tensor(np.ascontiguousarray(blk[:, :, 2:18]).copy())
            sc = Tensor(blk[:, :, 0:2].copy().view("<f2").reshape(n_out, -1, 1).astype(np.float32))
            nib = (qs & 0x0F).cat(qs >> 4, dim=-1)
            vals = Tensor(_IQ4_KVALUES.reshape(16).astype(np.float32)).gather(0, nib.cast(dtypes.int32).reshape(-1)).reshape(n_out, n_in // 32, 32)
            return (vals * sc).reshape(n_out, n_in).cast(dtypes.float16)

        if typ == "IQ4_XS":
            from tools.export_model import _IQ4_KVALUES

            # IQ4_XS: d:f16, eight packed 6-bit scales, then 128 nibbles.
            # Each scale covers 32 values in the 256-value super-block.
            d = Tensor(blk[:, :, 0:2].copy().view("<f2").reshape(n_out, -1, 1).astype(np.float32))
            scales_h = Tensor(blk[:, :, 2:4].copy().view("<u2").reshape(n_out, -1, 1))
            scales_l = Tensor(np.ascontiguousarray(blk[:, :, 4:8]).copy())
            low = [(scales_l[:, :, g // 2] >> (4 * (g & 1))) & 0x0F for g in range(8)]
            high = [(scales_h >> (2 * g)) & 0x03 for g in range(8)]
            qs = Tensor(np.ascontiguousarray(blk[:, :, 8:]).copy()).reshape(n_out, -1, 8, 16)
            out = []
            for g in range(8):
                nib = (qs[:, :, g] & 0x0F).cat(qs[:, :, g] >> 4, dim=-1)
                vals = Tensor(_IQ4_KVALUES.reshape(16).astype(np.float32)).gather(
                    0, nib.cast(dtypes.int32).reshape(-1)
                ).reshape(n_out, -1, 32)
                scale = (low[g] | (high[g] << 4)).cast(dtypes.int32) - 32
                out.append(vals * d * scale.cast(dtypes.float32).reshape(n_out, -1, 1))
            return out[0].cat(*out[1:], dim=-1).reshape(n_out, n_in).cast(dtypes.float16)

        if typ == "Q8_0":
            d = Tensor(blk[:, :, 0:2].copy().view("<f2").reshape(n_out, -1, 1).astype(np.float32))
            q = Tensor(blk[:, :, 2:].copy().view(np.int8))
            return (q.cast(dtypes.float32) * d).reshape(n_out, n_in).cast(dtypes.float16)

        if typ in ("Q4_K", "Q5_K"):
            d = Tensor(blk[:, :, 0:2].copy().view("<f2").reshape(n_out, -1, 1).astype(np.float32))
            dm = Tensor(blk[:, :, 2:4].copy().view("<f2").reshape(n_out, -1, 1).astype(np.float32))
            scb = Tensor(np.ascontiguousarray(blk[:, :, 4:16]).copy())
            sc = [
                scb[:, :, 0] & 63, scb[:, :, 1] & 63, scb[:, :, 2] & 63, scb[:, :, 3] & 63,
                (scb[:, :, 8] & 15) | ((scb[:, :, 0] >> 6) << 4),
                (scb[:, :, 9] & 15) | ((scb[:, :, 1] >> 6) << 4),
                (scb[:, :, 10] & 15) | ((scb[:, :, 2] >> 6) << 4),
                (scb[:, :, 11] & 15) | ((scb[:, :, 3] >> 6) << 4),
            ]
            mn = [
                scb[:, :, 4] & 63, scb[:, :, 5] & 63, scb[:, :, 6] & 63, scb[:, :, 7] & 63,
                (scb[:, :, 8] >> 4) | ((scb[:, :, 4] >> 6) << 4),
                (scb[:, :, 9] >> 4) | ((scb[:, :, 5] >> 6) << 4),
                (scb[:, :, 10] >> 4) | ((scb[:, :, 6] >> 6) << 4),
                (scb[:, :, 11] >> 4) | ((scb[:, :, 7] >> 6) << 4),
            ]
            out = []
            if typ == "Q4_K":
                qs = Tensor(np.ascontiguousarray(blk[:, :, 16:]).copy())
                q = qs.reshape(n_out, -1, 4, 32)
                for g in range(4):
                    for h in range(2):
                        qv = (q[:, :, g] >> (4 * h)) & 15
                        s = sc[2 * g + h].cast(dtypes.float32).reshape(n_out, -1, 1).expand((n_out, -1, 32))
                        m = mn[2 * g + h].cast(dtypes.float32).reshape(n_out, -1, 1).expand((n_out, -1, 32))
                        out.append((qv.cast(dtypes.float32) * d * s) - (dm * m))
            else:
                qs = Tensor(np.ascontiguousarray(blk[:, :, 48:]).copy())
                q = qs.reshape(n_out, -1, 4, 32)
                qh = Tensor(np.ascontiguousarray(blk[:, :, 16:48]).copy())
                for sub in range(8):
                    qv = (q[:, :, sub // 2] >> (4 * (sub & 1))) & 15
                    hi = (qh >> sub) & 1
                    qv = qv | (hi << 4)
                    s = sc[sub].cast(dtypes.float32).reshape(n_out, -1, 1).expand((n_out, -1, 32))
                    m = mn[sub].cast(dtypes.float32).reshape(n_out, -1, 1).expand((n_out, -1, 32))
                    out.append((qv.cast(dtypes.float32) * d * s) - (dm * m))
            return out[0].cat(*out[1:], dim=-1).reshape(n_out, n_in).cast(dtypes.float16)

        if typ == "Q6_K":
            ql = Tensor(np.ascontiguousarray(blk[:, :, 0:128]).copy())
            qh = Tensor(np.ascontiguousarray(blk[:, :, 128:192]).copy())
            scales = Tensor(np.ascontiguousarray(blk[:, :, 192:208]).copy()).cast(dtypes.int8)
            d = Tensor(blk[:, :, 208:210].copy().view("<f2").reshape(n_out, -1, 1).astype(np.float32))
            out = []
            for half in range(2):
                for group in range(4):
                    s = scales[:, :, half * 8 + group * 2:half * 8 + group * 2 + 2].reshape(n_out, -1, 2, 1).expand((n_out, -1, 2, 16)).reshape(n_out, -1, 32).cast(dtypes.float32)
                    qlo = ql[:, :, half * 64 + (group & 1) * 32:half * 64 + (group & 1) * 32 + 32]
                    qhi = qh[:, :, half * 32:half * 32 + 32]
                    qv = ((qlo & 15) if group < 2 else (qlo >> 4) & 15) | (((qhi >> (2 * group)) & 3) << 4)
                    out.append(qv.cast(dtypes.float32) * d * s - 32.0 * s * d)
            return out[0].cat(*out[1:], dim=-1).reshape(n_out, n_in).cast(dtypes.float16)

        raise NotImplementedError(typ)


class Qwen35TextPipeline:
    """Run two layer-sharded adapters with a direct device-to-device boundary.

    The mesh-facing adapter API intentionally returns NumPy arrays because the
    hidden state may cross a websocket.  That boundary is expensive when both
    shards live in the same process, though: it forces every token through
    device -> host -> device.  This helper keeps the hidden state as a
    tinygrad tensor, transfers it directly to the second GPU, and synchronizes
    the two WDDM CUDA queues before the next shard consumes it.

    tinygrad 0.14's combined multi-device TinyJit did not establish a reliable
    cross-device dependency on this WDDM driver (the correctness gate failed
    after recurrent state advanced), so the safe version intentionally uses the
    existing per-shard TinyJits plus the direct tensor boundary.

    The adapters remain independently usable; this is an additive fast path for
    local dual-device deployments and benchmarks.
    """

    def __init__(self, first: Qwen35TextAdapter, last: Qwen35TextAdapter) -> None:
        if first.layer_end != last.layer_start:
            raise ValueError(
                f"non-contiguous Qwen35 shards: {first.layer_start}:{first.layer_end} "
                f"then {last.layer_start}:{last.layer_end}"
            )
        if first.layer_start != 0 or not last.owns_output_norm:
            raise ValueError("Qwen35TextPipeline requires the first and final model shards")
        if first.d_model != last.d_model or first.seq_len != last.seq_len:
            raise ValueError("Qwen35 pipeline shards disagree on model dimensions")
        if Tensor is None or UOp is None:
            raise RuntimeError("tinygrad is not importable")
        from tinygrad import Device

        self.first = first
        self.last = last
        self._Device = Device
        # The destination queue sync is the validated fast default.  `both`
        # remains available as a conservative diagnostic fallback.
        self._sync_mode = os.environ.get("QWEN35_PIPELINE_SYNC", "dst").lower()
        if self._sync_mode not in {"src", "dst", "both"}:
            raise ValueError("QWEN35_PIPELINE_SYNC must be src, dst, or both")

    def _step(self, token_id: int, pos: int, output_key: str = "argmax") -> Tensor:
        token = Tensor([[int(token_id)]], dtype=dtypes.int32, device=self.first.dev)
        hidden = self.first._step("ids", token, int(pos))
        # On WDDM, the source and destination CUDA queues are independent.
        # Synchronize before and after the cross-device copy so the next
        # shard never consumes a stale hidden state.  This remains a device
        # transfer; it does not serialize through NumPy or base64.
        if self._sync_mode in ("src", "both"):
            self._Device[self.first.dev].synchronize()
        hidden = hidden.to(self.last.dev).realize()
        if self._sync_mode in ("dst", "both"):
            self._Device[self.last.dev].synchronize()
        return self.last._step(output_key, hidden, int(pos))

    def _step_batch(self, token_ids: Any, pos: int, output_key: str) -> Tensor:
        """Run a fixed-size prefill chunk through both shards on-device."""
        ids = np.asarray(token_ids, dtype=np.int32).reshape(1, -1)
        T = ids.shape[1]
        token = Tensor(ids, dtype=dtypes.int32, device=self.first.dev)
        hidden = self.first._step(f"ids{T}", token, int(pos))
        if self._sync_mode in ("src", "both"):
            self._Device[self.first.dev].synchronize()
        hidden = hidden.to(self.last.dev).realize()
        if self._sync_mode in ("dst", "both"):
            self._Device[self.last.dev].synchronize()
        if output_key == "hid":
            key = f"hid{T}"
        elif output_key == "argmax_last":
            key = f"argmax_last{T}"
        else:
            raise ValueError(f"unsupported batched pipeline output: {output_key}")
        return self.last._step(key, hidden, int(pos))

    def _step_hidden(self, token_id: int, pos: int) -> Tensor:
        """Run one token through both layer stacks without the output head."""
        token = Tensor([[int(token_id)]], dtype=dtypes.int32, device=self.first.dev)
        hidden = self.first._step("ids", token, int(pos))
        if self._sync_mode in ("src", "both"):
            self._Device[self.first.dev].synchronize()
        hidden = hidden.to(self.last.dev).realize()
        if self._sync_mode in ("dst", "both"):
            self._Device[self.last.dev].synchronize()
        return self.last._step("hid", hidden, int(pos))

    @staticmethod
    def _check_states(state0: dict[str, Any], state1: dict[str, Any]) -> int:
        pos0, pos1 = int(state0["pos"]), int(state1["pos"])
        if pos0 != pos1:
            raise ValueError(f"Qwen35 pipeline state positions differ: {pos0} != {pos1}")
        return pos0

    def run_token(self, token_id: int, state0: dict[str, Any], state1: dict[str, Any]) -> int:
        """Consume one token and return its greedy next-token id."""
        pos = self._check_states(state0, state1)
        logits = self._step(token_id, pos)
        state0["pos"] = pos + 1
        state1["pos"] = pos + 1
        return int(logits.item())

    def run(self, ids: Any, state0: dict[str, Any], state1: dict[str, Any]) -> np.ndarray:
        """Consume ids one at a time and return the final logits as NumPy."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("Qwen35TextPipeline.run requires at least one token")
        for token_id in ids[:-1]:
            pos = self._check_states(state0, state1)
            self._step_hidden(int(token_id), pos)
            state0["pos"] = pos + 1
            state1["pos"] = pos + 1
        pos = self._check_states(state0, state1)
        last = self._step(int(ids[-1]), pos, output_key="log")
        state0["pos"] = pos + 1
        state1["pos"] = pos + 1
        return np.asarray(last.numpy(), dtype=np.float32)

    def run_prefill(self, ids: Any, state0: dict[str, Any], state1: dict[str, Any]) -> int:
        """Consume a prompt while computing output logits only for its last row."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("Qwen35TextPipeline.run_prefill requires at least one token")
        # Four-token chunks are the measured sweet spot on the canonical dual-
        # 5060 Ti 27B setup. Keep the override for smaller cards/variants and
        # for apples-to-apples diagnostics against the sequential path.
        batch = max(1, int(os.environ.get("QWEN35_PREFILL_BATCH", "4")))
        # Keep one-token generation on the lean decode graph.  Batched mode is
        # only for actual prompt chunks; routing T=1 through the prefill
        # wrapper would add a separate JIT and a redundant last-row slice.
        if batch > 1 and ids.size > 1:
            pos = self._check_states(state0, state1)
            for start in range(0, ids.size, batch):
                chunk = ids[start:start + batch]
                last = start + chunk.size == ids.size
                out = self._step_batch(chunk, pos, "argmax_last" if last else "hid")
                pos += int(chunk.size)
                state0["pos"], state1["pos"] = pos, pos
                if last:
                    return int(out.item())
            raise AssertionError("prefill chunk loop produced no final output")
        for token_id in ids[:-1]:
            pos = self._check_states(state0, state1)
            self._step_hidden(int(token_id), pos)
            state0["pos"] = pos + 1
            state1["pos"] = pos + 1
        return self.run_token(int(ids[-1]), state0, state1)

    def run_greedy(self, ids: Any, state0: dict[str, Any], state1: dict[str, Any]) -> int:
        """Consume ids and return only the final greedy token on-device."""
        return self.run_prefill(ids, state0, state1)
