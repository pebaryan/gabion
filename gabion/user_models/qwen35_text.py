"""Qwen3.8/Qwen3.5 hybrid text inference for CUDA pebble workers.

NOTE: read ``docs/qwen35-inference-status.md`` FIRST — this adapter is the
ported version of the proven standalone runner (``D:/tmp/_q27_run.py``).
The previous per-layer-dequant + numpy-scan design measured ~35x slower
(14.2s/tok shard-0 vs 0.40s/tok) and was replaced.  Techniques ported:

- weights stay QUANTIZED as persistent VRAM u8 tensors (IQ4_NL / Q5_K);
  IQ4_NL single-row matmuls go through a fused dequant-GEMV custom kernel
  (``_QWeight.gemv``) so no f16 matrix is ever materialized for decode.
- the GatedDeltaNet recurrent scan runs IN-GRAPH on the GPU (T=1 scan step,
  state tensors updated via ``uop.store`` inside the TinyJit graph).
- decode/prefill rows are processed one token at a time through one TinyJit
  per entry point (ids/hidden/logits); no per-op Tensor<->numpy ping-pong.
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

try:
    from tinygrad import Tensor, dtypes, TinyJit
    from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo
    from tinygrad.llm.model import precompute_freqs_cis, apply_rope
except ImportError:  # keep the module importable in pure-meta contexts
    Tensor = dtypes = TinyJit = UOp = None  # type: ignore
    Ops = AxisType = KernelInfo = None  # type: ignore
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


def iq4nl_gemv(x: Tensor, u8: Tensor, sc: Tensor, ks: Tensor, out_dim: int, in_dim: int, dev: str) -> Tensor:
    """Fused dequant-GEMV for one row x (in_dim,) -> (1, 1, out_dim)."""
    was_f16 = x.dtype == dtypes.float16
    key = (out_dim, in_dim, x.dtype)
    fxn = _IQ4NL_FXN.get(key)
    if fxn is None:
        fxn = build_iq4nl_gemv(out_dim, in_dim, f"iq4nl_{out_dim}_{in_dim}")
        _IQ4NL_FXN[key] = fxn
    out = Tensor.empty(out_dim, device=dev)
    res = out.custom_kernel(x.reshape(in_dim), u8, sc, ks, fxn=fxn)[0].reshape(1, 1, out_dim)
    return res.cast(dtypes.float16) if was_f16 else res


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
            raw = np.ascontiguousarray(blk[:, 2:18])                       # (nb, 16) nibbles
            sc = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)   # (nb,)
            self.u8 = Tensor(raw, device=dev, dtype=dtypes.uint8).realize()
            self.sc = Tensor(sc, device=dev).realize()
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
            self.u8 = Tensor(blk, device=dev, dtype=dtypes.uint8).realize()
            self.d = Tensor(d, device=dev).realize()
            self.dmin = Tensor(dmin, device=dev).realize()
            self.sc = Tensor(sc, device=dev).realize()
            self.mn = Tensor(mn, device=dev).realize()
        else:
            raise ValueError(f"_QWeight: unsupported on-device type {gtype}")

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
            qs = self.u8[:, 48:176]
            qh = self.u8[:, 16:48]
            qs_lo = qs & 0x0F
            qs_hi = (qs >> 4) & 0x0F
            nib = Tensor.stack(qs_lo.reshape(nb, 4, 32), qs_hi.reshape(nb, 4, 32), dim=2).reshape(nb, 8, 32)
            hi_bits = ((qh.reshape(nb, 1, 32) >> Tensor.arange(8).to(self.dev).reshape(1, 8, 1)) & 1).reshape(nb, 8, 32)
            val = (nib | (hi_bits << 4)).float()
            dl = self.d.reshape(nb, 1, 1) * self.sc.reshape(nb, 8, 1)
            dm = self.dmin.reshape(nb, 1, 1) * self.mn.reshape(nb, 8, 1)
            return (dl * val - dm).reshape(self.shape).cast(dtypes.float16)
        raise AssertionError

    def gemv(self, x: Tensor) -> Tensor:
        """Fused dequant-GEMV (M=1 only) — replaces dequant().T @ x for IQ4_NL."""
        assert self.gtype == "IQ4_NL"
        out_dim, in_dim = self.shape
        return iq4nl_gemv(x, self.u8, self.sc, self.ks, out_dim, in_dim, self.dev)


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
            self.d = Tensor(blk[:, 208:210].copy().view("<f2")[:, 0].astype(np.float32), device=dev).realize()
            self.ql = Tensor(blk[:, 0:128], device=dev, dtype=dtypes.uint8).realize()
            self.qh = Tensor(blk[:, 128:192], device=dev, dtype=dtypes.uint8).realize()
            self.sc = Tensor(blk[:, 192:208].astype(np.int8).astype(np.float32), device=dev).realize()  # (nb,16)
        else:
            raise ValueError(f"_KQuant: unsupported type {gtype}")

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
        """Single-step in-graph GDN scan (T=1; the adapter processes one row at a time)."""
        r = self.a
        q, k, v, alpha, beta, out_gate, state, _ = self._attn_pre(x, start_pos)
        V, Vd, K = r.num_v_heads, r.head_v_dim, r.head_k_dim
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
        return self._attn_post(core, out_gate, start_pos)

    def _ffn(self, x: Tensor) -> Tensor:
        r = self.a
        w = self.w
        return r._mm(r._mm(x, w["ffn_gate"]).silu() * r._mm(x, w["ffn_up"]), w["ffn_down"])

    def __call__(self, x: Tensor, start_pos) -> Tensor:
        r = self.a
        h = x + self._attn(_rms(x, self.w["norm"], r.norm_eps), start_pos)
        return (h + self._ffn(_rms(h, self.w["ffn_norm"], r.norm_eps))).contiguous()


class Qwen35TextAdapter:
    """Layer-sharded Qwen3.5-family inference adapter.

    Weights are loaded once into persistent VRAM (quantized types stay u8 and
    dequant on-device per matmul; IQ4_NL single-row matmuls use the fused
    GEMV custom kernel).  Rows are processed one token at a time through one
    TinyJit per entry point, so decode is kernel-graph-replay, not per-op
    numpy ping-pong.
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
        self._rstride = 4

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
        self._rstride = int(os.environ.get("QWEN35_RSTRIDE", "4"))
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
        """Matmul x @ w.T — fused IQ4_NL GEMV for single-row decode when enabled.

        f32 path (FullAttn projections + FFNs): parity 5e-7, oracle-safe.
        f16 path (GDN attn projections): fused only for big models (QWEN35_FUSE_F16=auto
        -> GGUF >= 8GiB); the f16 matmul differs by 1 f16 ULP on ~0.1% of outputs
        (accumulation order) — 20x smaller than the IQ4_NL quantization error."""
        if (isinstance(w, _QWeight) and w.gtype == "IQ4_NL" and str(self.dev).startswith("CUDA")
                and x.numel() == w.shape[1] and (x.dtype == dtypes.float32 or (x.dtype == dtypes.float16 and self._fuse_f16))):
            return w.gemv(x)
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

    def _step(self, key: str, inp: Tensor, pos: int) -> Tensor:
        jit = self._jits.get(key)
        if jit is None:
            fn = {"ids": self._forward_ids, "hid": self._forward_hid, "log": self._forward_logits}[key]
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
