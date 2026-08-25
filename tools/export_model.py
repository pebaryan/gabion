#!/usr/bin/env python3
"""Export a BBT-architecture transformer to the gabion browser wire format.

Produces a single JSON file consumed by gabion/web/model_loader.js:

    {
      "config":   {vocab_size, d_model, n_heads, n_layers, seq_len, d_ff,
                   tie_weights, act_quant, rope_base, tokenizer?},
      "weights_b64": base64 of little-endian f16 weights, flat in gabion
                     loadFlatWeights order: tok_emb, [per-layer q,k,v,o,norm1,
                     gate_up,norm2,down]*, norm_f, [lm_head],
      "vocab"?, "merges"?: optional GPT-2 BPE tables (tokenizer)

Sources:
  --from-tinygrad PATH   npz or safetensors checkpoint with tensors named
                         tok_emb, layer{i}.q/k/v/o/norm1/gate_up/norm2/down,
                         norm_f, lm_head (tie_weights uses tok_emb).
  --from-gguf PATH       llama-family GGUF file. Supports 26 quant types
                         (F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K/
                         Q4_K/Q5_K/Q6_K/TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/XS/S/
                         IQ3_XXS/S/IQ4_NL/XS/MXFP4/NVFP4), GQA, untied lm_head,
                         embedded BPE tokenizer.
  --from-hf DIR          HuggingFace checkpoint dir (config.json +
                         model.safetensors[.index.json], optional
                         tokenizer.json). llama/qwen2/qwen3 family, F32/F16/
                         BF16 dtypes, sharded or single-file.

  --with-tokenizer gpt2  downloads vocab.json + merges.txt and embeds them.
  --out PATH             output JSON (default: <source stem>.gabion.json)
"""
from __future__ import annotations

import argparse
import base64
import json
import mmap
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# f16 wire codec (must match tinygrad_v0.js weightsToF16Base64)
# --------------------------------------------------------------------------

def f16_base64(f32: np.ndarray) -> str:
    flat = np.asarray(f32, dtype=np.float32).reshape(-1)
    half = flat.astype(np.float16)  # IEEE 754 half (round-to-nearest-even)
    return base64.b64encode(half.view("<u2").tobytes()).decode("ascii")


def f16_decode(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    u16 = np.frombuffer(raw, dtype="<u2")
    return u16.view(np.float16).astype(np.float32)


# --------------------------------------------------------------------------
# GGUF reader (subset: metadata + F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/
# Q3_K/Q4_K/Q5_K/Q6_K/TQ1_0/TQ2_0/BF16/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/
# IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS/MXFP4/NVFP4 tensors). Every dequant is
# byte-exact against gguf-py -- see tests/test_export_model.py::
# test_dequant_matches_gguf_py and tests/_check_real_iq.py (real files).
# --------------------------------------------------------------------------

# Current ggml/GGUF type enum (gguf-py GGMLQuantizationType; note the 2025
# renumbering: Q5_0=6, Q8_0=8, Q6_K=14, TQ1_0=34, TQ2_0=35)
GGML_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    41: "Q1_0",
}

GGUF_META_TYPES = {  # value_type -> (struct fmt, size) for scalars
    0: ("B", 1), 1: ("b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}


def _read_string(buf: bytes, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    return buf[off:off + n].decode("utf-8", errors="replace"), off + n


def _read_meta_value(buf: bytes, off: int, vtype: int) -> tuple[object, int]:
    if vtype == 8:  # string
        return _read_string(buf, off)
    if vtype == 9:  # array
        (etype, n) = struct.unpack_from("<IQ", buf, off)
        off += 12
        items = []
        for _ in range(n):
            item, off = _read_meta_value(buf, off, etype)
            items.append(item)
        return items, off
    fmt, size = GGUF_META_TYPES[vtype]
    (val,) = struct.unpack_from(fmt, buf, off)
    return val, off + size


def parse_gguf(path: Path) -> tuple[dict, list[tuple[str, tuple, str, int]], int]:
    """Returns (metadata, tensor_infos, data_base). tensor_infos are
    (name, dims_tuple, ggml_type, byte_offset) with byte_offset RELATIVE to the
    tensor data section (GGUF spec; llama.cpp writers use this). data_base is the
    absolute file position where the data section starts."""
    buf = Path(path).read_bytes()
    assert buf[:4] == b"GGUF", "not a GGUF file"
    (version, n_tensors, n_kv) = struct.unpack_from("<IQQ", buf, 4)
    if version != 3:
        print(f"[warn] GGUF version {version} (expected 3); attempting anyway")
    off = 4 + 4 + 8 + 8
    meta = {}
    for _ in range(n_kv):
        key, off = _read_string(buf, off)
        (vtype,) = struct.unpack_from("<I", buf, off)
        off += 4
        val, off = _read_meta_value(buf, off, vtype)
        meta[key] = val
    infos = []
    for _ in range(n_tensors):
        name, off = _read_string(buf, off)
        (n_dims,) = struct.unpack_from("<I", buf, off)
        off += 4
        dims = struct.unpack_from(f"<{'Q' * n_dims}", buf, off)
        off += 8 * n_dims
        (gtype, toff) = struct.unpack_from("<IQ", buf, off)
        off += 12
        infos.append((name, tuple(dims), GGML_TYPE.get(gtype, f"T{gtype}"), toff))
    # The data section is aligned to general.alignment (default 32, per spec).
    # Hardcoding 32 silently reads every tensor from the wrong offset in a file
    # that declares 64 -- the shapes still fit, so nothing raises.
    align = int(meta.get("general.alignment") or 32)
    if align <= 0 or (align & (align - 1)):
        raise ValueError(f"general.alignment must be a positive power of two, got {align}")
    return meta, infos, (off + align - 1) & ~(align - 1)


def _dequant(name: str, raw: bytes, dims: tuple, gtype: str) -> np.ndarray:
    # GGUF dims are most-significant-first; numpy wants row-major order.
    shape = tuple(dims[::-1])
    n = int(np.prod(shape))
    if gtype == "F32":
        return np.frombuffer(raw, dtype="<f4", count=n).reshape(shape)
    if gtype == "F16":
        return np.frombuffer(raw, dtype="<f2", count=n).astype(np.float32).reshape(shape)
    if gtype == "Q8_0":
        # block of 32: f16 d + 32 x i8 (34 B). value = x * d
        nb = n // 32
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 34).reshape(nb, 34)
        d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
        qs = blk[:, 2:].copy().view(np.int8).astype(np.float32)
        return (qs * d[:, None]).reshape(shape)
    if gtype == "Q4_0":
        # block of 32: f16 d + 16 nibble bytes (18 B). ggml puts the LOW nibble of
        # byte j at element j and the HIGH nibble at element j+16 -- the halves are
        # split, not interleaved. value = (x - 8) * d
        nb = n // 32
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 18).reshape(nb, 18)
        d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
        qs = blk[:, 2:]
        out = np.empty((nb, 32), dtype=np.float32)
        out[:, :16] = ((qs & 0x0F).astype(np.int32) - 8) * d[:, None]
        out[:, 16:] = ((qs >> 4).astype(np.int32) - 8) * d[:, None]
        return out.reshape(shape)
    if gtype == "Q4_1":
        # block of 32: f16 d + f16 m + 16 nibble bytes (20 B); same split-nibble
        # layout as Q4_0. value = x*d + m
        nb = n // 32
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 20).reshape(nb, 20)
        d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
        m = blk[:, 2:4].copy().view("<f2")[:, 0].astype(np.float32)
        qs = blk[:, 4:]
        out = np.empty((nb, 32), dtype=np.float32)
        out[:, :16] = (qs & 0x0F).astype(np.float32) * d[:, None] + m[:, None]
        out[:, 16:] = (qs >> 4).astype(np.float32) * d[:, None] + m[:, None]
        return out.reshape(shape)
    if gtype in ("Q4_K", "Q6_K"):
        return _dequant_k(raw, n, gtype).reshape(shape)
    if gtype in ("Q2_K", "TQ2_0"):
        return _dequant_k2(raw, n, gtype).reshape(shape)
    if gtype in ("Q3_K", "Q5_K"):
        return _dequant_k3(raw, n, gtype).reshape(shape)
    if gtype in ("Q5_1", "BF16", "TQ1_0", "IQ4_NL", "MXFP4", "NVFP4"):
        return _dequant_simple(raw, n, gtype).reshape(shape)
    if gtype in ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M", "IQ4_XS"):
        return _dequant_iq(raw, n, gtype).reshape(shape)
    if gtype == "Q5_0":
        # block of 32: f16 d + u32 qh + qs[16] (22 B)
        nb = n // 32
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 22).reshape(nb, 22)
        d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
        qh = blk[:, 2:6].copy().view("<u4")[:, 0].astype(np.uint32)
        qs = blk[:, 6:]
        j = np.arange(16)
        xh0 = (((qh[:, None] >> j) << 4) & 0x10).astype(np.int32)
        xh1 = ((qh[:, None] >> (j + 12)) & 0x10).astype(np.int32)
        lo = ((qs & 0x0F).astype(np.int32) | xh0) - 16
        hi = ((qs >> 4).astype(np.int32) | xh1) - 16
        out = np.empty((nb, 32), dtype=np.float32)
        out[:, :16] = lo * d[:, None]
        out[:, 16:] = hi * d[:, None]
        return out.reshape(shape)
    raise NotImplementedError(f"GGUF tensor '{name}': unsupported type {gtype}")


def _dequant_k(raw: bytes, n: int, gtype: str) -> np.ndarray:
    """Q4_K / Q6_K dequant (llama.cpp ggml-quants.c, QK_K = 256). Vectorized."""
    nb = n // 256
    if gtype == "Q4_K":
        # super-block: f16 d, f16 dmin, 12 scale bytes, 128 qs bytes (144 B)
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 144).reshape(nb, 144)
        d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
        dmin = blk[:, 2:4].copy().view("<f2")[:, 0].astype(np.float32)
        scales = blk[:, 4:16]
        qs = blk[:, 16:]
        # get_scale_min_k4(j): j<4 -> d=q[j]&63, m=q[j+4]&63
        #                    else   -> d=(q[j+4]&0xF)|((q[j-4]>>6)<<4), m=(q[j+4]>>4)|((q[j]>>6)<<4)
        sc = np.empty((nb, 8), dtype=np.uint16)
        mn = np.empty((nb, 8), dtype=np.uint16)
        sc[:, 0] = scales[:, 0] & 63; mn[:, 0] = scales[:, 4] & 63
        sc[:, 1] = scales[:, 1] & 63; mn[:, 1] = scales[:, 5] & 63
        sc[:, 2] = scales[:, 2] & 63; mn[:, 2] = scales[:, 6] & 63
        sc[:, 3] = scales[:, 3] & 63; mn[:, 3] = scales[:, 7] & 63
        sc[:, 4] = (scales[:, 8] & 0xF) | ((scales[:, 0] >> 6) << 4); mn[:, 4] = (scales[:, 8] >> 4) | ((scales[:, 4] >> 6) << 4)
        sc[:, 5] = (scales[:, 9] & 0xF) | ((scales[:, 1] >> 6) << 4); mn[:, 5] = (scales[:, 9] >> 4) | ((scales[:, 5] >> 6) << 4)
        sc[:, 6] = (scales[:, 10] & 0xF) | ((scales[:, 2] >> 6) << 4); mn[:, 6] = (scales[:, 10] >> 4) | ((scales[:, 6] >> 6) << 4)
        sc[:, 7] = (scales[:, 11] & 0xF) | ((scales[:, 3] >> 6) << 4); mn[:, 7] = (scales[:, 11] >> 4) | ((scales[:, 7] >> 6) << 4)
        qs32 = qs.reshape(nb, 4, 32)
        vals = np.empty((nb, 256), dtype=np.float32)
        for b in range(4):
            lo = (qs32[:, b] & 0x0F).astype(np.float32)
            hi = (qs32[:, b] >> 4).astype(np.float32)
            d1 = (d * sc[:, 2 * b]).astype(np.float32); m1 = (dmin * mn[:, 2 * b]).astype(np.float32)
            d2 = (d * sc[:, 2 * b + 1]).astype(np.float32); m2 = (dmin * mn[:, 2 * b + 1]).astype(np.float32)
            vals[:, b * 64:b * 64 + 32] = d1[:, None] * lo - m1[:, None]
            vals[:, b * 64 + 32:b * 64 + 64] = d2[:, None] * hi - m2[:, None]
        return vals.reshape(-1)
    # Q6_K: FILE layout (gguf-py/gguf.c): ql[128], qh[64], scales[16], d[2] (210 B)
    # — note d is at the END of the block, unlike the in-memory C struct.
    blk = np.frombuffer(raw, dtype="<u1", count=nb * 210).reshape(nb, 210)
    f16s = blk[:, 208:210].copy().view("<f2")[:, 0].astype(np.float32)
    ql = blk[:, 0:128]
    qh = blk[:, 128:192]
    sc = blk[:, 192:208].astype(np.int8).astype(np.float32)
    out = np.empty((nb, 256), dtype=np.float32)
    for half in range(2):
        q1 = ((ql[:, half * 64:half * 64 + 32] & 0x0F) | ((qh[:, half * 32:half * 32 + 32] & 0x03) << 4)).astype(np.float32) - 32
        q2 = ((ql[:, half * 64 + 32:half * 64 + 64] & 0x0F) | ((qh[:, half * 32:half * 32 + 32] >> 2) & 0x03) << 4).astype(np.float32) - 32
        q3 = ((ql[:, half * 64:half * 64 + 32] >> 4) | ((qh[:, half * 32:half * 32 + 32] >> 4) & 0x03) << 4).astype(np.float32) - 32
        q4 = ((ql[:, half * 64 + 32:half * 64 + 64] >> 4) | ((qh[:, half * 32:half * 32 + 32] >> 6) & 0x03) << 4).astype(np.float32) - 32
        s1 = sc[:, half * 8:half * 8 + 2].repeat(16, axis=1)
        s2 = sc[:, half * 8 + 2:half * 8 + 4].repeat(16, axis=1)
        s3 = sc[:, half * 8 + 4:half * 8 + 6].repeat(16, axis=1)
        s4 = sc[:, half * 8 + 6:half * 8 + 8].repeat(16, axis=1)
        base = half * 128
        out[:, base:base + 32] = f16s[:, None] * s1 * q1
        out[:, base + 32:base + 64] = f16s[:, None] * s2 * q2
        out[:, base + 64:base + 96] = f16s[:, None] * s3 * q3
        out[:, base + 96:base + 128] = f16s[:, None] * s4 * q4
    return out.reshape(-1)


def _dequant_k2(raw: bytes, n: int, gtype: str) -> np.ndarray:
    """Q2_K / TQ2_0 dequant (llama.cpp ggml-quants.c). Vectorized."""
    nb = n // 256
    if gtype == "Q2_K":
        # FILE layout (gguf-py/gguf.c): scales[16], qs[64], d[2], dmin[2] (84 B)
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 84).reshape(nb, 84)
        scales = blk[:, 0:16]
        qs = blk[:, 16:80]
        d = blk[:, 80:82].copy().view("<f2")[:, 0].astype(np.float32)
        dmin = blk[:, 82:84].copy().view("<f2")[:, 0].astype(np.float32)
        # per 128-half: 8 scales (scales[8h..8h+7]); value v (0..127):
        #   j = v//32, sub = v%32; byte = q[sub] (q[l]/q[l+16], l=sub%16),
        #   shift = j*2, scale = sc[j*2 + sub//16];
        #   val = d*(sc&0xF)*qbits - dmin*(sc>>4)
        v = np.arange(128)
        j = v // 32
        byte_idx = v % 32
        shift = j * 2
        sc_idx = j * 2 + (v % 32) // 16
        out = np.empty((nb, 256), dtype=np.float32)
        for half in range(2):
            sc = scales[:, half * 8:half * 8 + 8]
            dl = (d[:, None] * (sc[:, sc_idx] & 0x0F)).astype(np.float32)
            ml = (dmin[:, None] * (sc[:, sc_idx] >> 4)).astype(np.float32)
            qb = (qs[:, half * 32:half * 32 + 32][:, byte_idx] >> shift) & 0x03
            out[:, half * 128:half * 128 + 128] = dl * qb.astype(np.float32) - ml
        return out.reshape(-1)
    # TQ2_0: FILE layout: qs[64], d[2] (66 B); val = ((2-bit) - 1) * d.
    # ggml's dequantize_row_tq2_0 loops byte-group j (2 x 32 bytes) -> bit plane l
    # (4) -> byte m (32), so element j*128 + l*32 + m reads qs[j*32 + m] >> 2l.
    blk = np.frombuffer(raw, dtype="<u1", count=nb * 66).reshape(nb, 66)
    f16s = blk[:, 64:66].copy().view("<f2")[:, 0].astype(np.float32)
    qs = blk[:, 0:64]
    o = np.arange(256)
    byte = (o // 128) * 32 + (o % 32)
    shift = 2 * ((o % 128) // 32)
    bits = (qs[:, byte] >> shift) & 0x03
    return ((bits.astype(np.float32) - 1.0) * f16s[:, None]).reshape(-1)


# --------------------------------------------------------------------------
# New quant family (Q3_K / Q5_K / TQ1_0 / BF16 / Q5_1 / IQ* / MXFP4 / NVFP4).
# Every dequant below is a direct port of gguf-py's gguf.quants classes and is
# verified byte-exact against gguf-py on random data (test_dequant_matches_gguf_py).
# --------------------------------------------------------------------------

try:  # repo-root on sys.path (PYTHONPATH="D:/code/gabion") or script dir
    import _iq_tables as IQT
except ImportError:
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(__file__))
    import _iq_tables as IQT


def _iq_grid(grid_hex: bytes, grid_map, grid_shape) -> np.ndarray:
    """Decode a llama.cpp iq grid (hex-encoded packed table) into the float
    lookup table. Port of gguf-py __Quant.init_grid. Returns (1,1,R,C)."""
    bits_per_elem = int(np.ceil(np.log2(len(grid_map))))
    elems_per_byte = 8 // bits_per_elem
    g = np.frombuffer(grid_hex, dtype=np.uint8)
    g = g.reshape((-1, 2))
    g = (np.where(g > 0x40, g + 9, g) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
    g = g[..., 0] | g[..., 1]
    g = g.reshape((-1, 1)) >> np.array(list(range(0, 8, 8 // elems_per_byte)), dtype=np.uint8).reshape((1, elems_per_byte))
    g = (g & ((1 << bits_per_elem) - 1)).reshape((-1, 1))
    gm = np.array(grid_map, dtype=np.float32).reshape((1, -1))
    g = np.take_along_axis(gm, g, axis=-1)
    return g.reshape((1, 1, *grid_shape))


_IQ_GRIDS = {
    "IQ2_XXS": _iq_grid(IQT.IQ2_XXS_GRID_HEX, IQT.IQ2_XXS_GRID_MAP, IQT.IQ2_XXS_GRID_SHAPE),
    "IQ2_XS": _iq_grid(IQT.IQ2_XS_GRID_HEX, IQT.IQ2_XS_GRID_MAP, IQT.IQ2_XS_GRID_SHAPE),
    "IQ2_S": _iq_grid(IQT.IQ2_S_GRID_HEX, IQT.IQ2_S_GRID_MAP, IQT.IQ2_S_GRID_SHAPE),
    "IQ3_XXS": _iq_grid(IQT.IQ3_XXS_GRID_HEX, IQT.IQ3_XXS_GRID_MAP, IQT.IQ3_XXS_GRID_SHAPE),
    "IQ3_S": _iq_grid(IQT.IQ3_S_GRID_HEX, IQT.IQ3_S_GRID_MAP, IQT.IQ3_S_GRID_SHAPE),
    "IQ1_S": _iq_grid(IQT.IQ1_S_GRID_HEX, IQT.IQ1_S_GRID_MAP, IQT.IQ1_S_GRID_SHAPE),
    "IQ1_M": _iq_grid(IQT.IQ1_S_GRID_HEX, IQT.IQ1_S_GRID_MAP, IQT.IQ1_S_GRID_SHAPE),
}
_KSIGNS = np.frombuffer(IQT.KSIGNS, dtype=np.uint8).reshape((1, 1, 1, 128))
_IQ4_KVALUES = np.array(IQT.IQ4_NL_KVALUES, dtype=np.int8).reshape((1, 1, 16))
_IQ1_DELTA = np.float32(IQT.IQ1_S_DELTA)


def _dequant_k3(raw: bytes, n: int, gtype: str) -> np.ndarray:
    """Q3_K / Q5_K (llama.cpp ggml-quants.c, QK_K = 256). Vectorized."""
    nb = n // 256
    if gtype == "Q3_K":
        # FILE layout: hmask[32], qs[64], scales[12], d[2] (110 B) — d at END.
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 110).reshape(nb, 110)
        hmask = blk[:, 0:32]
        qs = blk[:, 32:96]
        scales = blk[:, 96:108]
        d = blk[:, 108:110].copy().view("<f2")[:, 0].astype(np.float32)
        # scales packed 6-bit: low nibbles in bytes 0..7, high 2 bits in 8..11
        lscales = scales[:, 0:8].reshape(nb, 1, 8) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
        lscales = lscales.reshape(nb, 16)
        hscales = scales[:, 8:12].reshape(nb, 1, 4) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4, 1))
        hscales = hscales.reshape(nb, 16)
        sc = (lscales & 0x0F) | ((hscales & 0x03) << 4)
        sc = (sc.astype(np.int8) - np.int8(32)).astype(np.float32)
        dl = (d[:, None] * sc).reshape(nb, 16, 1)
        ql = qs.reshape(nb, -1, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = hmask.reshape(nb, -1, 1, 32) >> np.array(list(range(8)), dtype=np.uint8).reshape((1, 1, 8, 1))
        ql = ql.reshape(nb, 16, 16) & 0x03
        qh = (qh.reshape(nb, 16, 16) & 0x01) ^ 0x01  # offset is zero when the bitmask is 1
        q = (ql.astype(np.int8) - (qh << 2).astype(np.int8)).astype(np.float32)
        return (dl * q).reshape(-1)
    # Q5_K: FILE layout: d[2], dmin[2], scales[12], qh[32], qs[128] (176 B)
    blk = np.frombuffer(raw, dtype="<u1", count=nb * 176).reshape(nb, 176)
    d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
    dmin = blk[:, 2:4].copy().view("<f2")[:, 0].astype(np.float32)
    scales = blk[:, 4:16]
    qh = blk[:, 16:48]
    qs = blk[:, 48:176]
    # get_scale_min_k4 over the 12 scale bytes (see _dequant_k)
    sc = np.empty((nb, 8), dtype=np.uint16)
    mn = np.empty((nb, 8), dtype=np.uint16)
    sc[:, 0] = scales[:, 0] & 63; mn[:, 0] = scales[:, 4] & 63
    sc[:, 1] = scales[:, 1] & 63; mn[:, 1] = scales[:, 5] & 63
    sc[:, 2] = scales[:, 2] & 63; mn[:, 2] = scales[:, 6] & 63
    sc[:, 3] = scales[:, 3] & 63; mn[:, 3] = scales[:, 7] & 63
    sc[:, 4] = (scales[:, 8] & 0xF) | ((scales[:, 0] >> 6) << 4); mn[:, 4] = (scales[:, 8] >> 4) | ((scales[:, 4] >> 6) << 4)
    sc[:, 5] = (scales[:, 9] & 0xF) | ((scales[:, 1] >> 6) << 4); mn[:, 5] = (scales[:, 9] >> 4) | ((scales[:, 5] >> 6) << 4)
    sc[:, 6] = (scales[:, 10] & 0xF) | ((scales[:, 2] >> 6) << 4); mn[:, 6] = (scales[:, 10] >> 4) | ((scales[:, 6] >> 6) << 4)
    sc[:, 7] = (scales[:, 11] & 0xF) | ((scales[:, 3] >> 6) << 4); mn[:, 7] = (scales[:, 11] >> 4) | ((scales[:, 7] >> 6) << 4)
    d = (d[:, None] * sc.astype(np.float32)).reshape(nb, -1, 1)
    dm = (dmin[:, None] * mn.astype(np.float32)).reshape(nb, -1, 1)
    ql = qs.reshape(nb, -1, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qhb = qh.reshape(nb, -1, 1, 32) >> np.array(list(range(8)), dtype=np.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape(nb, -1, 32)
    qhb = (qhb & 0x01).reshape(nb, -1, 32)
    q = (ql | (qhb << 4)).astype(np.float32)
    return (d * q - dm).reshape(-1)


def _dequant_simple(raw: bytes, n: int, gtype: str) -> np.ndarray:
    """Q5_1 / BF16 / TQ1_0 / IQ4_NL / MXFP4 / NVFP4 — the small-block types."""
    if gtype == "BF16":
        return (np.frombuffer(raw, dtype="<i2", count=n).astype(np.int32) << 16).view(np.float32)
    if gtype == "Q5_1":
        # block of 32: f16 d, f16 m, u32 qh, qs[16] (24 B); value = d*q + m
        # NOTE: unlike Q5_0, qh bit j is the 5th bit of ELEMENT j (0..31) —
        # bits 0..15 go to the low half, bits 16..31 to the high half.
        nb = n // 32
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 24).reshape(nb, 24)
        d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
        m = blk[:, 2:4].copy().view("<f2")[:, 0].astype(np.float32)
        qh = blk[:, 4:8].copy().view("<u4")[:, 0].astype(np.uint32)
        qs = blk[:, 8:]
        j32 = np.arange(32)
        qhb = ((qh[:, None] >> j32) & 0x01).astype(np.int32)
        lo = ((qs & 0x0F).astype(np.int32) | (qhb[:, :16] << 4))
        hi = ((qs >> 4).astype(np.int32) | (qhb[:, 16:] << 4))
        out = np.empty((nb, 32), dtype=np.float32)
        out[:, :16] = lo * d[:, None] + m[:, None]
        out[:, 16:] = hi * d[:, None] + m[:, None]
        return out.reshape(-1)
    if gtype == "TQ1_0":
        # FILE layout: qs[48], qh[4], d[2] (54 B); base-3 packed, val = (code-1)*d
        nb = n // 256
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 54).reshape(nb, 54)
        d = blk[:, 52:54].copy().view("<f2")[:, 0].astype(np.float32)
        qs0 = blk[:, 0:32]
        qs1 = blk[:, 32:48]
        qh = blk[:, 48:52]
        q0 = (qs0.reshape(nb, -1, 1, 32) * np.array([1, 3, 9, 27, 81], dtype=np.uint8).reshape(1, 1, 5, 1)).reshape(nb, -1)
        q1 = (qs1.reshape(nb, -1, 1, 16) * np.array([1, 3, 9, 27, 81], dtype=np.uint8).reshape(1, 1, 5, 1)).reshape(nb, -1)
        q2 = (qh.reshape(nb, -1, 1, 4) * np.array([1, 3, 9, 27], dtype=np.uint8).reshape(1, 1, 4, 1)).reshape(nb, -1)
        qs = np.concatenate([q0, q1, q2], axis=-1)
        qs = ((qs.astype(np.uint16) * 3) >> 8).astype(np.int8) - np.int8(1)
        return (d[:, None] * qs.astype(np.float32)).reshape(-1)
    if gtype == "IQ4_NL":
        # block of 32: f16 d + qs[16] nibbles (18 B); 16-entry kvalue lookup
        nb = n // 32
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 18).reshape(nb, 18)
        d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
        qs = blk[:, 2:]
        q = (qs.reshape(nb, -1, 1, 16) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & 0x0F
        q = q.reshape(nb, 32, 1)
        q = np.take_along_axis(_IQ4_KVALUES, q, axis=-1).astype(np.float32).reshape(nb, 32)
        return (d[:, None] * q).reshape(-1)
    if gtype == "MXFP4":
        # block of 32: e8m0 exponent byte + 16 nibble bytes (17 B); e2m1 kvalues
        nb = n // 32
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 17).reshape(nb, 17)
        e = blk[:, 0:1].astype(np.uint32)
        qs = blk[:, 1:]
        bits = np.where(e < 2, np.uint32(0x00200000) << e, np.uint32(e - 1) << np.uint32(23))
        d = bits.view(np.float32)
        q = (qs.reshape(nb, 1, 16) >> np.array([0, 4], dtype=np.uint8).reshape(1, 2, 1)) & 0x0F
        q = q.view(np.int8)
        q = np.take_along_axis(np.array(IQT.MXFP4_KVALUES, dtype=np.int8).reshape(1, 1, 16), q, axis=-1).reshape(nb, 32)
        return (d * q.astype(np.float32)).reshape(-1)
    # NVFP4: block of 64: 4x ue4m3 scales + 32 nibble bytes (36 B)
    nb = n // 64
    blk = np.frombuffer(raw, dtype="<u1", count=nb * 36).reshape(nb, 36)
    db = blk[:, 0:4]
    qs = blk[:, 4:]
    exp = (db >> 3).astype(np.int32) & 0xF
    man = (db & 0x7).astype(np.float32)
    rawd = np.where(exp == 0, man * 2.0 ** -9, (1.0 + man / 8.0) * (2.0 ** (exp.astype(np.float32) - 7)))
    d = np.where((db == 0) | (db == 0x7F), 0.0, rawd * 0.5).reshape(nb, 4, 1)
    qb = qs.reshape(nb, 4, 8)
    lo = (qb & 0x0F).view(np.int8)
    hi = (qb >> 4).view(np.int8)
    vals = np.concatenate([lo, hi], axis=-1)
    vals = np.take_along_axis(np.array(IQT.NVFP4_KVALUES, dtype=np.int8).reshape(1, 1, 16), vals, axis=-1)
    return (d * vals.astype(np.float32)).reshape(-1)


def _dequant_iq(raw: bytes, n: int, gtype: str) -> np.ndarray:
    """The i-quants (IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M/IQ4_XS).

    Port of gguf-py's dequantize_blocks. All blocks are QK_K = 256 elements;
    grid rows are indexed by raw packed codes (bytes/words), signs come from a
    ksigns table or plain bit masks, and each sub-block has its own scale.
    """
    nb = n // 256
    if gtype == "IQ2_XXS":
        # d[2] + qs[64] (66 B): 8 u32 pairs; low u32 = 4 grid indices, high = signs+scale
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 66).reshape(nb, 66)
        d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
        qs = blk[:, 2:].copy().view("<u4").reshape(nb, -1, 2)
        db = d[:, None] * (np.float32(0.5) + (qs[..., 1] >> 28).astype(np.float32)) * np.float32(0.25)
        db = db.reshape(nb, -1, 1, 1)
        signs = (qs[..., 1].reshape(nb, -1, 1) >> np.array([0, 7, 14, 21], dtype=np.uint32).reshape(1, 1, 4)) & 0x7F
        signs = signs.reshape(nb, -1, 4, 1)
        signs = np.take_along_axis(_KSIGNS, signs, axis=-1)
        signs = signs.reshape(nb, -1, 4, 1) >> np.array(list(range(8)), dtype=np.uint8).reshape(1, 1, 1, 8)
        signs = np.where((signs & 0x01) == 0, np.float32(1.0), np.float32(-1.0))
        signs = signs.reshape(nb, -1, 4, 8)
        grid = np.take_along_axis(_IQ_GRIDS["IQ2_XXS"], qs[..., 0].copy().view(np.uint8).reshape(nb, -1, 1, 1), axis=-2)
        grid = grid.reshape(nb, -1, 4, 8)
        return (db * grid * signs).reshape(-1)
    if gtype == "IQ2_XS":
        # d[2] + qs[64] (u16) + scales[8] (74 B)
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 74).reshape(nb, 74)
        d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
        qs = blk[:, 2:66].copy().view("<u2")
        scales = blk[:, 66:74]
        scales = (scales.reshape(nb, -1, 1) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2)) & 0x0F
        scales = scales.reshape(nb, -1)
        db = d[:, None] * (np.float32(0.5) + scales) * np.float32(0.25)
        db = db.reshape(nb, -1, 1, 1)
        signs = np.take_along_axis(_KSIGNS.reshape(1, 1, 128), (qs >> 9).reshape(nb, -1, 1), axis=-1)
        signs = signs.reshape(nb, -1, 1) >> np.array(list(range(8)), dtype=np.uint8).reshape(1, 1, 8)
        signs = np.where((signs & 0x01) == 0, np.float32(1.0), np.float32(-1.0))
        signs = signs.reshape(nb, -1, 2, 8)
        grid = np.take_along_axis(_IQ_GRIDS["IQ2_XS"], (qs & np.uint16(511)).reshape(nb, -1, 1, 1), axis=-2)
        grid = grid.reshape(nb, -1, 2, 8)
        return (db * grid * signs).reshape(-1)
    if gtype == "IQ2_S":
        # d[2] + qs[32] + signs[32] + qh[8] + scales[8] (82 B)
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 82).reshape(nb, 82)
        d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
        qs = blk[:, 2:34]
        sgn = blk[:, 34:66]
        qh = blk[:, 66:74]
        scales = blk[:, 74:82]
        scales = (scales.reshape(nb, -1, 1) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2)) & 0x0F
        db = d[:, None] * (np.float32(0.5) + scales.reshape(nb, -1)) * np.float32(0.25)
        db = db.reshape(nb, -1, 1, 1)
        signs = (sgn.reshape(nb, -1, 1) >> np.array(list(range(8)), dtype=np.uint8).reshape(1, 1, 8)) & 0x01
        signs = np.where(signs == 0, np.float32(1.0), np.float32(-1.0)).reshape(nb, -1, 2, 8)
        qh2 = (qh.reshape(nb, -1, 1) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4)) & 0x03
        qi = qs.astype(np.uint16) | (qh2.astype(np.uint16) << 8).reshape(nb, -1)
        grid = np.take_along_axis(_IQ_GRIDS["IQ2_S"], qi.reshape(nb, -1, 1, 1), axis=-2)
        grid = grid.reshape(nb, -1, 2, 8)
        return (db * grid * signs).reshape(-1)
    if gtype == "IQ3_XXS":
        # d[2] + qs[64] + scales[32] (u32) (98 B)
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 98).reshape(nb, 98)
        d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
        qs = blk[:, 2:66]
        scales = blk[:, 66:98].copy().view("<u4")
        db = d[:, None] * (np.float32(0.5) + (scales >> 28).astype(np.float32)) * np.float32(0.5)
        db = db.reshape(nb, -1, 1, 1)
        signs = (scales.reshape(nb, -1, 1) >> np.array([0, 7, 14, 21], dtype=np.uint32).reshape(1, 1, 4)) & 0x7F
        signs = signs.reshape(nb, -1, 4, 1)
        signs = np.take_along_axis(_KSIGNS, signs, axis=-1)
        signs = signs.reshape(nb, -1, 4, 1) >> np.array(list(range(8)), dtype=np.uint8).reshape(1, 1, 1, 8)
        signs = np.where((signs & 0x01) == 0, np.float32(1.0), np.float32(-1.0))
        signs = signs.reshape(nb, -1, 4, 8)
        grid = np.take_along_axis(_IQ_GRIDS["IQ3_XXS"], qs.reshape(nb, -1, 1, 1), axis=-2)
        grid = grid.reshape(nb, -1, 4, 8)
        return (db * grid * signs).reshape(-1)
    if gtype == "IQ3_S":
        # d[2] + qs[64] + qh[8] + signs[32] + scales[4] (110 B)
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 110).reshape(nb, 110)
        d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
        qs = blk[:, 2:66]
        qh = blk[:, 66:74]
        sgn = blk[:, 74:106]
        scales = blk[:, 106:110]
        scales = (scales.reshape(nb, -1, 1) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2)) & 0x0F
        db = d[:, None] * (1 + 2 * scales.reshape(nb, -1))
        db = db.reshape(nb, -1, 1, 1)
        signs = (sgn.reshape(nb, -1, 1) >> np.array(list(range(8)), dtype=np.uint8).reshape(1, 1, 8)) & 0x01
        signs = np.where(signs == 0, np.float32(1.0), np.float32(-1.0)).reshape(nb, -1, 4, 8)
        qhb = ((qh.reshape(nb, -1, 1) >> np.array(list(range(8)), dtype=np.uint8).reshape(1, 1, 8)) & 0x01).astype(np.uint16)
        qi = qs.astype(np.uint16) | (qhb.reshape(nb, -1) << 8)
        grid = np.take_along_axis(_IQ_GRIDS["IQ3_S"], qi.reshape(nb, -1, 1, 1), axis=-2)
        grid = grid.reshape(nb, -1, 4, 8)
        return (db * grid * signs).reshape(-1)
    if gtype == "IQ1_S":
        # d[2] + qs[32] + qh[16] (u16) (50 B)
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 50).reshape(nb, 50)
        d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
        qs = blk[:, 2:34]
        qh = blk[:, 34:50].copy().view("<u2")
        dl = d[:, None] * (2 * ((qh >> 12) & 7) + 1)
        dl = dl.reshape(nb, -1, 1, 1)
        delta = np.where((qh & np.uint16(0x8000)) == 0, _IQ1_DELTA, -_IQ1_DELTA).reshape(nb, -1, 1, 1)
        qhb = (qh.reshape(nb, -1, 1) >> np.array([0, 3, 6, 9], dtype=np.uint16).reshape(1, 1, 4)) & 7
        qi = qs.astype(np.uint16) | (qhb.reshape(nb, -1) << 8)
        grid = np.take_along_axis(_IQ_GRIDS["IQ1_S"], qi.reshape(nb, -1, 1, 1), axis=-2)
        grid = grid.reshape(nb, -1, 4, 8)
        return (dl * (grid + delta)).reshape(-1)
    if gtype == "IQ1_M":
        # qs[32] + qh[16] + scales[8] (56 B) — the f16 scale is packed across bytes
        blk = np.frombuffer(raw, dtype="<u1", count=nb * 56).reshape(nb, 56)
        qs = blk[:, 0:32]
        qh = blk[:, 32:48]
        scales = blk[:, 48:56].copy().view("<u2")
        d = ((scales.reshape(nb, 4) & np.uint16(0xF000)) >> np.array([12, 8, 4, 0], dtype=np.uint16).reshape(1, 4))
        d = (d[..., 0] | d[..., 1] | d[..., 2] | d[..., 3]).view("<f2").astype(np.float32).reshape(nb, 1)
        sc = (scales.reshape(nb, -1, 1) >> np.array([0, 3, 6, 9], dtype=np.uint16).reshape(1, 1, 4)) & 7
        dl = d * (2 * sc.reshape(nb, -1) + 1)
        dl = dl.reshape(nb, -1, 2, 1, 1)
        qhb = (qh.reshape(nb, -1, 1) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2)) & 7
        qi = qs.astype(np.uint16) | (qhb.astype(np.uint16).reshape(nb, -1) << 8)
        delta = np.where((qh.reshape(nb, -1, 1) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2)) & 8 == 0, _IQ1_DELTA, -_IQ1_DELTA)
        delta = delta.reshape(nb, -1, 2, 2, 1)
        grid = np.take_along_axis(_IQ_GRIDS["IQ1_S"], qi.reshape(nb, -1, 1, 1), axis=-2)
        grid = grid.reshape(nb, -1, 2, 2, 8)
        return (dl * (grid + delta)).reshape(-1)
    # IQ4_XS: d[2] + scales_h[2] (u16) + scales_l[4] + qs[128] (136 B)
    blk = np.frombuffer(raw, dtype="<u1", count=nb * 136).reshape(nb, 136)
    d = blk[:, 0:2].copy().view("<f2")[:, 0].astype(np.float32)
    scales_h = blk[:, 2:4].copy().view("<u2")
    scales_l = blk[:, 4:8]
    qs = blk[:, 8:136]
    sl = (scales_l.reshape(nb, -1, 1) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2)) & 0x0F
    sh = (scales_h.reshape(nb, 1, -1) >> np.array([2 * i for i in range(8)], dtype=np.uint16).reshape(1, -1, 1))
    sl = sl.reshape(nb, -1) & 0x0F
    sh = sh.reshape(nb, -1).astype(np.uint8) & 0x03
    scales = (sl | (sh << 4)).astype(np.int8) - np.int8(32)
    dl = (d[:, None] * scales.astype(np.float32)).reshape(nb, -1, 1)
    q = (qs.reshape(nb, -1, 1, 16) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)).reshape(nb, -1, 32, 1) & 0x0F
    q = np.take_along_axis(_IQ4_KVALUES.reshape(1, 1, 1, 16), q, axis=-1).astype(np.float32).reshape(nb, -1, 32)
    return (dl * q).reshape(-1)


def _tensor_data(buf: bytes, name: str, dims: tuple, gtype: str, offset: int, base: int = 0) -> np.ndarray:
    if gtype in ("F32", "F16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q4_K", "Q6_K", "Q2_K", "TQ2_0",
                 "Q5_1", "BF16", "Q3_K", "Q5_K", "TQ1_0", "IQ2_XXS", "IQ2_XS", "IQ2_S",
                 "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M", "IQ4_NL", "IQ4_XS", "MXFP4", "NVFP4"):
        # memoryview, not buf[base + offset:]: slicing bytes COPIES the whole tail,
        # which on a multi-hundred-MB GGUF costs a full-file copy per tensor.
        # np.frombuffer and struct.unpack_from both accept a memoryview.
        return _dequant(name, memoryview(buf)[base + offset:], dims, gtype)
    raise NotImplementedError(f"GGUF tensor '{name}': unsupported type {gtype}")


# --------------------------------------------------------------------------
# layout mapping
# --------------------------------------------------------------------------

# JS loadFlatWeights order (see bbt_forward.js):
#   tok_emb, [q,k,v,o,norm1,gate_up,norm2,down] per layer, norm_f, [lm_head]

def _npy(path: Path, name: str) -> np.ndarray:
    return np.load(path)[name]


# safetensors dtype strings (HF spec) -> numpy dtype; BF16 is special-cased
_ST_DTYPES = {
    "F64": "<f8", "F32": "<f4", "F16": "<f2",
    "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "i1",
    "U64": "<u8", "U32": "<u4", "U16": "<u2", "U8": "u1",
}
_ST_UNSUPPORTED = ("F8_E4M3", "F8_E5M2", "F4_E2M1", "BOOL", "U1")


def _st_tensor(buf, hdr_len: int, info: dict) -> np.ndarray:
    """Read one safetensors tensor from a buffer (bytes or mmap).

    data_offsets are byte ranges in the payload; the header is 8-byte length +
    JSON. Returns float32 for BF16, native dtype otherwise."""
    dtype = info["dtype"]
    if dtype in _ST_UNSUPPORTED:
        raise NotImplementedError(f"safetensors dtype {dtype!r} is not supported")
    start, end = info["data_offsets"]
    if dtype == "BF16":
        u = np.frombuffer(buf, dtype="<u2", count=(end - start) // 2, offset=8 + hdr_len + start)
        f = (u.astype(np.int32) << 16).view(np.float32)
        return f.reshape(info["shape"])
    if dtype not in _ST_DTYPES:
        raise NotImplementedError(f"safetensors dtype {dtype!r} is not supported")
    dt = np.dtype(_ST_DTYPES[dtype])
    arr = np.frombuffer(buf, dtype=dt, count=(end - start) // dt.itemsize,
                        offset=8 + hdr_len + start)
    return arr.reshape(info["shape"])


def _safetensors(path: Path, name: str) -> np.ndarray:
    raw = Path(path).read_bytes()
    (hdr_len,) = struct.unpack_from("<Q", raw, 0)
    hdr = json.loads(raw[8:8 + hdr_len])
    info = hdr.get(name)
    if info is None:
        raise KeyError(f"{name} not in safetensors")
    return _st_tensor(raw, hdr_len, info)


def _hf_config(path: Path) -> dict:
    """Read config.json for a llama-family (qwen2/qwen3/llama/gemma) checkpoint.

    Multimodal wrappers (e.g. Qwen3.5) nest the text params under text_config."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(cfg.get("text_config"), dict):
        cfg = cfg["text_config"]
    def num(key, default=None):
        return cfg.get(key, cfg.get(f"num_{key}", default))
    heads = int(num("attention_heads"))
    kv_heads = int(num("key_value_heads", heads))
    if kv_heads > heads:
        raise ValueError(
            f"head_count_kv={kv_heads} > head_count={heads}: more KV heads than query heads is not supported")
    rope_theta = cfg.get("rope_theta")
    if rope_theta is None and isinstance(cfg.get("rope_parameters"), dict):
        rope_theta = cfg["rope_parameters"].get("rope_theta", 10000.0)
    head_dim = cfg.get("head_dim")
    d_model = int(cfg["hidden_size"])
    if head_dim is not None and int(head_dim) != d_model // heads:
        raise ValueError(f"head_dim={head_dim} != hidden_size/heads={d_model // heads}: "
                         "non-uniform head dims are not supported")
    return {
        "vocab_size": int(cfg["vocab_size"]),
        "d_model": d_model,
        "n_heads": heads,
        "n_kv_heads": kv_heads,
        "n_layers": int(num("hidden_layers")),
        "seq_len": min(int(cfg.get("max_position_embeddings", 2048)), 4096),
        "d_ff": int(cfg.get("intermediate_size") or 4 * d_model),
        "tie_weights": bool(cfg.get("tie_word_embeddings", True)),
        "act_quant": True,
        "rope_base": float(rope_theta or 10000.0),
    }


def _hf_tokenizer(path: Path) -> tuple[dict, list[str], str] | None:
    """Read tokenizer.json (HF BPE schema) -> (vocab dict, merges, model_type).

    The vocab is already {token: id}; merges are the same 'a b' strings the
    GGML path embeds. Returns None if tokenizer.json is absent or not BPE."""
    tj = Path(path) / "tokenizer.json"
    if not tj.is_file():
        return None
    data = json.loads(tj.read_text(encoding="utf-8"))
    model = data.get("model") or {}
    if model.get("type", "BPE").upper() != "BPE":
        return None
    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict) or not isinstance(merges, list):
        return None
    # drop merges that reference unknown tokens (added tokens can leak in)
    known = set(vocab)
    merges = [m for m in merges if isinstance(m, str) and " " in m
              and all(tok in known for tok in m.split())]
    # specials: HF flags them via added_tokens[].special. Added tokens often
    # live ONLY there (not in model.vocab) -- merge them into the vocab so the
    # JS tokenizer can look them up (their ids are valid embedding rows).
    special = []
    for at in data.get("added_tokens", []) or []:
        if isinstance(at, dict) and at.get("special") and at.get("content"):
            tok = at["content"]
            if tok not in vocab:
                vocab[tok] = int(at.get("id", len(vocab)))
            special.append(tok)
    if not special:
        # no added_tokens: fall back to vocab entries no merge can produce
        mergeable = {t.split()[0] for t in merges} | {t.split()[1] for t in merges}
        special = [t for t in vocab if t not in mergeable]
    # chat_template: tokenizer.json first, then tokenizer_config.json
    chat_template = data.get("chat_template")
    if not chat_template:
        tjc = Path(path) / "tokenizer_config.json"
        if tjc.is_file():
            chat_template = json.loads(tjc.read_text(encoding="utf-8")).get("chat_template")
    return (vocab, merges, str(data.get("tokenizer_class", "BPE")), special,
            chat_template)


def export_hf(path: Path, config: dict | None = None) -> dict:
    """Export a HuggingFace safetensors checkpoint directory (config.json +
    model.safetensors, optionally sharded via model.safetensors.index.json) to
    the gabion wire format. The wire is identical to the GGUF path: llama-family
    weights as f16, {vocab, merges} when tokenizer.json is present."""
    cfg = _hf_config(path / "config.json")
    if config is not None:
        cfg = dict(config)
    # tie_weights is decided by the checkpoint, not the caller (mirror GGUF path)
    cfg.pop("tie_weights", None)

    # locate the tensor shards
    idx_path = path / "model.safetensors.index.json"
    if idx_path.is_file():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        weight_map = idx["weight_map"]
    else:
        st = path / "model.safetensors"
        if not st.is_file():
            raise ValueError(f"no model.safetensors or model.safetensors.index.json in {path}")
        weight_map = None

    files = {}
    headers = {}

    def shard_of(name: str):
        if weight_map is None:
            return path / "model.safetensors"
        return path / weight_map[name]

    def get(name: str) -> np.ndarray | None:
        if weight_map is not None and name not in weight_map:
            return None
        spath = shard_of(name)
        mm = files.get(spath)
        if mm is None:
            fh = open(spath, "rb")
            (hdr_len,) = struct.unpack_from("<Q", fh.read(8))
            fh.seek(0)
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            files[spath] = mm
            headers[spath] = (hdr_len, json.loads(mm[8:8 + hdr_len]))
        hdr_len, hdr = headers[spath]
        info = hdr.get(name)
        if info is None:
            return None
        return _st_tensor(mm, hdr_len, info)

    def as_linear(name: str, want: tuple) -> np.ndarray:
        # HF linear weights are stored (out, in); the wire wants (in, out).
        t = get(name)
        if t is None:
            raise ValueError(f"checkpoint is missing required tensor '{name}'")
        t = t.T
        if t.shape != want:
            raise ValueError(f"tensor '{name}': shape {t.shape} vs expected {want}")
        return t

    def build_flat() -> np.ndarray:
        """Assemble the f16 wire flat. Inner scope so every mmap-derived view
        becomes unreachable before the mmaps are closed (BufferError otherwise)."""
        D, L, heads, kv_heads, d_ff, n_vocab = (
            cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["n_kv_heads"],
            cfg["d_ff"], cfg["vocab_size"])
        kvD = D // heads * kv_heads

        tok = get("model.embed_tokens.weight")
        if tok is None:
            raise ValueError("checkpoint is missing 'model.embed_tokens.weight'")
        if tok.shape != (n_vocab, D):
            raise ValueError(f"model.embed_tokens.weight: shape {tok.shape} vs expected {(n_vocab, D)}")
        tensors = [tok]
        for i in range(L):
            q = as_linear(f"model.layers.{i}.self_attn.q_proj.weight", (D, D))
            k = as_linear(f"model.layers.{i}.self_attn.k_proj.weight", (D, kvD))
            v = as_linear(f"model.layers.{i}.self_attn.v_proj.weight", (D, kvD))
            o = as_linear(f"model.layers.{i}.self_attn.o_proj.weight", (D, D))
            n1 = get(f"model.layers.{i}.input_layernorm.weight")
            if n1 is None:
                raise ValueError(f"missing model.layers.{i}.input_layernorm.weight")
            gate = as_linear(f"model.layers.{i}.mlp.gate_proj.weight", (D, d_ff))
            up = as_linear(f"model.layers.{i}.mlp.up_proj.weight", (D, d_ff))
            gate_up = np.concatenate([gate, up], axis=1)  # JS splitLast([dFF, dFF]) -> gate first
            n2 = get(f"model.layers.{i}.post_attention_layernorm.weight")
            if n2 is None:
                raise ValueError(f"missing model.layers.{i}.post_attention_layernorm.weight")
            down = as_linear(f"model.layers.{i}.mlp.down_proj.weight", (d_ff, D))
            tensors += [q, k, v, o, n1.reshape(-1), gate_up, n2.reshape(-1), down]
        tensors.append(get("model.norm.weight").reshape(-1))

        if get("lm_head.weight") is None:
            tie = True
            print("[info] checkpoint has no lm_head; using tied embedding")
        else:
            tie = False
            tensors.append(as_linear("lm_head.weight", (D, n_vocab)))
        cfg["tie_weights"] = tie

        # f16 flat directly (no f32 copy): halves peak RAM on big checkpoints
        return np.concatenate([np.asarray(t).astype(np.float16).reshape(-1) for t in tensors])

    flat16 = build_flat()
    wire = {"config": cfg,
            "weights_b64": base64.b64encode(flat16.view("<u2").tobytes()).decode("ascii")}
    # Attention biases (q/k/v per layer) — Qwen2.5-Instruct trains with them.
    # Separate wire fields (not in the flat) so the weight cursor stays untouched.
    qb, kb, vb = [], [], []
    for i in range(cfg["n_layers"]):
        qb.append(np.asarray(get(f"model.layers.{i}.self_attn.q_proj.bias"), dtype=np.float32).tolist()
                  if get(f"model.layers.{i}.self_attn.q_proj.bias") is not None else None)
        kb.append(np.asarray(get(f"model.layers.{i}.self_attn.k_proj.bias"), dtype=np.float32).tolist()
                  if get(f"model.layers.{i}.self_attn.k_proj.bias") is not None else None)
        vb.append(np.asarray(get(f"model.layers.{i}.self_attn.v_proj.bias"), dtype=np.float32).tolist()
                  if get(f"model.layers.{i}.self_attn.v_proj.bias") is not None else None)
    if any(qb) or any(kb) or any(vb):
        wire["q_bias"] = qb
        wire["k_bias"] = kb
        wire["v_bias"] = vb
    cfg["attention_bias"] = bool(any(qb) or any(kb) or any(vb))
    tokdata = _hf_tokenizer(path)
    if tokdata is not None:
        vocab, merges, toktype, special, chat_template = tokdata
        wire["vocab"] = vocab
        wire["merges"] = merges
        if special:
            wire["special"] = special
        if chat_template:
            wire["chat_template"] = chat_template
        cfg["tokenizer"] = f"hf:{toktype}"
    for mm in files.values():
        mm.close()
    return wire


def export_tinygrad(path: Path, config: dict) -> dict:
    if path.suffix == ".npz":
        load = lambda n: _npy(path, n)  # noqa: E731
    elif path.suffix in (".safetensors", ".safetensor"):
        load = lambda n: _safetensors(path, n)  # noqa: E731
    else:
        raise ValueError("tinygrad source must be .npz or .safetensors")
    L = config["n_layers"]
    tensors = [load("tok_emb")]
    for i in range(L):
        for part in ("q", "k", "v", "o", "norm1", "gate_up", "norm2", "down"):
            tensors.append(load(f"layer{i}.{part}"))
    tensors.append(load("norm_f"))
    if not config.get("tie_weights", True):
        tensors.append(load("lm_head"))
    flat = np.concatenate([np.asarray(t, dtype=np.float32).reshape(-1) for t in tensors])
    return {"config": config, "weights_b64": f16_base64(flat)}


def export_gguf(path: Path, config: dict | None = None) -> dict:
    buf = Path(path).read_bytes()
    meta, infos, data_base = parse_gguf(path)
    by_name = {n: (dims, gtype, off) for (n, dims, gtype, off) in infos}

    # Arch-prefixed metadata (llama, qwen2, gemma, ... all share the key layout).
    arch = str(meta.get("general.architecture", "llama"))

    def kv(key: str):
        return meta.get(f"{arch}.{key}") if f"{arch}.{key}" in meta else meta.get(f"llama.{key}")

    L = int(kv("block_count"))
    D = int(kv("embedding_length"))
    heads = int(kv("attention.head_count"))
    kv_heads = int(kv("attention.head_count_kv") or heads)
    if kv_heads > heads:
        raise NotImplementedError(
            f"head_count_kv={kv_heads} > head_count={heads}: more KV heads than query heads is not supported")
    kvD = kv_heads * (D // heads)
    d_ff = int(kv("feed_forward_length") or 4 * D)
    rope_base = float(kv("rope.freq_base") or 10000.0)
    ctx = int(kv("context_length") or 2048)
    tokens = meta.get("tokenizer.ggml.tokens")
    # Vocab size: token_embd's own ne is authoritative (ne = (n_embd, n_vocab)).
    # Metadata can disagree with the actual matrix, and falling back to a bare
    # 32000 invents a size for any model whose vocab is smaller.
    tok_dims = by_name["token_embd.weight"][0] if "token_embd.weight" in by_name else ()
    if len(tok_dims) == 2:
        n_vocab = int(tok_dims[1])
    elif kv("vocab_size"):
        n_vocab = int(kv("vocab_size"))
    elif tokens:
        n_vocab = len(tokens)
    else:
        raise ValueError("cannot determine vocab size: no token_embd.weight, "
                         "no vocab_size metadata, no tokenizer table")

    def get(name: str) -> np.ndarray | None:
        if name not in by_name:
            return None
        dims, gtype, off = by_name[name]
        return _tensor_data(buf, name, dims, gtype, off, data_base)

    def as_linear(name: str, want: tuple) -> np.ndarray:
        """A GGUF linear weight has ne = (in_features, out_features), and _dequant
        reverses ne, so it hands back (out, in). The wire format wants (in, out) for
        the JS `x.matmul(weight)`, i.e. always the transpose. Never infer this from
        the shape: a square matrix matches both ways and would silently pass through
        in the wrong orientation."""
        t = get(name)
        if t is None:
            raise ValueError(f"GGUF is missing required tensor '{name}'")
        t = t.T
        if t.shape != want:
            raise ValueError(f"GGUF tensor '{name}': shape {t.shape} vs expected {want}")
        return t

    tok = get("token_embd.weight")
    # token_embd ne = (n_embd, n_vocab), so _dequant already yields (n_vocab, n_embd)
    if tok.shape != (n_vocab, D):
        raise ValueError(f"token_embd.weight: shape {tok.shape} vs expected {(n_vocab, D)}")
    tensors = [tok]
    for i in range(L):
        q = as_linear(f"blk.{i}.attn_q.weight", (D, D))
        k = as_linear(f"blk.{i}.attn_k.weight", (D, kvD))
        v = as_linear(f"blk.{i}.attn_v.weight", (D, kvD))
        o = as_linear(f"blk.{i}.attn_output.weight", (D, D))
        n1 = get(f"blk.{i}.attn_norm.weight").reshape(-1)
        gate = as_linear(f"blk.{i}.ffn_gate.weight", (D, d_ff))
        up = as_linear(f"blk.{i}.ffn_up.weight", (D, d_ff))
        gate_up = np.concatenate([gate, up], axis=1)  # JS splitLast([dFF, dFF]) -> gate first
        n2 = get(f"blk.{i}.ffn_norm.weight").reshape(-1)
        down = as_linear(f"blk.{i}.ffn_down.weight", (d_ff, D))
        tensors += [q, k, v, o, n1, gate_up, n2, down]
    tensors.append(get("output_norm.weight").reshape(-1))

    if "output.weight" not in by_name:
        tie = True
        print("[info] GGUF has no output.weight; using tied embedding")
    else:
        tie = False
        tensors.append(as_linear("output.weight", (D, n_vocab)))
    if config is None:
        cfg = {
            "vocab_size": n_vocab, "d_model": D, "n_heads": heads,
            "n_kv_heads": kv_heads, "n_layers": L,
            "seq_len": min(ctx, 4096), "d_ff": d_ff, "tie_weights": tie,
            "act_quant": True, "rope_base": rope_base,
        }
    else:
        # Copy: cfg picks up a "tokenizer" key below, and that must not leak back
        # into the caller's dict.
        cfg = dict(config)
        # tie_weights is not a free choice -- it describes whether lm_head is in
        # the flat buffer, which is decided by the GGUF. Disagreeing silently
        # mis-sizes lm_head at load (subarray clamps, leaving it zero-filled).
        if "tie_weights" in cfg and bool(cfg["tie_weights"]) != tie:
            raise ValueError(
                f"config sets tie_weights={cfg['tie_weights']}, but this GGUF "
                f"{'has no' if tie else 'has an'} output.weight tensor, so the "
                f"weight buffer is built {'tied' if tie else 'untied'}")
        cfg["tie_weights"] = tie
    flat = np.concatenate([np.asarray(t, dtype=np.float32).reshape(-1) for t in tensors])
    wire = {"config": cfg, "weights_b64": f16_base64(flat)}
    # Embed the GGUF's own byte-level BPE tokenizer when present (the JS
    # tokenizer is generic over {vocab, merges}).
    if tokens and meta.get("tokenizer.ggml.merges"):
        wire["vocab"] = {t: i for i, t in enumerate(tokens)}
        # No '#'-filter here: tokenizer.ggml.merges is a structured array with no
        # '#version:' header, and byte-level BPE has real merges starting with '#'
        # (GPT-2 has 8, e.g. "# #" -> the "##" token).
        wire["merges"] = [ln for ln in meta["tokenizer.ggml.merges"] if ln]
        # special tokens: GGUF token types 3 (control) and 4 (user-defined) --
        # the JS tokenizer must keep these atomic (chat-template markers etc.)
        ttypes = meta.get("tokenizer.ggml.token_type")
        if ttypes:
            wire["special"] = [t for i, t in enumerate(tokens)
                               if i < len(ttypes) and ttypes[i] in (3, 4)]
        if meta.get("tokenizer.chat_template"):
            wire["chat_template"] = meta["tokenizer.chat_template"]
        cfg["tokenizer"] = f"{arch}:{meta.get('tokenizer.ggml.pre', 'bpe')}"
    return wire


# --------------------------------------------------------------------------
# tokenizer embed (GPT-2 byte-level BPE)
# --------------------------------------------------------------------------

GPT2_URLS = {
    "vocab": "https://huggingface.co/gpt2/resolve/main/vocab.json",
    "merges": "https://huggingface.co/gpt2/resolve/main/merges.txt",
}


def fetch_gpt2_tokenizer() -> tuple[dict, list[str]]:
    with urllib.request.urlopen(GPT2_URLS["vocab"], timeout=30) as r:
        vocab = json.loads(r.read().decode("utf-8"))
    with urllib.request.urlopen(GPT2_URLS["merges"], timeout=30) as r:
        text = r.read().decode("utf-8")
    # anchor to the '#version:' header -- '#' alone also drops real merges
    merges = [ln for ln in text.splitlines() if ln and not ln.startswith("#version")]
    return vocab, merges


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-tinygrad", type=Path, metavar="PATH")
    src.add_argument("--from-gguf", type=Path, metavar="PATH")
    src.add_argument("--from-hf", type=Path, metavar="DIR",
                     help="HuggingFace checkpoint dir (config.json + model.safetensors[.index.json])")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-kv-heads", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--rope-base", type=float, default=10000.0)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--no-tie", action="store_true")
    ap.add_argument("--with-tokenizer", choices=["gpt2"], default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-bin", type=Path, default=None,
                    help="write a binary wire instead: model.json (config/vocab/merges, "
                         "no weights) + weights.f16 (raw little-endian f16 flat) into DIR")
    args = ap.parse_args()

    if args.from_gguf:
        out = export_gguf(args.from_gguf)
    elif args.from_hf:
        out = export_hf(args.from_hf)
    else:
        cfg = {
            "vocab_size": args.vocab_size, "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_kv_heads": args.n_kv_heads or args.n_heads,
            "n_layers": args.n_layers,
            "seq_len": args.seq_len,
            "d_ff": args.d_ff or (args.d_model * 4),
            "tie_weights": not args.no_tie, "act_quant": True,
            "rope_base": args.rope_base,
        }
        out = export_tinygrad(args.from_tinygrad, cfg)

    if args.with_tokenizer:
        try:
            vocab, merges = fetch_gpt2_tokenizer()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] tokenizer download failed ({e}); continuing without it", file=sys.stderr)
        else:
            out["vocab"] = vocab
            out["merges"] = merges
            out["config"]["tokenizer"] = args.with_tokenizer

    n_weights = len(f16_decode(out["weights_b64"]))
    if args.out_bin:
        d = Path(args.out_bin)
        d.mkdir(parents=True, exist_ok=True)
        f16 = base64.b64decode(out["weights_b64"])
        (d / "weights.f16").write_bytes(f16)
        out.pop("weights_b64")
        (d / "model.json").write_text(json.dumps(out), encoding="utf-8")
        print(f"wrote {d / 'model.json'} + {d / 'weights.f16'} ({n_weights} weights, "
              f"{len(out.get('vocab', {})) if out.get('vocab') else 0} vocab tokens)")
        return 0
    out_path = args.out or Path(str((args.from_gguf or args.from_tinygrad or args.from_hf)) + ".gabion.json")
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {out_path} ({n_weights} weights, "
          f"{len(out.get('vocab', {})) if out.get('vocab') else 0} vocab tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
