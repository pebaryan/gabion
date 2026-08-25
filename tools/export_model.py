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
  --from-gguf PATH       llama-family GGUF file. Supports F32/F16/Q8_0/Q4_0/
                         Q4_1 tensors; requires attention.head_count ==
                         attention.head_count_kv (no GQA yet).

  --with-tokenizer gpt2  downloads vocab.json + merges.txt and embeds them.
  --out PATH             output JSON (default: <source stem>.gabion.json)
"""
from __future__ import annotations

import argparse
import base64
import json
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
# GGUF reader (subset: metadata + F32/F16/Q4_0/Q4_1/Q5_0/Q8_0/Q2_K/Q4_K/Q6_K/
# TQ2_0 tensors). Every dequant is byte-exact against gguf-py -- see
# tests/test_export_model.py::test_dequant_matches_gguf_py.
# --------------------------------------------------------------------------

# Current ggml/GGUF type enum (gguf-py GGMLQuantizationType; note the 2025
# renumbering: Q5_0=6, Q8_0=8, Q6_K=14, TQ1_0=34, TQ2_0=35)
GGML_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
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


def _tensor_data(buf: bytes, name: str, dims: tuple, gtype: str, offset: int, base: int = 0) -> np.ndarray:
    if gtype in ("F32", "F16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q4_K", "Q6_K", "Q2_K", "TQ2_0"):
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


def _safetensors(path: Path, name: str) -> np.ndarray:
    import json as _json

    raw = Path(path).read_bytes()
    (hdr_len,) = struct.unpack_from("<Q", raw, 0)
    hdr = _json.loads(raw[8:8 + hdr_len])
    info = hdr.get(name)
    if info is None:
        raise KeyError(f"{name} not in safetensors")
    dt = np.dtype(info["dtype"]).newbyteorder("<")
    arr = np.frombuffer(raw, 8 + hdr_len + info["data_offsets"][0],
                        dtype=dt, count=info["data_offsets"][1] - info["data_offsets"][0])
    return arr.reshape(info["shape"])


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
    # vocab size: explicit key, else the embedded tokenizer table
    n_vocab = int(kv("vocab_size") or 32000)
    tokens = meta.get("tokenizer.ggml.tokens")
    if tokens:
        n_vocab = max(n_vocab, len(tokens))

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
    cfg = config or {
        "vocab_size": n_vocab, "d_model": D, "n_heads": heads,
        "n_kv_heads": kv_heads, "n_layers": L,
        "seq_len": min(ctx, 4096), "d_ff": d_ff, "tie_weights": tie,
        "act_quant": True, "rope_base": rope_base,
    }
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
    args = ap.parse_args()

    if args.from_gguf:
        out = export_gguf(args.from_gguf)
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
    out_path = args.out or Path(str((args.from_gguf or args.from_tinygrad)) + ".gabion.json")
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {out_path} ({n_weights} weights, "
          f"{len(out.get('vocab', {})) if out.get('vocab') else 0} vocab tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
