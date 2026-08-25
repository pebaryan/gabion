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
# GGUF reader (subset: metadata + F32/F16/Q8_0/Q4_0/Q4_1 tensors)
# --------------------------------------------------------------------------

GGML_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q8_0",
    8: "Q8_1", 9: "Q2_K", 10: "Q3_K", 11: "Q4_K", 12: "Q5_K", 13: "Q6_K",
    14: "Q8_K", 15: "IQ2_XXS", 16: "IQ2_XS", 17: "IQ3_XXS",
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


def parse_gguf(path: Path) -> tuple[dict, list[tuple[str, tuple, str, int]]]:
    """Returns (metadata, tensor_infos) where tensor_infos are
    (name, dims_tuple, ggml_type, byte_offset)."""
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
    return meta, infos


def _dequant(name: str, raw: bytes, dims: tuple, gtype: str) -> np.ndarray:
    # GGUF dims are most-significant-first; numpy wants row-major order.
    shape = tuple(dims[::-1])
    n = int(np.prod(shape))
    if gtype == "F32":
        return np.frombuffer(raw, dtype="<f4", count=n).reshape(shape)
    if gtype == "F16":
        return np.frombuffer(raw, dtype="<f2", count=n).astype(np.float32).reshape(shape)
    if gtype == "Q8_0":
        # block of 32: f16 scale + 32 x i8
        nblocks = n // 32
        vals = np.empty(n, dtype=np.float32)
        for b in range(nblocks):
            (d,) = struct.unpack_from("<e", raw, b * 34)
            xs = struct.unpack_from(f"<{32}b", raw, b * 34 + 2)
            vals[b * 32:(b + 1) * 32] = np.asarray(xs, dtype=np.float32) * float(d)
        return vals.reshape(shape)
    if gtype == "Q4_0":
        # block of 16: f16 scale + 16 x 4-bit signed (offset 8)
        nblocks = n // 16
        vals = np.empty(n, dtype=np.float32)
        for b in range(nblocks):
            (d,) = struct.unpack_from("<e", raw, b * 18)
            packed = raw[b * 18 + 2:b * 18 + 18]
            nib = np.frombuffer(packed, dtype=np.uint8).astype(np.int16)
            xs = np.empty(16, dtype=np.float32)
            for i in range(16):
                v = (nib[i // 2] >> (4 * (i % 2))) & 0x0F
                xs[i] = float(v - 8) * float(d)
            vals[b * 16:(b + 1) * 16] = xs
        return vals.reshape(shape)
    if gtype == "Q4_1":
        # block of 16: f16 d + f16 m + 16 x 4-bit (value = x*d + m)
        nblocks = n // 16
        vals = np.empty(n, dtype=np.float32)
        for b in range(nblocks):
            (d, m) = struct.unpack_from("<ee", raw, b * 20)
            packed = raw[b * 20 + 4:b * 20 + 20]
            nib = np.frombuffer(packed, dtype=np.uint8)
            xs = np.empty(16, dtype=np.float32)
            for i in range(16):
                v = (nib[i // 2] >> (4 * (i % 2))) & 0x0F
                xs[i] = float(v) * float(d) + float(m)
            vals[b * 16:(b + 1) * 16] = xs
        return vals.reshape(shape)
    raise NotImplementedError(f"GGUF tensor '{name}': unsupported type {gtype}")


def _tensor_data(buf: bytes, name: str, dims: tuple, gtype: str, offset: int) -> np.ndarray:
    if gtype in ("F32", "F16", "Q8_0", "Q4_0", "Q4_1"):
        return _dequant(name, buf[offset:], dims, gtype)
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
    meta, infos = parse_gguf(path)
    by_name = {n: (dims, gtype, off) for (n, dims, gtype, off) in infos}

    L = int(meta["llama.block_count"])
    D = int(meta["llama.embedding_length"])
    heads = int(meta["llama.attention.head_count"])
    kv_heads = int(meta.get("llama.attention.head_count_kv", heads))
    if kv_heads > heads:
        raise NotImplementedError(
            f"head_count_kv={kv_heads} > head_count={heads}: more KV heads than query heads is not supported")
    kvD = kv_heads * (D // heads)
    d_ff = int(meta.get("llama.feed_forward_length", 4 * D))
    rope_base = float(meta.get("llama.rope.freq_base", 10000.0))
    ctx = int(meta.get("llama.context_length", 2048))
    n_vocab = int(meta.get("llama.vocab_size", 32000))

    def get(name: str) -> np.ndarray | None:
        if name not in by_name:
            return None
        dims, gtype, off = by_name[name]
        return _tensor_data(buf, name, dims, gtype, off)

    def as_2d(t: np.ndarray, want: tuple) -> np.ndarray:
        if t.shape == want:
            return t
        if t.shape == want[::-1]:
            return t.T
        raise ValueError(f"shape {t.shape} vs expected {want}")

    tok = get("token_embd.weight")
    tok = as_2d(tok, (n_vocab, D))
    tensors = [tok]
    for i in range(L):
        q = as_2d(get(f"blk.{i}.attn_q.weight"), (D, D))
        k = as_2d(get(f"blk.{i}.attn_k.weight"), (D, kvD))
        v = as_2d(get(f"blk.{i}.attn_v.weight"), (D, kvD))
        o = as_2d(get(f"blk.{i}.attn_output.weight"), (D, D))
        n1 = get(f"blk.{i}.attn_norm.weight").reshape(-1)
        gate = as_2d(get(f"blk.{i}.ffn_gate.weight"), (D, d_ff))
        up = as_2d(get(f"blk.{i}.ffn_up.weight"), (D, d_ff))
        gate_up = np.concatenate([gate, up], axis=1)  # JS splitLast([dFF, dFF]) -> gate first
        n2 = get(f"blk.{i}.ffn_norm.weight").reshape(-1)
        down = as_2d(get(f"blk.{i}.ffn_down.weight"), (d_ff, D))
        tensors += [q, k, v, o, n1, gate_up, n2, down]
    tensors.append(get("output_norm.weight").reshape(-1))

    tie = get("output.weight") is None
    if not tie:
        print("[info] GGUF has output.weight; using tied embedding anyway (set tie_weights=true)")
    cfg = config or {
        "vocab_size": n_vocab, "d_model": D, "n_heads": heads,
        "n_kv_heads": kv_heads, "n_layers": L,
        "seq_len": min(ctx, 4096), "d_ff": d_ff, "tie_weights": True,
        "act_quant": True, "rope_base": rope_base,
    }
    flat = np.concatenate([np.asarray(t, dtype=np.float32).reshape(-1) for t in tensors])
    return {"config": cfg, "weights_b64": f16_base64(flat)}


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
    merges = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
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
