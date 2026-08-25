"""Round-trip and GGUF tests for tools/export_model.py (browser wire format)."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.export_model import f16_base64, f16_decode, export_gguf, export_tinygrad

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "gabion")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def test_f16_codec_roundtrip():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, size=1000).astype(np.float32)
    x[0] = 0.0
    x[1] = -3.5
    x[2] = 65504.0  # f16 max
    back = f16_decode(f16_base64(x))
    # f16 rounding: relative error <= 2^-11 for normalized values
    assert np.max(np.abs(back - x)) <= 0.5 * np.abs(x).max() * 2**-10 + 1e-6
    assert back[0] == 0.0 and back[1] == -3.5


def test_export_tinygrad_roundtrip(tmp_path):
    # Build a tiny adapter, flatten via the gabion adapter contract, export, re-decode.
    sys_path = [str(ROOT / "gabion"), str(ROOT)]
    import sys

    for p in sys_path:
        if p not in sys.path:
            sys.path.insert(0, p)
    from gabion.pebble.adapters import flatten_tensors, unflatten_to_tensors
    from gabion.user_models.bbt_transformer import BBTTransformerAdapter
    from tinygrad import Tensor

    cfg = dict(input_dim=17, d_model=8, n_heads=2, n_layers=1, seq_len=5, act_quant=False)
    adapter = BBTTransformerAdapter(**cfg)
    params = adapter.init_params(seed=3)
    flat = np.asarray(flatten_tensors(params), dtype=np.float32)

    # Save as an npz the exporter understands
    names = ["tok_emb"]
    for i in range(cfg["n_layers"]):
        names += [f"layer{i}.{p}" for p in ("q", "k", "v", "o", "norm1", "gate_up", "norm2", "down")]
    names.append("norm_f")
    npz_path = tmp_path / "model.npz"
    arrays = {}
    template = adapter.init_params(seed=3)
    cursor = 0
    for name, tp in zip(names, template):
        n = tp.numel()
        arr = flat[cursor:cursor + n].reshape(tp.shape)
        arrays[name] = arr
        cursor += n
    assert cursor == len(flat)
    np.savez(npz_path, **arrays)

    model_cfg = dict(
        vocab_size=17, d_model=8, n_heads=2, n_layers=1, seq_len=5,
        d_ff=32, tie_weights=True, act_quant=False, rope_base=10000.0,
    )
    out = export_tinygrad(npz_path, model_cfg)
    flat_back = f16_decode(out["weights_b64"])
    assert len(flat_back) == len(flat)
    # f16 tolerance
    err = np.max(np.abs(flat_back - flat))
    assert err < 1e-2, f"max err {err}"

    # JS-side consumption contract: f16Base64ToWeights -> loadFlatWeights order preserved
    js_order = ["tok_emb"]
    for i in range(cfg["n_layers"]):
        js_order += [f"layer{i}.{p}" for p in ("q", "k", "v", "o", "norm1", "gate_up", "norm2", "down")]
    js_order.append("norm_f")
    assert names == js_order


def _write_gguf(path: Path, meta: dict, tensors: dict[str, np.ndarray],
                align: int = 32) -> tuple[int, int]:
    """Write a minimal GGUF v3 file with F32/F16/Q8_0 tensors.

    `align` is the data-section alignment actually used for the layout; pass a
    matching "general.alignment" in meta to exercise the non-default path.
    """
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(meta))

    def put_str(s: str):
        nonlocal out
        b = s.encode("utf-8")
        out += struct.pack("<Q", len(b)) + b

    def put_val(v):
        nonlocal out
        if isinstance(v, str):
            out += struct.pack("<I", 8)
            put_str(v)
        elif isinstance(v, bool):
            out += struct.pack("<I", 7) + struct.pack("<?", v)
        elif isinstance(v, int):
            out += struct.pack("<I", 11) + struct.pack("<q", v)
        elif isinstance(v, float):
            out += struct.pack("<I", 6) + struct.pack("<f", v)
        elif isinstance(v, list):  # array: u32 elem_type, u64 count, bare elements
            assert all(isinstance(e, str) for e in v), "only string arrays supported"
            out += struct.pack("<I", 9) + struct.pack("<IQ", 8, len(v))
            for e in v:
                put_str(e)
        else:
            raise TypeError(type(v))

    for k, v in meta.items():
        put_str(k)
        put_val(v)

    # tensor infos: name, n_dims, dims, ggml_type, offset (offsets fixed up below).
    # Arrays are handed in torch orientation and written as ne = shape[::-1], which
    # is exactly what gguf-py does -- so a linear weight (out, in) lands as
    # ne = (in, out) and the fixtures exercise the real export path.
    infos = []
    for name, arr in tensors.items():
        put_str(name)
        out += struct.pack("<I", arr.ndim)
        for d in arr.shape[::-1]:
            out += struct.pack("<Q", d)
        gtype = 0 if arr.dtype == np.float32 else 1
        out += struct.pack("<I", gtype)
        out += struct.pack("<Q", 0)  # placeholder offset

    # data (F32/F16 for the hand-written file); GGUF offsets are relative to the
    # 32-byte-aligned start of the data section (spec)
    info_end = len(out)          # where the tensor-info section stops
    data_base = (info_end + align - 1) & ~(align - 1)
    out += b"\x00" * (data_base - len(out))
    offsets = {}
    for name, arr in tensors.items():
        offsets[name] = len(out) - data_base
        out += np.asarray(arr, dtype="<f4" if arr.dtype == np.float32 else "<f2").tobytes()

    # fix up offsets in the info section
    buf = bytes(out)
    out2 = bytearray(buf[:26])  # header + counts
    pos = 26
    for k, v in meta.items():
        pos += 8 + len(k.encode())
        # skip value (type + payload) — recompute precisely
        # (simpler: rebuild with correct offsets below)
        break
    # Rebuild cleanly: we know all meta is strings/ints/floats/bools; redo with offsets known
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(meta))
    for k, v in meta.items():
        put_str(k)
        put_val(v)
    for name, arr in tensors.items():
        put_str(name)
        out += struct.pack("<I", arr.ndim)
        for d in arr.shape[::-1]:
            out += struct.pack("<Q", d)
        gtype = 0 if arr.dtype == np.float32 else 1
        out += struct.pack("<I", gtype)
        out += struct.pack("<Q", offsets[name])
    out += b"\x00" * (data_base - len(out))
    for name, arr in tensors.items():
        out += np.asarray(arr, dtype="<f4" if arr.dtype == np.float32 else "<f2").tobytes()
    Path(path).write_bytes(bytes(out))
    return info_end, data_base


def test_gguf_parse_and_export(tmp_path):
    meta = {
        "general.architecture": "llama",
        "llama.block_count": 1,
        "llama.embedding_length": 8,
        "llama.attention.head_count": 2,
        "llama.attention.head_count_kv": 2,
        "llama.feed_forward_length": 32,
        "llama.rope.freq_base": 10000.0,
        "llama.context_length": 64,
        "llama.vocab_size": 17,
    }
    rng = np.random.default_rng(11)
    D, dFF, V = 8, 32, 17
    tensors = {
        "token_embd.weight": rng.normal(0, 0.1, (V, D)).astype(np.float32),
        "blk.0.attn_norm.weight": rng.normal(0, 0.1, (D,)).astype(np.float32),
        # linear weights in torch orientation (out_features, in_features)
        "blk.0.attn_q.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.attn_k.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.attn_v.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.attn_output.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.ffn_gate.weight": rng.normal(0, 0.1, (dFF, D)).astype(np.float32),
        "blk.0.ffn_up.weight": rng.normal(0, 0.1, (dFF, D)).astype(np.float32),
        "blk.0.ffn_down.weight": rng.normal(0, 0.1, (D, dFF)).astype(np.float32),
        "blk.0.ffn_norm.weight": rng.normal(0, 0.1, (D,)).astype(np.float32),
        "output_norm.weight": rng.normal(0, 0.1, (D,)).astype(np.float32),
    }
    gguf = tmp_path / "tiny.gguf"
    _write_gguf(gguf, meta, tensors)

    out = export_gguf(gguf)
    cfg = out["config"]
    assert cfg["d_model"] == 8 and cfg["n_heads"] == 2 and cfg["n_layers"] == 1
    assert cfg["d_ff"] == 32 and cfg["tie_weights"] is True
    flat = f16_decode(out["weights_b64"])

    # expected layout: tok_emb, [q,k,v,o,n1,gate_up,n2,down], norm_f
    n_tok = V * D
    per_layer = 4 * D * D + 2 * D + 2 * D * dFF + dFF * D
    assert len(flat) == n_tok + per_layer + D, len(flat)
    tok = flat[:n_tok].reshape(V, D)
    assert np.max(np.abs(tok - tensors["token_embd.weight"])) < 1e-2
    # gate_up = concat(gate, up) along last dim (gate first, matching JS splitLast)
    gu = flat[n_tok + 4 * D * D + D: n_tok + 4 * D * D + D + 2 * D * dFF].reshape(D, 2 * dFF)
    assert np.max(np.abs(gu[:, :dFF] - tensors["blk.0.ffn_gate.weight"].T)) < 1e-2
    assert np.max(np.abs(gu[:, dFF:] - tensors["blk.0.ffn_up.weight"].T)) < 1e-2
    # square projections must be transposed too -- the case a shape check can't see
    q = flat[n_tok:n_tok + D * D].reshape(D, D)
    assert np.max(np.abs(q - tensors["blk.0.attn_q.weight"].T)) < 1e-2
    o = flat[n_tok + 3 * D * D:n_tok + 4 * D * D].reshape(D, D)
    assert np.max(np.abs(o - tensors["blk.0.attn_output.weight"].T)) < 1e-2


def test_gguf_gqa_export(tmp_path):
    """GQA GGUF (4 query heads, 1 KV head) exports with grouped k/v tensors."""
    meta = {
        "general.architecture": "llama",
        "llama.block_count": 1,
        "llama.embedding_length": 8,
        "llama.attention.head_count": 4,
        "llama.attention.head_count_kv": 1,  # GQA
        "llama.feed_forward_length": 32,
        "llama.context_length": 64,
        "llama.vocab_size": 17,
    }
    rng = np.random.default_rng(1)
    D, kvD = 8, 2  # head_dim = 2, kv_heads=1
    tensors = {
        "token_embd.weight": rng.normal(0, 0.1, (17, D)).astype(np.float32),
        # linear weights in torch orientation (out_features, in_features)
        "blk.0.attn_q.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.attn_k.weight": rng.normal(0, 0.1, (kvD, D)).astype(np.float32),
        "blk.0.attn_v.weight": rng.normal(0, 0.1, (kvD, D)).astype(np.float32),
        "blk.0.attn_output.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.attn_norm.weight": rng.normal(0, 0.1, (D,)).astype(np.float32),
        "blk.0.ffn_gate.weight": rng.normal(0, 0.1, (32, D)).astype(np.float32),
        "blk.0.ffn_up.weight": rng.normal(0, 0.1, (32, D)).astype(np.float32),
        "blk.0.ffn_norm.weight": rng.normal(0, 0.1, (D,)).astype(np.float32),
        "blk.0.ffn_down.weight": rng.normal(0, 0.1, (D, 32)).astype(np.float32),
        "output_norm.weight": rng.normal(0, 0.1, (D,)).astype(np.float32),
    }
    gguf = tmp_path / "gqa.gguf"
    _write_gguf(gguf, meta, tensors)
    out = export_gguf(gguf)
    assert out["config"]["n_heads"] == 4
    assert out["config"]["n_kv_heads"] == 1
    flat = f16_decode(out["weights_b64"])
    n_tok = 17 * D
    # q,o [D,D]; k,v [D,kvD]; norms 2*D; gate_up 2*D*32; down 32*D
    per_layer = 2 * D * D + 2 * D * kvD + 2 * D + 2 * D * 32 + 32 * D
    assert len(flat) == n_tok + per_layer + D
    k = flat[n_tok + D * D:n_tok + D * D + D * kvD].reshape(D, kvD)
    assert np.allclose(k, tensors["blk.0.attn_k.weight"].T, atol=1e-2)
    # gate_up holds gate then up (after q,k,v,o,n1)
    gu_off = n_tok + 2 * D * D + 2 * D * kvD + D
    gu = flat[gu_off:gu_off + 2 * D * 32].reshape(D, 2 * 32)
    assert np.allclose(gu[:, :32], tensors["blk.0.ffn_gate.weight"].T, atol=1e-2)
    assert np.allclose(gu[:, 32:], tensors["blk.0.ffn_up.weight"].T, atol=1e-2)


def test_export_tinygrad_gqa(tmp_path):
    """tinygrad npz with grouped KV heads exports to the expected kvD layout."""
    from gabion.user_models.bbt_transformer import BBTTransformerAdapter

    cfg = {"vocab_size": 32, "d_model": 16, "n_heads": 4, "n_kv_heads": 2,
           "n_layers": 1, "seq_len": 8, "d_ff": 32, "tie_weights": True,
           "act_quant": True}
    adapter = BBTTransformerAdapter(input_dim=cfg["vocab_size"], d_model=cfg["d_model"],
                                    n_heads=cfg["n_heads"], n_kv_heads=cfg["n_kv_heads"],
                                    n_layers=cfg["n_layers"], seq_len=cfg["seq_len"],
                                    d_ff=cfg["d_ff"], act_quant=cfg["act_quant"],
                                    tie_weights=cfg["tie_weights"], use_wikitext=False)
    params = adapter.init_params(seed=5)
    from gabion.pebble.adapters import flatten_tensors

    names = ["tok_emb"] + [f"layer0.{p}" for p in ("q", "k", "v", "o", "norm1", "gate_up", "norm2", "down")] + ["norm_f"]
    np.savez(tmp_path / "m.npz", **dict(zip(names, [np.asarray(p.numpy()) for p in params])))
    out = export_tinygrad(tmp_path / "m.npz", cfg)
    flat = f16_decode(out["weights_b64"])
    D, kvD = 16, 8
    assert out["config"]["n_kv_heads"] == 2
    n_tok = 32 * D
    k = flat[n_tok + D * D:n_tok + D * D + D * kvD].reshape(D, kvD)
    assert np.allclose(k, np.asarray(params[2].numpy()), atol=1e-2)  # layer0.k
    v = flat[n_tok + D * D + D * kvD:n_tok + D * D + 2 * D * kvD].reshape(D, kvD)
    assert np.allclose(v, np.asarray(params[3].numpy()), atol=1e-2)  # layer0.v


def test_gguf_k_quants(tmp_path):
    """Q2_K / Q4_K / Q6_K / TQ2_0 dequant on hand-crafted single super-blocks."""
    from tools.export_model import _dequant

    np16 = np.float16

    # Q2_K: scales[16], qs[64], d, dmin (file layout); sc=2, min=1, qs=0x66
    #   qb = (0x66 >> shift) & 3 = [2,1,2,1] per 32-block; val = 0.5*2*qb - 0.25
    q2 = bytes([0x12]) * 16 + bytes([0x66]) * 64 + struct.pack("<ee", np16(0.5), np16(0.25))
    got = _dequant("t", q2, (256,), "Q2_K")
    qb = np.array([2] * 32 + [1] * 32 + [2] * 32 + [1] * 32, dtype=np.float32)
    exp = np.tile(qb - 0.25, 2)
    assert np.allclose(got, exp, atol=1e-6), np.max(np.abs(got - exp))

    # Q4_K: d, dmin first (file layout), scales all -> sc=10, m=10, qs=0x66
    #   val = 1.0*10*6 - 0.5*10 = 55
    q4 = struct.pack("<ee", np16(1.0), np16(0.5)) + bytes([10, 10, 10, 10, 10, 10, 10, 10, 0xAA, 0xAA, 0xAA, 0xAA]) + bytes([0x66]) * 128
    got = _dequant("t", q4, (256,), "Q4_K")
    assert np.allclose(got, 55.0, atol=1e-5), np.max(np.abs(got - 55.0))

    # Q6_K: ql[128], qh[64], sc[16], d[2] (file layout); d=2.0, sc=1, ql=0xFF, qh=0
    #   q=-17 -> val = 2*1*-17 = -34
    q6 = bytes([0xFF]) * 128 + bytes([0x00]) * 64 + bytes([0x01]) * 16 + struct.pack("<e", np16(2.0))
    got = _dequant("t", q6, (256,), "Q6_K")
    assert np.allclose(got, -34.0, atol=1e-5)

    # TQ2_0: qs[64], d[2] (file layout); qs=0xFF -> bits=3 -> val = (3-1)*0.5 = 1.0
    tq = bytes([0xFF]) * 64 + struct.pack("<e", np16(0.5))
    got = _dequant("t", tq, (256,), "TQ2_0")
    assert np.allclose(got, 1.0, atol=1e-6)

    # Also round-trip through a real GGUF file layout (data-section-relative offsets)
    tensors = {"t1": _dequant("t", q2, (256,), "Q2_K")}
    _write_gguf(tmp_path / "k.gguf", {"general.architecture": "llama"}, tensors)  # F32 only
    from tools.export_model import parse_gguf, _tensor_data

    buf = (tmp_path / "k.gguf").read_bytes()
    meta, infos, base = parse_gguf(tmp_path / "k.gguf")
    name, dims, gtype, off = infos[0]
    assert gtype == "F32"
    assert _tensor_data(buf, name, dims, gtype, off, base).shape == (256,)


def test_gguf_honours_general_alignment(tmp_path):
    """A file declaring general.alignment=64 must be read at 64-byte offsets.

    Whether a 32-vs-64 mistake actually shifts the data section depends on where
    the tensor-info section happens to end, so sweep a padding key across the
    residues: a reader hardcoding 32 gets the wrong base for some of them.
    """
    from tools.export_model import parse_gguf

    D, V = 8, 4
    rng = np.random.default_rng(5)
    emb = rng.normal(0, 0.1, (V, D)).astype(np.float32)
    mismatched_under_32 = 0
    for pad in range(0, 64, 4):
        meta = {
            "general.architecture": "llama", "general.alignment": 64,
            "general.pad": "x" * pad,
            "llama.block_count": 0, "llama.embedding_length": D,
            "llama.attention.head_count": 2, "llama.attention.head_count_kv": 2,
            "llama.feed_forward_length": 16, "llama.context_length": 32,
            "llama.vocab_size": V,
        }
        tensors = {"token_embd.weight": emb,
                   "output_norm.weight": np.ones(D, np.float32)}
        gguf = tmp_path / f"align64_{pad}.gguf"
        info_end, written_base = _write_gguf(gguf, meta, tensors, align=64)

        _meta, infos, base = parse_gguf(gguf)
        assert base == written_base, f"pad={pad}: base {base} != written {written_base}"
        assert base % 64 == 0, base
        # would a reader hardcoding 32 have landed somewhere else?
        if ((info_end + 31) & ~31) != written_base:
            mismatched_under_32 += 1

        out = export_gguf(gguf)
        got = f16_decode(out["weights_b64"])[:V * D].reshape(V, D)
        assert np.max(np.abs(got - emb)) < 1e-2, f"pad={pad}"

    assert mismatched_under_32 > 0, "sweep never produced a 32-vs-64 divergence"


def test_dequant_matches_gguf_py():
    """Every dequant, byte-exact against gguf-py on RANDOM data.

    The hand-written fixtures above use uniform fills, so any within-block
    element permutation passes them. Random bytes pin the ordering down -- this
    is what catches a wrong nibble split (Q4_0/Q4_1) or bit-plane order (TQ2_0).
    """
    gguf = pytest.importorskip("gguf")
    from gguf.constants import GGMLQuantizationType as QT
    from tools.export_model import _dequant

    rng = np.random.default_rng(7)

    # types gguf-py can round-trip from floats
    for name, qt in [("Q4_0", QT.Q4_0), ("Q4_1", QT.Q4_1),
                     ("Q5_0", QT.Q5_0), ("Q8_0", QT.Q8_0)]:
        x = rng.standard_normal(1024).astype(np.float32) * 0.5
        raw = gguf.quants.quantize(x, qt).tobytes()
        ref = gguf.quants.dequantize(np.frombuffer(raw, np.uint8), qt).astype(np.float32)
        got = _dequant(name, raw, (1024,), name)
        assert np.array_equal(got, ref.reshape(-1)), f"{name}: max {np.abs(got - ref).max()}"

    # k-quants: gguf-py only dequantizes these, so drive them from random blocks
    for name, qt, block_bytes in [("Q2_K", QT.Q2_K, 84), ("Q4_K", QT.Q4_K, 144),
                                  ("Q6_K", QT.Q6_K, 210), ("TQ2_0", QT.TQ2_0, 66)]:
        nb = 4
        raw = rng.integers(0, 256, nb * block_bytes, dtype=np.uint8)
        raw[1::2] &= 0x3B  # tame every f16 slot's exponent: no NaN/Inf in the ref
        raw = raw.tobytes()
        ref = gguf.quants.dequantize(np.frombuffer(raw, np.uint8), qt).astype(np.float32)
        got = _dequant(name, raw, (nb * 256,), name)
        assert np.array_equal(got, ref.reshape(-1)), f"{name}: max {np.abs(got - ref).max()}"

    # ---- remaining quant family (round 9284259+: Q5_1/BF16/MXFP4 have quantizers) ----
    for name, qt in [("Q5_1", QT.Q5_1), ("BF16", QT.BF16), ("MXFP4", QT.MXFP4)]:
        x = rng.standard_normal(1024).astype(np.float32) * 0.5
        raw = gguf.quants.quantize(x, qt).tobytes()
        ref = gguf.quants.dequantize(np.frombuffer(raw, np.uint8), qt).astype(np.float32)
        got = _dequant(name, raw, (1024,), name)
        assert np.array_equal(got, ref.reshape(-1)), f"{name}: max {np.abs(got - ref).max()}"

    # ---- the rest: dequant-only types from random blocks (f16 slots tamed) ----
    # (block_bytes, block_elems) per type; NVFP4's ue4m3 scales cannot produce
    # NaN so it needs no taming but the odd-byte mask is harmless there.
    for name, qt, block_bytes, block_elems in [
        ("Q3_K", QT.Q3_K, 110, 256), ("Q5_K", QT.Q5_K, 176, 256),
        ("TQ1_0", QT.TQ1_0, 54, 256), ("IQ2_XXS", QT.IQ2_XXS, 66, 256),
        ("IQ2_XS", QT.IQ2_XS, 74, 256), ("IQ2_S", QT.IQ2_S, 82, 256),
        ("IQ3_XXS", QT.IQ3_XXS, 98, 256), ("IQ3_S", QT.IQ3_S, 110, 256),
        ("IQ1_S", QT.IQ1_S, 50, 256), ("IQ1_M", QT.IQ1_M, 56, 256),
        ("IQ4_NL", QT.IQ4_NL, 18, 32), ("IQ4_XS", QT.IQ4_XS, 136, 256),
        ("NVFP4", QT.NVFP4, 36, 64),
    ]:
        nb = 4
        raw = rng.integers(0, 256, nb * block_bytes, dtype=np.uint8)
        raw[1::2] &= 0x3B  # tame f16 exponent slots -> no NaN/Inf in either side
        raw = raw.tobytes()
        ref = gguf.quants.dequantize(np.frombuffer(raw, np.uint8), qt).astype(np.float32)
        got = _dequant(name, raw, (nb * block_elems,), name)
        assert np.array_equal(got, ref.reshape(-1)), f"{name}: max {np.abs(got - ref).max()}"


def test_gguf_merges_keep_hash_prefixed_entries(tmp_path):
    """Byte-level BPE has real merges starting with '#'; they must survive export."""
    D, V = 8, 4
    merges = ["Ġ t", "# #", "## #", "#$ #$"]
    meta = {
        "general.architecture": "llama", "llama.block_count": 0,
        "llama.embedding_length": D, "llama.attention.head_count": 2,
        "llama.attention.head_count_kv": 2, "llama.feed_forward_length": 16,
        "llama.context_length": 32, "llama.vocab_size": V,
        "tokenizer.ggml.tokens": ["a", "b", "#", "##"],
        "tokenizer.ggml.merges": merges,
    }
    tensors = {
        "token_embd.weight": np.zeros((V, D), np.float32),
        "output_norm.weight": np.ones(D, np.float32),
    }
    gguf = tmp_path / "tok.gguf"
    _write_gguf(gguf, meta, tensors)
    out = export_gguf(gguf)
    assert out["merges"] == merges
    assert out["vocab"] == {"a": 0, "b": 1, "#": 2, "##": 3}


def _minimal_gguf(tmp_path, name, V=17, D=8, dFF=16, untied=False,
                  vocab_size_meta=True):
    """One-layer llama GGUF in torch orientation; optionally with output.weight."""
    rng = np.random.default_rng(2)
    meta = {
        "general.architecture": "llama", "llama.block_count": 1,
        "llama.embedding_length": D, "llama.attention.head_count": 2,
        "llama.attention.head_count_kv": 2, "llama.feed_forward_length": dFF,
        "llama.context_length": 32,
    }
    if vocab_size_meta is not False:   # True -> V, or an explicit override value
        meta["llama.vocab_size"] = V if vocab_size_meta is True else vocab_size_meta
    t = {
        "token_embd.weight": rng.normal(0, 0.1, (V, D)).astype(np.float32),
        "blk.0.attn_norm.weight": np.ones(D, np.float32),
        "blk.0.attn_q.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.attn_k.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.attn_v.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.attn_output.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.ffn_norm.weight": np.ones(D, np.float32),
        "blk.0.ffn_gate.weight": rng.normal(0, 0.1, (dFF, D)).astype(np.float32),
        "blk.0.ffn_up.weight": rng.normal(0, 0.1, (dFF, D)).astype(np.float32),
        "blk.0.ffn_down.weight": rng.normal(0, 0.1, (D, dFF)).astype(np.float32),
        "output_norm.weight": np.ones(D, np.float32),
    }
    if untied:
        t["output.weight"] = rng.normal(0, 0.1, (V, D)).astype(np.float32)
    path = tmp_path / name
    _write_gguf(path, meta, t)
    return path


def test_vocab_size_comes_from_token_embd(tmp_path):
    """Vocab size follows the embedding matrix, not a metadata guess."""
    # no vocab_size key and no tokenizer table: the old code defaulted to 32000
    gguf = _minimal_gguf(tmp_path, "novocab.gguf", V=17, vocab_size_meta=False)
    assert export_gguf(gguf)["config"]["vocab_size"] == 17

    # metadata that disagrees with the actual matrix loses to the matrix
    gguf2 = _minimal_gguf(tmp_path, "wrongvocab.gguf", V=17, vocab_size_meta=99999)
    assert export_gguf(gguf2)["config"]["vocab_size"] == 17


def test_config_tie_weights_must_match_tensors(tmp_path):
    """A caller config cannot claim a tie_weights the weight buffer contradicts."""
    untied = _minimal_gguf(tmp_path, "untied.gguf", untied=True)
    tied = _minimal_gguf(tmp_path, "tied.gguf", untied=False)

    base = {"vocab_size": 17, "d_model": 8, "n_heads": 2, "n_kv_heads": 2,
            "n_layers": 1, "seq_len": 32, "d_ff": 16, "act_quant": True}

    with pytest.raises(ValueError, match="tie_weights"):
        export_gguf(untied, {**base, "tie_weights": True})
    with pytest.raises(ValueError, match="tie_weights"):
        export_gguf(tied, {**base, "tie_weights": False})

    # agreeing values are accepted, and the flat buffer length matches
    out = export_gguf(untied, {**base, "tie_weights": False})
    assert out["config"]["tie_weights"] is False
    D, dFF, V = 8, 16, 17
    per = 4 * D * D + 2 * D + 2 * D * dFF + dFF * D
    assert len(f16_decode(out["weights_b64"])) == V * D + per + D + D * V


def test_export_gguf_does_not_mutate_caller_config(tmp_path):
    """cfg picks up a 'tokenizer' key; that must not leak into the caller's dict."""
    meta_extra = {"tokenizer.ggml.tokens": ["a", "b"],
                  "tokenizer.ggml.merges": ["a b"]}
    rng = np.random.default_rng(4)
    D, dFF, V = 8, 16, 2
    meta = {"general.architecture": "llama", "llama.block_count": 0,
            "llama.embedding_length": D, "llama.attention.head_count": 2,
            "llama.attention.head_count_kv": 2, "llama.feed_forward_length": dFF,
            "llama.context_length": 32, **meta_extra}
    t = {"token_embd.weight": rng.normal(0, 0.1, (V, D)).astype(np.float32),
         "output_norm.weight": np.ones(D, np.float32)}
    gguf = tmp_path / "tokmut.gguf"
    _write_gguf(gguf, meta, t)

    cfg = {"vocab_size": V, "d_model": D, "n_heads": 2, "n_kv_heads": 2,
           "n_layers": 0, "seq_len": 32, "d_ff": dFF, "act_quant": True}
    before = dict(cfg)
    out = export_gguf(gguf, cfg)
    assert cfg == before, f"caller config was mutated: {cfg}"
    assert "tokenizer" in out["config"]        # the copy did get it


def test_export_cli(tmp_path, monkeypatch):
    import tools.export_model as em
    from tools.export_model import main as main_fn

    def fake_export(path, config=None):
        return {"config": {"d_model": 8}, "weights_b64": f16_base64(np.zeros(4, dtype=np.float32))}

    monkeypatch.setattr(em, "export_gguf", fake_export)
    out = tmp_path / "m.json"
    monkeypatch.setattr(
        "sys.argv",
        ["export_model.py", "--from-gguf", "x.gguf", "--out", str(out)],
    )
    main_fn()
    assert out.exists()
    data = json.loads(out.read_text())
    assert "weights_b64" in data and data["config"]["d_model"] == 8


def test_export_gguf_lfm2(tmp_path):
    """Synthetic lfm2 GGUF (hybrid attn/conv layers) -> wire: config, layout,
    flat sizes, tied head, tokenizer. Mirrors the real LFM2.5-2.6B pattern."""
    import numpy as np
    from tools.export_model import f16_decode
    D, H, KVH, HD, DFF, L, V = 32, 4, 2, 8, 64, 6, 16
    rng = np.random.default_rng(7)
    lt = ["conv", "conv", "attn", "conv", "conv", "attn"]
    tensors = {
        "token_embd.weight": rng.standard_normal((V, D)).astype(np.float32),
        "token_embd_norm.weight": rng.standard_normal(D).astype(np.float32),
    }
    for i in range(L):
        tensors[f"blk.{i}.attn_norm.weight"] = rng.standard_normal(D).astype(np.float32)
        tensors[f"blk.{i}.ffn_norm.weight"] = rng.standard_normal(D).astype(np.float32)
        tensors[f"blk.{i}.ffn_gate.weight"] = rng.standard_normal((DFF, D)).astype(np.float32)
        tensors[f"blk.{i}.ffn_up.weight"] = rng.standard_normal((DFF, D)).astype(np.float32)
        tensors[f"blk.{i}.ffn_down.weight"] = rng.standard_normal((D, DFF)).astype(np.float32)
        if lt[i] == "attn":
            tensors[f"blk.{i}.attn_q.weight"] = rng.standard_normal((D, D)).astype(np.float32)
            tensors[f"blk.{i}.attn_k.weight"] = rng.standard_normal((KVH * HD, D)).astype(np.float32)
            tensors[f"blk.{i}.attn_v.weight"] = rng.standard_normal((KVH * HD, D)).astype(np.float32)
            tensors[f"blk.{i}.attn_q_norm.weight"] = rng.standard_normal(HD).astype(np.float32)
            tensors[f"blk.{i}.attn_k_norm.weight"] = rng.standard_normal(HD).astype(np.float32)
            tensors[f"blk.{i}.attn_output.weight"] = rng.standard_normal((D, D)).astype(np.float32)
        else:
            tensors[f"blk.{i}.shortconv.in_proj.weight"] = rng.standard_normal((3 * D, D)).astype(np.float32)
            tensors[f"blk.{i}.shortconv.conv.weight"] = rng.standard_normal((D, 3)).astype(np.float32)
            tensors[f"blk.{i}.shortconv.out_proj.weight"] = rng.standard_normal((D, D)).astype(np.float32)
    meta = {
        "general.architecture": "lfm2",
        "lfm2.block_count": L,
        "lfm2.embedding_length": D,
        "lfm2.attention.head_count": H,
        "lfm2.feed_forward_length": DFF,
        "lfm2.rope.freq_base": 1e7,
        "lfm2.context_length": 128,
        "lfm2.attention.layer_norm_rms_epsilon": 1e-5,
        "lfm2.shortconv.l_cache": 3,
        "tokenizer.ggml.tokens": ["a", "b"],
        "tokenizer.ggml.merges": ["a b"],
    }
    path = tmp_path / "lfm2.gguf"
    _write_gguf(path, meta, tensors)
    out = export_gguf(path)
    cfg = out["config"]
    assert cfg["arch"] == "lfm2"
    assert cfg["layer_types"] == lt
    assert cfg["n_kv_heads"] == KVH and cfg["head_dim"] == HD
    assert cfg["conv_l_cache"] == 3 and cfg["norm_eps"] == pytest.approx(1e-5, rel=1e-6)
    assert cfg["tie_weights"] is True
    assert cfg["n_heads"] == H and cfg["d_model"] == D and cfg["n_layers"] == L
    # flat size: emb + 2 attn layers + 4 conv layers + final norm (tied head)
    attn_n = 2 * D * D + 2 * D * (KVH * HD) + 2 * D * DFF + DFF * D + 2 * D + 2 * HD
    conv_n = 4 * D * D + 2 * D * DFF + DFF * D + 2 * D + 3 * D
    f16 = f16_decode(out["weights_b64"])
    assert len(f16) == V * D + 2 * attn_n + 4 * conv_n + D
    # spot-check: blk.2 (attn) layout starts after emb + layers 0,1 (conv)
    off2 = V * D + 2 * conv_n
    f32 = np.array(f16, dtype=np.float16).astype(np.float32)
    # layer 2: attn_norm [D], then attn_q [D, D]
    q_flat = f32[off2 + D: off2 + D + D * D]
    assert np.allclose(q_flat.reshape(D, D), tensors["blk.2.attn_q.weight"].T, atol=1e-3)
    # conv layer 0: attn_norm [D], then in_proj [D, 3D]
    inp_flat = f32[V * D + D: V * D + D + D * 3 * D]
    assert np.allclose(inp_flat.reshape(D, 3 * D), tensors["blk.0.shortconv.in_proj.weight"].T, atol=1e-3)
    # conv weight ne (3, D) -> wire (D, 3), no transpose
    cv_flat = f32[V * D + D + D * 3 * D: V * D + D + D * 3 * D + D * 3]
    assert np.allclose(cv_flat.reshape(D, 3), tensors["blk.0.shortconv.conv.weight"], atol=1e-3)
    # tokenizer embedded
    assert out["vocab"] == {"a": 0, "b": 1} and out["merges"] == ["a b"]
    assert cfg["tokenizer"] == "lfm2:bpe"


def test_export_gguf_qwen35(tmp_path):
    """Synthetic qwen35 GGUF (GatedDeltaNet hybrid) -> wire: config, layout,
    flat sizes, tied head, tokenizer. Mirrors the real Qwen3.5-4B pattern:
    layers where (i+1) % 4 == 0 are full attention, the rest are linear."""
    import numpy as np
    from tools.export_model import f16_decode
    D, H, KVH, HD, DFF, L, V = 32, 4, 2, 8, 64, 6, 16
    DK, NK, DV, NV, KERN = 4, 2, 4, 4, 3  # head_k_dim, k heads, head_v_dim, v heads
    KD, VD = DK * NK, DV * NV
    CONV = KD * 2 + VD
    rng = np.random.default_rng(11)
    lt = ["linear", "linear", "linear", "full", "linear", "linear"]
    tensors = {"token_embd.weight": rng.standard_normal((V, D)).astype(np.float32)}
    for i in range(L):
        tensors[f"blk.{i}.attn_norm.weight"] = rng.standard_normal(D).astype(np.float32)
        tensors[f"blk.{i}.post_attention_norm.weight"] = rng.standard_normal(D).astype(np.float32)
        tensors[f"blk.{i}.ffn_gate.weight"] = rng.standard_normal((DFF, D)).astype(np.float32)
        tensors[f"blk.{i}.ffn_up.weight"] = rng.standard_normal((DFF, D)).astype(np.float32)
        tensors[f"blk.{i}.ffn_down.weight"] = rng.standard_normal((D, DFF)).astype(np.float32)
        if lt[i] == "full":
            tensors[f"blk.{i}.attn_q.weight"] = rng.standard_normal((2 * H * HD, D)).astype(np.float32)
            tensors[f"blk.{i}.attn_q_norm.weight"] = rng.standard_normal(HD).astype(np.float32)
            tensors[f"blk.{i}.attn_k.weight"] = rng.standard_normal((KVH * HD, D)).astype(np.float32)
            tensors[f"blk.{i}.attn_k_norm.weight"] = rng.standard_normal(HD).astype(np.float32)
            tensors[f"blk.{i}.attn_v.weight"] = rng.standard_normal((KVH * HD, D)).astype(np.float32)
            tensors[f"blk.{i}.attn_output.weight"] = rng.standard_normal((D, H * HD)).astype(np.float32)
        else:
            tensors[f"blk.{i}.attn_qkv.weight"] = rng.standard_normal((CONV, D)).astype(np.float32)
            tensors[f"blk.{i}.attn_gate.weight"] = rng.standard_normal((VD, D)).astype(np.float32)
            tensors[f"blk.{i}.ssm_a"] = rng.standard_normal(NV).astype(np.float32)
            tensors[f"blk.{i}.ssm_alpha.weight"] = rng.standard_normal((NV, D)).astype(np.float32)
            tensors[f"blk.{i}.ssm_beta.weight"] = rng.standard_normal((NV, D)).astype(np.float32)
            tensors[f"blk.{i}.ssm_dt.bias"] = rng.standard_normal(NV).astype(np.float32)
            tensors[f"blk.{i}.ssm_conv1d.weight"] = rng.standard_normal((CONV, KERN)).astype(np.float32)
            tensors[f"blk.{i}.ssm_norm.weight"] = rng.standard_normal(DV).astype(np.float32)
            tensors[f"blk.{i}.ssm_out.weight"] = rng.standard_normal((D, VD)).astype(np.float32)
    meta = {
        "general.architecture": "qwen35",
        "qwen35.block_count": L,
        "qwen35.embedding_length": D,
        "qwen35.attention.head_count": H,
        "qwen35.attention.head_count_kv": KVH,
        "qwen35.attention.key_length": HD,
        "qwen35.attention.value_length": HD,
        "qwen35.feed_forward_length": DFF,
        "qwen35.rope.freq_base": 1e7,
        "qwen35.rope.dimension_count": 4,
        "qwen35.context_length": 128,
        "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen35.full_attention_interval": 4,
        "qwen35.ssm.conv_kernel": KERN,
        "qwen35.ssm.inner_size": VD,
        "qwen35.ssm.state_size": DK,
        "qwen35.ssm.time_step_rank": NV,
        "qwen35.ssm.group_count": NK,
        "tokenizer.ggml.tokens": ["a", "b"],
        "tokenizer.ggml.merges": ["a b"],
    }
    path = tmp_path / "qwen35.gguf"
    _write_gguf(path, meta, tensors)
    out = export_gguf(path)
    cfg = out["config"]
    assert cfg["arch"] == "qwen35"
    assert cfg["layer_types"] == lt
    assert cfg["n_kv_heads"] == KVH and cfg["head_dim"] == HD
    assert cfg["rope_dim"] == 4
    s = cfg["ssm"]
    assert s["conv_kernel"] == KERN and s["inner_size"] == VD
    assert s["state_size"] == DK and s["dt_rank"] == NV and s["group_count"] == NK
    assert s["q_dim"] == KD and s["v_dim"] == VD and s["conv_dim"] == CONV
    assert cfg["tie_weights"] is True
    # flat size: emb + 1 full layer + 5 linear layers + final norm (tied head)
    full_n = D + 2 * H * HD * D + HD + KVH * HD * D + HD + KVH * HD * D + H * HD * D + 2 * D * DFF + DFF * D
    lin_n = (D + CONV * D + VD * D + NV * (2 + 2 * D) + CONV * KERN + DV + VD * D
             + D + 2 * D * DFF + DFF * D)
    f16 = f16_decode(out["weights_b64"])
    assert len(f16) == V * D + full_n + 5 * lin_n + D
    f32 = np.array(f16, dtype=np.float16).astype(np.float32)
    # linear layer 0: norm [D], qkv [D, CONV]
    qkv_flat = f32[V * D + D: V * D + D + D * CONV]
    assert np.allclose(qkv_flat.reshape(D, CONV), tensors["blk.0.attn_qkv.weight"].T, atol=1e-3)
    # conv1d stored [CONV, KERN] as-is (no transpose)
    off_conv = V * D + D + CONV * D + VD * D + NV * (2 + 2 * D)
    cv_flat = f32[off_conv: off_conv + CONV * KERN]
    assert np.allclose(cv_flat.reshape(CONV, KERN), tensors["blk.0.ssm_conv1d.weight"], atol=1e-3)
    # full layer 3: after 3 linear layers: norm [D], then q [D, 2*H*HD]
    off3 = V * D + 3 * lin_n + D
    q_flat = f32[off3: off3 + D * 2 * H * HD]
    assert np.allclose(q_flat.reshape(D, 2 * H * HD), tensors["blk.3.attn_q.weight"].T, atol=1e-3)
    assert out["vocab"] == {"a": 0, "b": 1} and out["merges"] == ["a b"]
    assert cfg["tokenizer"] == "qwen2:bpe"
