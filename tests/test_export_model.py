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


def _write_gguf(path: Path, meta: dict, tensors: dict[str, np.ndarray]) -> None:
    """Write a minimal GGUF v3 file with F32/F16/Q8_0 tensors."""
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
        else:
            raise TypeError(type(v))

    for k, v in meta.items():
        put_str(k)
        put_val(v)

    # tensor infos: name, n_dims, dims, ggml_type, offset (offsets fixed up below)
    infos = []
    for name, arr in tensors.items():
        put_str(name)
        out += struct.pack("<I", arr.ndim)
        for d in arr.shape[::-1]:
            out += struct.pack("<Q", d)
        gtype = 0 if arr.dtype == np.float32 else 1
        out += struct.pack("<I", gtype)
        out += struct.pack("<Q", 0)  # placeholder offset

    # data (F32/F16 only for the hand-written file)
    offsets = {}
    for name, arr in tensors.items():
        offsets[name] = len(out)
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
    for name, arr in tensors.items():
        out += np.asarray(arr, dtype="<f4" if arr.dtype == np.float32 else "<f2").tobytes()
    Path(path).write_bytes(bytes(out))


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
        "blk.0.attn_q.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.attn_k.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.attn_v.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.attn_output.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.ffn_gate.weight": rng.normal(0, 0.1, (D, dFF)).astype(np.float32),
        "blk.0.ffn_up.weight": rng.normal(0, 0.1, (D, dFF)).astype(np.float32),
        "blk.0.ffn_down.weight": rng.normal(0, 0.1, (dFF, D)).astype(np.float32),
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
    assert np.max(np.abs(gu[:, :dFF] - tensors["blk.0.ffn_gate.weight"])) < 1e-2
    assert np.max(np.abs(gu[:, dFF:] - tensors["blk.0.ffn_up.weight"])) < 1e-2


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
        "blk.0.attn_q.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.attn_k.weight": rng.normal(0, 0.1, (D, kvD)).astype(np.float32),
        "blk.0.attn_v.weight": rng.normal(0, 0.1, (D, kvD)).astype(np.float32),
        "blk.0.attn_output.weight": rng.normal(0, 0.1, (D, D)).astype(np.float32),
        "blk.0.attn_norm.weight": rng.normal(0, 0.1, (D,)).astype(np.float32),
        "blk.0.ffn_gate.weight": rng.normal(0, 0.1, (D, 32)).astype(np.float32),
        "blk.0.ffn_up.weight": rng.normal(0, 0.1, (D, 32)).astype(np.float32),
        "blk.0.ffn_norm.weight": rng.normal(0, 0.1, (D,)).astype(np.float32),
        "blk.0.ffn_down.weight": rng.normal(0, 0.1, (32, D)).astype(np.float32),
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
    assert np.allclose(k, tensors["blk.0.attn_k.weight"], atol=1e-2)
    # gate_up holds gate then up (after q,k,v,o,n1)
    gu_off = n_tok + 2 * D * D + 2 * D * kvD + D
    gu = flat[gu_off:gu_off + 2 * D * 32].reshape(D, 2 * 32)
    assert np.allclose(gu[:, :32], tensors["blk.0.ffn_gate.weight"], atol=1e-2)
    assert np.allclose(gu[:, 32:], tensors["blk.0.ffn_up.weight"], atol=1e-2)


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
