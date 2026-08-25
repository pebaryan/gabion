"""Safetensors / HuggingFace export path (tools/export_model.py::export_hf)."""
import json
import struct

import numpy as np
import pytest

from tools.export_model import export_hf, f16_decode, main as export_main


def _f16_from_bytes(b: bytes) -> np.ndarray:
    """Raw little-endian f16 bytes -> float32 array (f16_decode expects base64)."""
    return np.frombuffer(b, dtype="<u2").view(np.float16).astype(np.float32)


def _write_st(path, tensors: dict):
    """Write a minimal safetensors file. tensors: {name: np.ndarray}."""
    hdr = {}
    payload = b""
    for name, arr in tensors.items():
        dt = arr.dtype
        dtype = {"float32": "F32", "float16": "F16", "int32": "I32"}[dt.name]
        hdr[name] = {"dtype": dtype, "shape": list(arr.shape),
                     "data_offsets": [len(payload), len(payload) + arr.nbytes]}
        payload += arr.tobytes()
    with open(path, "wb") as f:
        raw = json.dumps(hdr).encode("utf-8")
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(payload)


def _write_hf_dir(tmp_path, *, D=8, heads=2, kv=2, L=2, d_ff=16, vocab=8,
                  tied=True, dtype="F32", sharded=False, bias=False):
    """Build a tiny llama-family checkpoint dir; returns the tensor dict."""
    tensors = {}
    rng = np.random.default_rng(1234)
    dt = np.dtype("float32") if dtype == "F32" else np.dtype("float16")
    tensors["model.embed_tokens.weight"] = rng.standard_normal((vocab, D)).astype(dt)
    for i in range(L):
        tensors[f"model.layers.{i}.input_layernorm.weight"] = rng.standard_normal(D).astype(dt)
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = rng.standard_normal((D, D)).astype(dt)
        tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = rng.standard_normal((kv * (D // heads), D)).astype(dt)
        tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = rng.standard_normal((kv * (D // heads), D)).astype(dt)
        tensors[f"model.layers.{i}.self_attn.o_proj.weight"] = rng.standard_normal((D, D)).astype(dt)
        tensors[f"model.layers.{i}.post_attention_layernorm.weight"] = rng.standard_normal(D).astype(dt)
        tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = rng.standard_normal((d_ff, D)).astype(dt)
        tensors[f"model.layers.{i}.mlp.up_proj.weight"] = rng.standard_normal((d_ff, D)).astype(dt)
        tensors[f"model.layers.{i}.mlp.down_proj.weight"] = rng.standard_normal((D, d_ff)).astype(dt)
        if bias:
            tensors[f"model.layers.{i}.self_attn.q_proj.bias"] = rng.standard_normal(D).astype(dt)
            tensors[f"model.layers.{i}.self_attn.k_proj.bias"] = rng.standard_normal(kv * (D // heads)).astype(dt)
            tensors[f"model.layers.{i}.self_attn.v_proj.bias"] = rng.standard_normal(kv * (D // heads)).astype(dt)
    tensors["model.norm.weight"] = rng.standard_normal(D).astype(dt)
    if not tied:
        tensors["lm_head.weight"] = rng.standard_normal((vocab, D)).astype(dt)

    cfg = {
        "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
        "hidden_size": D, "num_hidden_layers": L,
        "num_attention_heads": heads, "num_key_value_heads": kv,
        "intermediate_size": d_ff, "vocab_size": vocab,
        "max_position_embeddings": 4096, "rope_theta": 10000.0,
        "tie_word_embeddings": tied, "rms_norm_eps": 1e-6,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    if sharded:
        names = sorted(tensors)
        half = len(names) // 2
        shard1 = {n: tensors[n] for n in names[:half]}
        shard2 = {n: tensors[n] for n in names[half:]}
        _write_st(tmp_path / "model-00001-of-00002.safetensors", shard1)
        _write_st(tmp_path / "model-00002-of-00002.safetensors", shard2)
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {**{n: "model-00001-of-00002.safetensors" for n in shard1},
                           **{n: "model-00002-of-00002.safetensors" for n in shard2}}}))
    else:
        _write_st(tmp_path / "model.safetensors", tensors)

    tok = {"tokenizer_class": "Qwen2Tokenizer", "model": {
        "type": "BPE",
        "vocab": {"<unk>": 0, "a": 1, "b": 2, "ab": 3, "Ġ": 4, "Ġa": 5, "#": 6, "##": 7},
        "merges": ["a b", "Ġ a", "Ġa b", "# #"], "fuse_unk": False}}
    (tmp_path / "tokenizer.json").write_text(json.dumps(tok))
    return tensors


def test_export_hf_basic(tmp_path):
    t = _write_hf_dir(tmp_path, D=8, heads=2, kv=2, L=2, d_ff=16, vocab=8, tied=True)
    out = export_hf(tmp_path)
    cfg = out["config"]
    assert cfg["d_model"] == 8 and cfg["n_heads"] == 2 and cfg["n_kv_heads"] == 2
    assert cfg["n_layers"] == 2 and cfg["d_ff"] == 16 and cfg["vocab_size"] == 8
    assert cfg["tie_weights"] is True and cfg["rope_base"] == 10000.0
    # wire order: [emb, (q,k,v,o,n1,gate_up,n2,down)*L, norm] (tied)
    w = f16_decode(out["weights_b64"])
    off = 0
    exp = [t["model.embed_tokens.weight"].astype(np.float32)]
    for i in range(2):
        exp.append(t[f"model.layers.{i}.self_attn.q_proj.weight"].T.astype(np.float32))
        exp.append(t[f"model.layers.{i}.self_attn.k_proj.weight"].T.astype(np.float32))
        exp.append(t[f"model.layers.{i}.self_attn.v_proj.weight"].T.astype(np.float32))
        exp.append(t[f"model.layers.{i}.self_attn.o_proj.weight"].T.astype(np.float32))
        exp.append(t[f"model.layers.{i}.input_layernorm.weight"].astype(np.float32))
        exp.append(np.concatenate([t[f"model.layers.{i}.mlp.gate_proj.weight"].T,
                                   t[f"model.layers.{i}.mlp.up_proj.weight"].T], axis=1).astype(np.float32))
        exp.append(t[f"model.layers.{i}.post_attention_layernorm.weight"].astype(np.float32))
        exp.append(t[f"model.layers.{i}.mlp.down_proj.weight"].T.astype(np.float32))
    exp.append(t["model.norm.weight"].astype(np.float32))
    ref = np.concatenate([np.asarray(e, np.float32).reshape(-1) for e in exp]).astype(np.float16)
    assert w.shape == ref.shape
    assert np.array_equal(w.astype(np.float32), ref.astype(np.float32)), "f16 wire mismatch"


def test_export_hf_untied(tmp_path):
    t = _write_hf_dir(tmp_path, tied=False)
    out = export_hf(tmp_path)
    assert out["config"]["tie_weights"] is False
    w = f16_decode(out["weights_b64"])
    # last block is lm_head (vocab, D) -> transposed (D, vocab)
    n_vocab, D = 8, 8
    tail = w[-n_vocab * D:]
    assert np.array_equal(tail.astype(np.float32),
                          t["lm_head.weight"].T.astype(np.float32).reshape(-1).astype(np.float16).astype(np.float32))


def test_export_hf_sharded(tmp_path):
    _write_hf_dir(tmp_path, sharded=True)
    out = export_hf(tmp_path)
    assert out["config"]["n_layers"] == 2


def test_export_hf_tokenizer(tmp_path):
    _write_hf_dir(tmp_path)
    out = export_hf(tmp_path)
    assert out["vocab"] == {"<unk>": 0, "a": 1, "b": 2, "ab": 3, "Ġ": 4, "Ġa": 5, "#": 6, "##": 7}
    # merges referencing unknown tokens are dropped; '# #' stays (real BPE)
    assert out["merges"] == ["a b", "Ġ a", "Ġa b", "# #"]
    assert out["config"]["tokenizer"] == "hf:Qwen2Tokenizer"


def test_export_hf_specials_and_chat_template(tmp_path):
    _write_hf_dir(tmp_path)
    tok = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))
    # added_tokens with special:true -> wire["special"]; chat_template passes through
    tok["model"]["vocab"]["<|im_start|>"] = 100
    tok["model"]["vocab"]["<|im_end|>"] = 101
    tok["added_tokens"] = [
        {"id": 100, "content": "<|im_start|>", "lstrip": False, "rstrip": False, "special": True},
        {"id": 101, "content": "<|im_end|>", "lstrip": False, "rstrip": False, "special": True},
        {"id": 102, "content": "not_special", "special": False},
    ]
    tok["chat_template"] = "{{ messages[0]['content'] }}"
    (tmp_path / "tokenizer.json").write_text(json.dumps(tok))
    out = export_hf(tmp_path)
    assert out["special"] == ["<|im_start|>", "<|im_end|>"]
    assert out["chat_template"] == "{{ messages[0]['content'] }}"
    # added tokens NOT in model.vocab are merged in (they live only in added_tokens)
    tok2 = {"tokenizer_class": "Qwen2Tokenizer", "model": {
        "type": "BPE",
        "vocab": {"a": 0, "b": 1, "ab": 2},
        "merges": ["a b"]},
        "added_tokens": [{"id": 5, "content": "<|im_start|>", "special": True}]}
    (tmp_path / "tokenizer.json").write_text(json.dumps(tok2))
    out2 = export_hf(tmp_path)
    assert out2["vocab"]["<|im_start|>"] == 5
    assert out2["special"] == ["<|im_start|>"]
    # chat_template falls back to tokenizer_config.json
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "fallback"}))
    tok3 = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))
    tok3.pop("chat_template", None)
    (tmp_path / "tokenizer.json").write_text(json.dumps(tok3))
    out3 = export_hf(tmp_path)
    assert out3["chat_template"] == "fallback"


def test_export_hf_attention_bias(tmp_path):
    t = _write_hf_dir(tmp_path, D=8, heads=2, kv=2, L=2, d_ff=16, vocab=8, bias=True)
    out = export_hf(tmp_path)
    assert len(out["q_bias"]) == 2 and len(out["k_bias"]) == 2 and len(out["v_bias"]) == 2
    assert len(out["q_bias"][0]) == 8 and len(out["k_bias"][0]) == 8  # q: [D], k/v: [kv*hd]
    np.testing.assert_allclose(out["q_bias"][0], t["model.layers.0.self_attn.q_proj.bias"], atol=1e-6)
    np.testing.assert_allclose(out["k_bias"][1], t["model.layers.1.self_attn.k_proj.bias"], atol=1e-6)
    assert out["config"]["attention_bias"] is True


def test_export_hf_no_bias_by_default(tmp_path):
    _write_hf_dir(tmp_path)
    out = export_hf(tmp_path)
    assert "q_bias" not in out
    assert out["config"].get("attention_bias") is False


def test_export_hf_shards_reconstruct(tmp_path, capsys):
    """--out-shards slices the flat: coordinator + shards reassemble byte-exact."""
    t = _write_hf_dir(tmp_path, D=8, heads=2, kv=2, L=4, d_ff=16, vocab=8, bias=True)
    out = export_hf(tmp_path)
    full = f16_decode(out["weights_b64"])
    cfg = out["config"]
    D, V, L = cfg["d_model"], cfg["vocab_size"], cfg["n_layers"]
    kvD = D // cfg["n_heads"] * cfg["n_kv_heads"]
    dFF = cfg["d_ff"]
    per = 2 * D * D + 2 * D * kvD + 2 * D + 2 * D * dFF + dFF * D
    tok_n = V * D

    shard_dir = tmp_path / "shards"
    rc = export_main(["--from-hf", str(tmp_path), "--out-shards", "2", str(shard_dir)])
    assert rc == 0
    m = json.loads((shard_dir / "model.json").read_text())
    assert [s["layers"] for s in m["shards"] if s] == [[0, 2], [2, 4]]
    assert m["shards"][0]["n_layers"] == 2
    # coordinator = embedding + tail (final norm); shards = contiguous layer slices
    coord = _f16_from_bytes((shard_dir / "coord.f16").read_bytes())
    np.testing.assert_array_equal(coord, np.concatenate([full[:tok_n], full[tok_n + L * per:]]))
    for s in m["shards"]:
        if not s:
            continue
        a, b = s["layers"]
        sl = _f16_from_bytes((shard_dir / s["file"]).read_bytes())
        np.testing.assert_array_equal(sl, full[tok_n + a * per: tok_n + b * per])
    # byte-exact reassembly of the full flat
    rebuilt = np.concatenate([coord[:tok_n], *[_f16_from_bytes((shard_dir / s["file"]).read_bytes())
                                               for s in m["shards"] if s], coord[tok_n:]])
    np.testing.assert_array_equal(rebuilt, full)
    # biases ride in the manifest untouched
    assert len(m["q_bias"]) == L and m["config"]["attention_bias"] is True


def test_export_hf_f16(tmp_path):
    _write_hf_dir(tmp_path, dtype="F16")
    out = export_hf(tmp_path)
    assert out["config"]["d_model"] == 8
    w = f16_decode(out["weights_b64"])
    assert np.isfinite(w).all()


def test_export_hf_gqa_reject(tmp_path):
    _write_hf_dir(tmp_path, heads=4, kv=8)
    with pytest.raises(ValueError, match="more KV heads than query heads"):
        export_hf(tmp_path)


def test_export_hf_missing_embed(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 2, "intermediate_size": 16, "vocab_size": 8}))
    _write_st(tmp_path / "model.safetensors", {"model.norm.weight": np.zeros(8, np.float32)})
    with pytest.raises(ValueError, match="embed_tokens"):
        export_hf(tmp_path)


def test_st_tensor_bf16_and_counts():
    """_st_tensor: BF16 upcast + the byte-vs-element count fix (old code read
    2x/4x the tensor with multi-byte dtypes)."""
    import mmap as _mmap
    from tools.export_model import _st_tensor
    rng = np.random.default_rng(9)
    f32 = (rng.standard_normal(64) * 0.5).astype(np.float32)
    bf16 = (f32.view(np.uint32) >> 16).astype(np.uint16)
    f16 = f32.astype(np.float16)
    i32 = rng.integers(-100, 100, 32).astype(np.int32)
    payload = bf16.tobytes() + f16.tobytes() + i32.tobytes()
    hdr = {
        "a": {"dtype": "BF16", "shape": [64], "data_offsets": [0, 128]},
        "b": {"dtype": "F16", "shape": [64], "data_offsets": [128, 256]},
        "c": {"dtype": "I32", "shape": [32], "data_offsets": [256, 384]},
    }
    raw = struct.pack("<Q", len(json.dumps(hdr).encode())) + json.dumps(hdr).encode() + payload
    hdr_len = struct.unpack_from("<Q", raw, 0)[0]
    a = _st_tensor(raw, hdr_len, hdr["a"])
    b = _st_tensor(raw, hdr_len, hdr["b"])
    c = _st_tensor(raw, hdr_len, hdr["c"])
    assert a.shape == (64,) and b.shape == (64,) and c.shape == (32,)
    # BF16 upcast = top 16 bits of the f32 pattern, bit-exact
    assert np.array_equal(a, (bf16.astype(np.int32) << 16).view(np.float32))
    assert np.array_equal(b, f32.astype(np.float16))
    assert np.array_equal(c, i32)
    with pytest.raises(NotImplementedError):
        _st_tensor(raw, hdr_len, {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]})
