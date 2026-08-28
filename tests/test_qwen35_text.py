from __future__ import annotations

import numpy as np
import pytest


def _fixture(tmp_path):
    # Reuse the minimal GGUF writer used by exporter tests.  This exercises the
    # real GGUF dimension convention while keeping CI small enough for CUDA/CPU.
    from test_export_model import _write_gguf

    rng = np.random.default_rng(23)
    D, H, KV, HD, DFF, L, V = 16, 2, 1, 8, 32, 5, 19
    DK, NK, DV, NV, KERN = 2, 1, 2, 2, 3
    KD, VD, CONV = DK * NK, DV * NV, 2 * DK * NK + DV * NV
    tensors = {
        "token_embd.weight": rng.normal(0, 0.03, (V, D)).astype(np.float32),
        "output.weight": rng.normal(0, 0.03, (V, D)).astype(np.float32),
        "output_norm.weight": np.ones(D, dtype=np.float32),
    }
    kinds = ["linear", "linear", "linear", "full", "full"]
    for i, kind in enumerate(kinds):
        p = f"blk.{i}."
        tensors[p + "attn_norm.weight"] = np.ones(D, dtype=np.float32)
        tensors[p + "post_attention_norm.weight"] = np.ones(D, dtype=np.float32)
        tensors[p + "ffn_gate.weight"] = rng.normal(0, 0.03, (DFF, D)).astype(np.float32)
        tensors[p + "ffn_up.weight"] = rng.normal(0, 0.03, (DFF, D)).astype(np.float32)
        tensors[p + "ffn_down.weight"] = rng.normal(0, 0.03, (D, DFF)).astype(np.float32)
        if kind == "linear":
            tensors[p + "attn_qkv.weight"] = rng.normal(0, 0.03, (CONV, D)).astype(np.float32)
            tensors[p + "attn_gate.weight"] = rng.normal(0, 0.03, (VD, D)).astype(np.float32)
            tensors[p + "ssm_a"] = np.zeros(NV, dtype=np.float32)
            tensors[p + "ssm_alpha.weight"] = rng.normal(0, 0.03, (NV, D)).astype(np.float32)
            tensors[p + "ssm_beta.weight"] = rng.normal(0, 0.03, (NV, D)).astype(np.float32)
            tensors[p + "ssm_dt.bias"] = np.zeros(NV, dtype=np.float32)
            tensors[p + "ssm_conv1d.weight"] = rng.normal(0, 0.03, (CONV, KERN)).astype(np.float32)
            tensors[p + "ssm_norm.weight"] = np.ones(DV, dtype=np.float32)
            tensors[p + "ssm_out.weight"] = rng.normal(0, 0.03, (D, VD)).astype(np.float32)
        else:
            tensors[p + "attn_q.weight"] = rng.normal(0, 0.03, (2 * H * HD, D)).astype(np.float32)
            tensors[p + "attn_q_norm.weight"] = np.ones(HD, dtype=np.float32)
            tensors[p + "attn_k.weight"] = rng.normal(0, 0.03, (KV * HD, D)).astype(np.float32)
            tensors[p + "attn_k_norm.weight"] = np.ones(HD, dtype=np.float32)
            tensors[p + "attn_v.weight"] = rng.normal(0, 0.03, (KV * HD, D)).astype(np.float32)
            tensors[p + "attn_output.weight"] = rng.normal(0, 0.03, (D, H * HD)).astype(np.float32)
    meta = {
        "general.architecture": "qwen35",
        "qwen35.block_count": L,
        "qwen35.embedding_length": D,
        "qwen35.attention.head_count": H,
        "qwen35.attention.head_count_kv": KV,
        "qwen35.attention.key_length": HD,
        "qwen35.feed_forward_length": DFF,
        "qwen35.rope.freq_base": 10000.0,
        "qwen35.rope.dimension_count": 4,
        "qwen35.context_length": 64,
        "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen35.full_attention_interval": 4,
        "qwen35.ssm.conv_kernel": KERN,
        "qwen35.ssm.inner_size": VD,
        "qwen35.ssm.state_size": DK,
        "qwen35.ssm.time_step_rank": NV,
        "qwen35.ssm.group_count": NK,
    }
    path = tmp_path / "qwen35-fixture.gguf"
    _write_gguf(path, meta, tensors)
    return path, tensors


def test_qwen35_layer_presence_and_pipeline_parity(tmp_path):
    from gabion.user_models.qwen35_text import Qwen35TextAdapter, Qwen35TextPipeline
    from tools.export_model import export_gguf

    path, _ = _fixture(tmp_path)
    wire = export_gguf(path)
    assert wire["config"]["layer_types"] == ["linear", "linear", "linear", "full", "full"]
    assert wire["config"]["tie_weights"] is False

    # The fast path processes rows one at a time through a TinyJit per entry
    # point.  Two different shard splits must produce identical logits, since
    # every layer sees the same input in both pipelines (states align at
    # positions 0..T-1).  This exercises sharding, the GPU GDN scan, the
    # full-attention cache and the fused matmul routing end to end.
    ids = [2, 4, 7]

    left = Qwen35TextAdapter.from_gguf_shard(path, 0, 2)
    right = Qwen35TextAdapter.from_gguf_shard(path, 1, 2)
    s0, s1 = left.new_state(32), right.new_state(32)
    hidden = left.forward_shard_ids_to_hidden(ids, s0)
    assert hidden.shape == (3, left.d_model)
    got = right.forward_shard_hidden_to_logits(hidden, s1)

    left2 = Qwen35TextAdapter.from_gguf_shard(path, 0, 3)
    mid2 = Qwen35TextAdapter.from_gguf_shard(path, 1, 3)
    right2 = Qwen35TextAdapter.from_gguf_shard(path, 2, 3)
    hidden2 = left2.forward_shard_ids_to_hidden(ids, left2.new_state(32))
    h2 = mid2.forward_shard_hidden_to_hidden(hidden2, mid2.new_state(32))
    got2 = right2.forward_shard_hidden_to_logits(h2, right2.new_state(32))

    np.testing.assert_allclose(got, got2, rtol=1e-3, atol=1e-3)
    # both pipelines must agree on the greedy argmax
    assert int(np.argmax(got)) == int(np.argmax(got2))

    # The local dual-device fast path must preserve the mesh-facing split
    # result while avoiding the hidden NumPy round-trip between shards.
    native_left = Qwen35TextAdapter.from_gguf_shard(path, 0, 2)
    native_right = Qwen35TextAdapter.from_gguf_shard(path, 1, 2)
    native = Qwen35TextPipeline(native_left, native_right)
    ns0, ns1 = native_left.new_state(32), native_right.new_state(32)
    got_native = native.run(ids, ns0, ns1)
    np.testing.assert_allclose(got, got_native, rtol=1e-3, atol=1e-3)
    assert ns0["pos"] == ns1["pos"] == len(ids)
    gs0, gs1 = native_left.new_state(32), native_right.new_state(32)
    assert native.run_greedy(ids, gs0, gs1) == int(np.argmax(got))


def test_qwen35_decode_keeps_state_between_tokens(tmp_path):
    from gabion.user_models.qwen35_text import Qwen35TextAdapter

    path, _ = _fixture(tmp_path)
    model = Qwen35TextAdapter.from_gguf(path)
    state = model.stream_state("s", reset=True)
    first = model.forward_shard_ids_to_hidden([2, 4], state)
    pos = state["pos"]
    second = model.forward_shard_ids_to_hidden([7], state)
    assert first.shape == (2, model.d_model)
    assert second.shape == (1, model.d_model)
    assert pos == 2 and state["pos"] == 3
    model.release_stream("s")
    assert "s" not in model._stream_states


def test_qwen35_q5_gemv_matches_reference():
    from tinygrad import Device, Tensor, dtypes

    if str(Device.DEFAULT) != "CUDA":
        pytest.skip("Q5_K GEMV check requires CUDA")
    from gabion.user_models.qwen35_text import _QWeight

    rng = np.random.default_rng(37)
    out_dim, in_dim = 256, 512
    blk = rng.integers(0, 256, size=(out_dim * (in_dim // 256), 176), dtype=np.uint8)
    blk[:, 0:2] = np.frombuffer(np.float16(0.02).tobytes(), dtype=np.uint8)
    blk[:, 2:4] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)
    weight = _QWeight("CUDA", "Q5_K", blk, (out_dim, in_dim))
    x = Tensor(rng.normal(0, 0.15, size=(1, in_dim)).astype(np.float32), device="CUDA", dtype=dtypes.float32).realize()
    ref = (x @ weight.dequant().T).realize().numpy().reshape(-1)
    got = weight.gemv(x).realize().numpy().reshape(-1)
    np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-3)


def test_qwen35_iq4_xs_gpu_dequant_matches_reference():
    from tinygrad import Device

    if str(Device.DEFAULT) != "CUDA":
        pytest.skip("IQ4_XS adapter check requires CUDA")
    from gabion.user_models.qwen35_text import Qwen35TextAdapter
    from tools.export_model import _dequant

    rng = np.random.default_rng(31)
    raw = bytearray(rng.integers(0, 256, size=136, dtype=np.uint8).tobytes())
    raw[0:2] = bytes((0, 0x3C))  # finite f16 scale 1.0
    raw = bytes(raw)
    model = Qwen35TextAdapter(
        vocab_size=1, d_model=1, n_heads=1, n_kv_heads=1, n_layers=1,
        d_ff=1, layer_types=["linear"], head_dim=1, rope_base=1.0,
        rope_dim=1, norm_eps=1e-6,
        ssm={"conv_kernel": 1, "state_size": 1, "num_k_heads": 1,
             "head_k_dim": 1, "num_v_heads": 1, "head_v_dim": 1},
        tie_weights=False, seq_len=1,
    )
    model._gguf_buf = raw
    got = model._gpu_dequant_k((256, 1), "IQ4_XS", 0).numpy().reshape(-1)
    want = _dequant("fixture", raw, (256, 1), "IQ4_XS").reshape(-1)
    # The CUDA path intentionally materializes f16 weights; the reference is
    # f32, so values near a large f16 bin can differ by one quantized unit.
    np.testing.assert_allclose(got, want, rtol=0, atol=1)
