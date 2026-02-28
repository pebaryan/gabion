"""Tests for BBTTransformerAdapter aligned with BitByteLM-style behavior."""

from pathlib import Path

import pytest

from gabion.user_models.bbt_transformer import BBTTransformerAdapter


class TestBBTTransformerAdapter:
    """Test suite for BBTTransformerAdapter."""

    def test_default_config(self):
        adapter = BBTTransformerAdapter()
        assert adapter.input_dim == 256
        assert adapter.vocab_size == 256
        assert adapter.d_model == 64
        assert adapter.n_heads == 4
        assert adapter.n_layers == 2
        assert adapter.seq_len == 32

    def test_init_params(self):
        adapter = BBTTransformerAdapter()
        params = adapter.init_params(seed=42)

        # 1 token embedding + n_layers*8 block params + 1 final norm
        assert len(params) == 1 + (adapter.n_layers * 8) + 1

        from tinygrad import Tensor

        for p in params:
            assert isinstance(p, Tensor)

    def test_sample_batch(self):
        adapter = BBTTransformerAdapter()
        x, y = adapter.sample_batch(batch_size=16, seed=42)

        assert x.shape == (16, 32)
        assert y.shape == (16, 31)
        assert x.min().item() >= 0
        assert x.max().item() < 256
        assert y.min().item() >= 0
        assert y.max().item() < 256

    def test_forward(self):
        adapter = BBTTransformerAdapter()
        params = adapter.init_params(seed=42)
        x, _ = adapter.sample_batch(batch_size=16, seed=42)

        logits = adapter.forward(params, x, ternarize=False)
        assert logits.shape == (16, 31, 256)

    def test_loss(self):
        adapter = BBTTransformerAdapter()
        params = adapter.init_params(seed=42)
        x, y = adapter.sample_batch(batch_size=16, seed=42)

        logits = adapter.forward(params, x, ternarize=False)
        loss = adapter.loss(logits, y)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_custom_config(self):
        adapter = BBTTransformerAdapter(
            input_dim=128,
            d_model=64,
            n_heads=4,
            n_layers=3,
            d_ff=128,
            seq_len=40,
            tie_weights=False,
        )

        params = adapter.init_params(seed=123)
        x, y = adapter.sample_batch(batch_size=8, seed=123)
        logits = adapter.forward(params, x, ternarize=False)
        loss = adapter.loss(logits, y)

        assert x.shape == (8, 40)
        assert y.shape == (8, 39)
        assert logits.shape == (8, 39, 128)
        assert loss.item() >= 0

    def test_batch_size_variations(self):
        adapter = BBTTransformerAdapter()

        for batch_size in [1, 8, 16]:
            x, y = adapter.sample_batch(batch_size=batch_size, seed=42)
            params = adapter.init_params(seed=42)
            logits = adapter.forward(params, x, ternarize=False)
            loss = adapter.loss(logits, y)

            assert x.shape[0] == batch_size
            assert logits.shape[0] == batch_size
            assert loss.item() >= 0

    def test_reproducibility(self):
        adapter = BBTTransformerAdapter()

        params1 = adapter.init_params(seed=42)
        x1, y1 = adapter.sample_batch(batch_size=4, seed=42)
        logits1 = adapter.forward(params1, x1, ternarize=False)
        loss1 = adapter.loss(logits1, y1)

        params2 = adapter.init_params(seed=42)
        x2, y2 = adapter.sample_batch(batch_size=4, seed=42)
        logits2 = adapter.forward(params2, x2, ternarize=False)
        loss2 = adapter.loss(logits2, y2)

        assert abs(loss1.item() - loss2.item()) < 1e-6

    def test_ternarization_function(self):
        """BitLinear quantization should produce scaled ternary weights."""
        adapter = BBTTransformerAdapter()
        params = adapter.init_params(seed=42)
        tern_params = adapter.ternarize_params(params)

        # Check only BitLinear weight slots that are quantized.
        quantized_idxs = []
        idx = 1  # skip embedding
        for _ in range(adapter.n_layers):
            quantized_idxs.extend([idx, idx + 1, idx + 2, idx + 3, idx + 5, idx + 7])
            idx += 8
        if not adapter.tie_weights:
            quantized_idxs.append(len(params) - 1)

        for qi in quantized_idxs:
            p, tp = params[qi], tern_params[qi]
            gamma = p.abs().mean().maximum(adapter.eps)
            scaled = tp / gamma
            is_discrete = ((scaled - scaled.round()).abs() < 1e-5).all().item()
            in_range = (scaled.abs() <= 1.00001).all().item()
            assert is_discrete and in_range

    def test_legacy_threshold_ternarize_tensor(self):
        adapter = BBTTransformerAdapter()
        params = adapter.init_params(seed=42)
        t = params[1]
        tt = adapter.ternarize_tensor(t, threshold=0.2)
        is_ternary = ((tt == -1.0) + (tt == 0.0) + (tt == 1.0) > 0).all().item()
        assert is_ternary

    def test_forward_with_ternarize_flag(self):
        adapter = BBTTransformerAdapter()
        params = adapter.init_params(seed=42)
        x, _ = adapter.sample_batch(batch_size=8, seed=42)
        logits = adapter.forward(params, x, ternarize=True)
        assert logits.shape == (8, 31, 256)

    def test_sample_batch_uses_local_wikitext_file(self, tmp_path: Path):
        text = "The quick brown fox jumps over the lazy dog.\n" * 32
        corpus_file = tmp_path / "train.txt"
        corpus_file.write_text(text, encoding="utf-8")

        adapter = BBTTransformerAdapter(wikitext_path=str(corpus_file), use_wikitext=True)
        x, y = adapter.sample_batch(batch_size=4, seed=123)

        assert x.shape == (4, adapter.seq_len)
        assert y.shape == (4, adapter.seq_len - 1)
        # Next-token target for each sampled sequence.
        assert (x[:, 1:] == y).all().item()

    def test_sample_batch_wikitext_is_seed_deterministic(self, tmp_path: Path):
        text = "WikiText sample line.\n" * 64
        corpus_file = tmp_path / "train.txt"
        corpus_file.write_text(text, encoding="utf-8")

        adapter = BBTTransformerAdapter(wikitext_path=str(corpus_file), use_wikitext=True)
        x1, y1 = adapter.sample_batch(batch_size=4, seed=11)
        x2, y2 = adapter.sample_batch(batch_size=4, seed=11)

        assert (x1 == x2).all().item()
        assert (y1 == y2).all().item()

    def test_sample_batch_uses_shard_glob(self, tmp_path: Path):
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_file = shard_dir / "shard_00000.bin"
        shard_file.write_bytes(bytes([i % 256 for i in range(4096)]))

        adapter = BBTTransformerAdapter(shard_glob=str(shard_dir / "shard_*.bin"), use_wikitext=True)
        x1, y1 = adapter.sample_batch(batch_size=4, seed=9)
        x2, y2 = adapter.sample_batch(batch_size=4, seed=9)

        assert x1.shape == (4, adapter.seq_len)
        assert y1.shape == (4, adapter.seq_len - 1)
        assert (x1[:, 1:] == y1).all().item()
        assert (x1 == x2).all().item()
        assert (y1 == y2).all().item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
