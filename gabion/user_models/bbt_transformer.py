"""
BitByte Transformer adapter aligned with D:\\code\\bbt architecture.

This implementation follows the core BitByte design:
- byte token embedding
- pre-norm transformer blocks with RMSNorm
- causal multi-head attention with RoPE
- SwiGLU MLP
- BitLinear-style quantization (activation quant + ternary scaled weights)
- language modeling objective (next-byte prediction with CE loss)
"""

from __future__ import annotations

import math
import os
import random
from glob import glob
from pathlib import Path


class BBTTransformerAdapter:
    def __init__(
        self,
        input_dim: int = 256,  # kept for backward compatibility; treated as vocab size
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        seq_len: int = 32,
        d_ff: int | None = None,
        act_quant: bool = True,
        rope_base: float = 10000.0,
        tie_weights: bool = True,
        use_wikitext: bool = True,
        wikitext_path: str | None = None,
        wikitext_split: str = "train",
        shard_glob: str | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.vocab_size = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.d_ff = d_ff if d_ff is not None else d_model * 4
        self.act_quant = act_quant
        self.rope_base = rope_base
        self.tie_weights = tie_weights
        self.use_wikitext = use_wikitext
        self.wikitext_path = wikitext_path
        self.wikitext_split = wikitext_split
        self.shard_glob = shard_glob
        self.eps = 1e-6
        self._cached_corpus: list[int] | None = None
        self._cached_shards: list[Path] | None = None
        self._cached_shard_sizes: list[int] | None = None

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        # Precompute RoPE inverse frequencies.
        from tinygrad import Tensor

        freq_idx = Tensor.arange(0, self.head_dim, 2).float()
        self._inv_freq = Tensor.exp(-(math.log(self.rope_base) / self.head_dim) * freq_idx).realize()

    def init_params(self, seed: int) -> list:
        from tinygrad import Tensor

        Tensor.manual_seed(seed)

        xavier = math.sqrt(2.0 / (self.d_model + self.d_model))
        emb_std = 1.0 / math.sqrt(self.d_model)

        params = []
        tok_emb = Tensor.normal(self.vocab_size, self.d_model, mean=0.0, std=emb_std)
        params.append(tok_emb)

        for _ in range(self.n_layers):
            q_w = Tensor.uniform(self.d_model, self.d_model, low=-xavier, high=xavier)
            k_w = Tensor.uniform(self.d_model, self.d_model, low=-xavier, high=xavier)
            v_w = Tensor.uniform(self.d_model, self.d_model, low=-xavier, high=xavier)
            o_w = Tensor.uniform(self.d_model, self.d_model, low=-xavier, high=xavier)
            n1_w = Tensor.ones(self.d_model)
            gate_up_w = Tensor.uniform(self.d_model, 2 * self.d_ff, low=-xavier, high=xavier)
            n2_w = Tensor.ones(self.d_model)
            down_w = Tensor.uniform(self.d_ff, self.d_model, low=-xavier, high=xavier)
            params.extend([q_w, k_w, v_w, o_w, n1_w, gate_up_w, n2_w, down_w])

        norm_f_w = Tensor.ones(self.d_model)
        params.append(norm_f_w)

        if not self.tie_weights:
            lm_head_w = Tensor.normal(self.d_model, self.vocab_size, mean=0.0, std=emb_std)
            params.append(lm_head_w)

        return params

    def sample_batch(self, batch_size: int, seed: int) -> tuple:
        from tinygrad import Tensor

        total_len = self.seq_len + 1
        if self.use_wikitext:
            shard_samples = self._sample_from_shards(batch_size=batch_size, total_len=total_len, seed=seed)
            if shard_samples is not None:
                batch = Tensor(shard_samples)
                x = batch[:, :-1]
                y = x[:, 1:]
                return x, y

            corpus = self._load_wikitext_corpus()
            if len(corpus) >= total_len:
                rng = random.Random(seed)
                samples: list[list[int]] = []
                for _ in range(batch_size):
                    start = rng.randint(0, len(corpus) - total_len)
                    sample = corpus[start : start + total_len]
                    samples.append(sample)
                batch = Tensor(samples)
                x = batch[:, :-1]
                y = x[:, 1:]
                return x, y

        Tensor.manual_seed(seed)
        x = Tensor.randint(batch_size, self.seq_len, low=0, high=self.vocab_size)
        y = x[:, 1:]
        return x, y

    def _sample_from_shards(self, batch_size: int, total_len: int, seed: int) -> list[list[int]] | None:
        shards = self._load_shard_metadata()
        if not shards:
            return None
        shard_paths, shard_sizes = shards
        if not shard_paths:
            return None

        rng = random.Random(seed)
        samples: list[list[int]] = []
        max_tries = batch_size * 8
        tries = 0
        while len(samples) < batch_size and tries < max_tries:
            tries += 1
            shard_idx = rng.randint(0, len(shard_paths) - 1)
            shard_path = shard_paths[shard_idx]
            shard_size = shard_sizes[shard_idx]
            if shard_size < total_len:
                continue
            start = rng.randint(0, shard_size - total_len)
            with shard_path.open("rb") as f:
                f.seek(start)
                chunk = f.read(total_len)
            if len(chunk) != total_len:
                continue
            samples.append([int(b) % self.vocab_size for b in chunk])

        return samples if len(samples) == batch_size else None

    def _load_wikitext_corpus(self) -> list[int]:
        if self._cached_corpus is not None:
            return self._cached_corpus

        text = self._load_wikitext_text()
        if not text:
            self._cached_corpus = []
            return self._cached_corpus

        # Byte-level LM objective: clamp to vocab size to stay index-safe.
        self._cached_corpus = [b % self.vocab_size for b in text.encode("utf-8")]
        return self._cached_corpus

    def _load_wikitext_text(self) -> str:
        if self.wikitext_path:
            path = Path(self.wikitext_path)
            if path.exists():
                return path.read_text(encoding="utf-8", errors="ignore")

        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=self.wikitext_split)
            lines = [str(row.get("text", "")) for row in ds if str(row.get("text", "")).strip()]
            if lines:
                return "\n".join(lines)
        except Exception:
            pass

        fallback = self._repo_root() / ".deps" / "datasets" / "wikitext-2" / f"{self.wikitext_split}.txt"
        if fallback.exists():
            return fallback.read_text(encoding="utf-8", errors="ignore")

        return ""

    def _load_shard_metadata(self) -> tuple[list[Path], list[int]] | None:
        if self._cached_shards is not None and self._cached_shard_sizes is not None:
            return self._cached_shards, self._cached_shard_sizes

        matched: list[Path] = []
        for pattern in self._candidate_shard_globs():
            files = [Path(p) for p in sorted(glob(pattern))]
            if files:
                matched = files
                break
        if not matched:
            self._cached_shards = []
            self._cached_shard_sizes = []
            return self._cached_shards, self._cached_shard_sizes

        sizes = [os.path.getsize(p) for p in matched]
        self._cached_shards = matched
        self._cached_shard_sizes = sizes
        return self._cached_shards, self._cached_shard_sizes

    def _candidate_shard_globs(self) -> list[str]:
        candidates: list[str] = []
        if self.shard_glob:
            candidates.append(self.shard_glob)
        env_glob = os.environ.get("BBT_SHARD_GLOB")
        if env_glob:
            candidates.append(env_glob)

        repo_root = self._repo_root()
        candidates.extend(
            [
                str(repo_root / ".deps" / "datasets" / "wikitext_103" / "shards" / "train" / "shard_*.bin"),
                str(repo_root.parent / "bbt" / "artifacts" / "datasets" / "wikitext_103" / "shards" / "train" / "shard_*.bin"),
                r"D:\code\bbt\artifacts\datasets\wikitext_103\shards\train\shard_*.bin",
            ]
        )
        return candidates

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def forward(self, params, x, ternarize: bool = False) -> "Tensor":
        batch_size, seq_len = x.shape

        if ternarize:
            params = self.ternarize_params(params)
            quantized_weights = True
        else:
            quantized_weights = False

        idx = 0
        tok_emb = params[idx]
        idx += 1

        x = tok_emb[x]  # [B, T, C]

        for _ in range(self.n_layers):
            layer_params = params[idx:idx + 8]
            idx += 8
            x = self._block(x, layer_params, quantized_weights=quantized_weights)

        norm_f_w = params[idx]
        idx += 1
        x = self._rms_norm(x, norm_f_w)

        if self.tie_weights:
            logits = x @ tok_emb.transpose(0, 1)
        else:
            lm_head_w = params[idx]
            logits = x @ lm_head_w

        # Predict next token at each position i -> target x[i+1].
        return logits[:, :seq_len - 1, :]

    def loss(self, logits: "Tensor", y: "Tensor") -> "Tensor":
        return logits.sparse_categorical_crossentropy(y)

    def ternarize_params(self, params: list, threshold: float = None) -> list:
        # Threshold kept for API compatibility; BitByte uses abs-mean scaled ternary quantization.
        _ = threshold

        tern = []
        idx = 0

        tern.append(params[idx])  # token embedding (not quantized in BitLinear path)
        idx += 1

        for _ in range(self.n_layers):
            tern.append(self._ternary_quant_bitnet(params[idx])); idx += 1  # q
            tern.append(self._ternary_quant_bitnet(params[idx])); idx += 1  # k
            tern.append(self._ternary_quant_bitnet(params[idx])); idx += 1  # v
            tern.append(self._ternary_quant_bitnet(params[idx])); idx += 1  # o
            tern.append(params[idx]); idx += 1  # n1 weight
            tern.append(self._ternary_quant_bitnet(params[idx])); idx += 1  # gate_up
            tern.append(params[idx]); idx += 1  # n2 weight
            tern.append(self._ternary_quant_bitnet(params[idx])); idx += 1  # down

        tern.append(params[idx])  # final norm weight
        idx += 1

        if not self.tie_weights:
            tern.append(self._ternary_quant_bitnet(params[idx]))

        return tern

    def ternarize_tensor(self, t: "Tensor", threshold: float = 0.5) -> "Tensor":
        # Legacy helper retained for explicit threshold-based ternarization checks.
        return (t.sign() * (t.abs() >= threshold).float()).realize()

    def _block(self, x: "Tensor", layer_params: list, quantized_weights: bool) -> "Tensor":
        q_w, k_w, v_w, o_w, n1_w, gate_up_w, n2_w, down_w = layer_params

        h = self._rms_norm(x, n1_w)
        h = self._causal_self_attention(h, q_w, k_w, v_w, o_w, quantized_weights=quantized_weights)
        x = x + h

        h = self._rms_norm(x, n2_w)
        gate_up = self._bitlinear(h, gate_up_w, quantized_weights=quantized_weights)
        a, b = gate_up.chunk(2, dim=-1)
        h = a.silu() * b
        h = self._bitlinear(h, down_w, quantized_weights=quantized_weights)
        x = x + h
        return x

    def _causal_self_attention(
        self,
        x: "Tensor",
        q_w: "Tensor",
        k_w: "Tensor",
        v_w: "Tensor",
        o_w: "Tensor",
        quantized_weights: bool,
    ) -> "Tensor":
        from tinygrad import Tensor

        bsz, seq_len, _ = x.shape

        q = self._bitlinear(x, q_w, quantized_weights=quantized_weights)
        k = self._bitlinear(x, k_w, quantized_weights=quantized_weights)
        v = self._bitlinear(x, v_w, quantized_weights=quantized_weights)

        q = q.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = k.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = self._apply_rope(q)
        k = self._apply_rope(k)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,T,T]

        # Causal mask: forbid attending to future positions.
        mask = Tensor.ones(seq_len, seq_len).triu(1).reshape(1, 1, seq_len, seq_len)
        scores = scores.masked_fill(mask == 1, float("-inf"))

        attn = scores.softmax(axis=-1)
        y = attn @ v

        y = y.transpose(1, 2).reshape(bsz, seq_len, self.d_model)
        return self._bitlinear(y, o_w, quantized_weights=quantized_weights)

    def _apply_rope(self, x: "Tensor") -> "Tensor":
        # x: [B, H, T, D]
        from tinygrad import Tensor

        seq_len = x.shape[2]
        pos = Tensor.arange(seq_len).float().reshape(seq_len, 1)
        freqs = pos * self._inv_freq.reshape(1, -1)  # [T, D/2]
        emb = freqs.cat(freqs, dim=-1)  # [T, D]
        cos = emb.cos().reshape(1, 1, seq_len, self.head_dim)
        sin = emb.sin().reshape(1, 1, seq_len, self.head_dim)
        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x: "Tensor") -> "Tensor":
        x1, x2 = x.chunk(2, dim=-1)
        return (-x2).cat(x1, dim=-1)

    def _rms_norm(self, x: "Tensor", w: "Tensor") -> "Tensor":
        norm = (x * x).mean(axis=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * w

    def _act_quant_per_token(self, x: "Tensor", q: int = 127) -> "Tensor":
        if not self.act_quant:
            return x
        x2 = x.reshape(-1, x.shape[-1])
        s = x2.abs().max(axis=-1, keepdim=True) / q
        s = s.maximum(self.eps)
        x_scaled = x2 / s
        xq_hard = x_scaled.round().clip(-q, q)
        # STE for activation quantization.
        xq = x_scaled + (xq_hard - x_scaled).detach()
        return (xq * s).reshape(x.shape)

    def _ternary_quant_bitnet(self, w: "Tensor", ste: bool = False) -> "Tensor":
        gamma = w.abs().mean().maximum(self.eps)
        w_scaled = w / gamma
        wq_hard = w_scaled.clip(-1.0, 1.0).round()
        # STE for training: forward uses quantized values, backward uses identity.
        wq = w_scaled + (wq_hard - w_scaled).detach() if ste else wq_hard
        return (wq * gamma).realize()

    def _bitlinear(self, x: "Tensor", w: "Tensor", quantized_weights: bool) -> "Tensor":
        x = self._act_quant_per_token(x)
        if quantized_weights:
            wq = w
        else:
            wq = self._ternary_quant_bitnet(w, ste=True)
        return x @ wq
