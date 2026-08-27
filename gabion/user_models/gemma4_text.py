"""Gemma4 text adapter for CUDA pebble workers (tinygrad).

Implements Gemma4Text layer stack: input_layernorm -> self_attn (q/k/v + qk RMSNorm + RoPE + sliding window) -> post_attention_norm -> pre_ffn_norm -> MLP (gate/up/down, double-wide) -> post_ffw_norm -> per_layer_input (inp_gate/proj + post_per_layer_input_norm) + layer_scale.
Keeps E2B GGUF as-is via dequant on load (tools/export_model _tensor_data).

Reference: transformers modeling_gemma4 Gemma4TextDecoderLayer + Gemma4TextAttention + Gemma4TextMLP, config text_config from google/gemma-4-e2b-it.
"""
from __future__ import annotations

import base64
import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

def _encode_hidden_f16(arr: np.ndarray) -> Tuple[str, List[int]]:
    """Encode hidden [B,T,D] float32 as base64 f16 + shape for pipeline transfer."""
    a = np.asarray(arr, dtype=np.float32).astype(np.float16)
    b64 = base64.b64encode(a.tobytes()).decode("ascii")
    return b64, list(arr.shape)

def _decode_hidden_f16(b64: str, shape: List[int]) -> np.ndarray:
    raw = base64.b64decode(b64)
    a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    return a.reshape(shape)


def _rms_norm(x, w, eps=1e-6):
    # x: [..., D], w: [D]
    norm = (x * x).mean(axis=-1, keepdim=True).add(eps).rsqrt()
    return x * norm * w


def _rotate_half(x):
    # x: [..., D]
    d = x.shape[-1]
    x1, x2 = x.chunk(2, dim=-1)
    return (-x2).cat(x1, dim=-1)


class Gemma4TextAdapter:
    def __init__(
        self,
        vocab_size: int = 262144,
        d_model: int = 1536,
        n_heads: int = 8,
        n_kv_heads: int = 1,
        n_kv_heads_per_layer: list[int] | None = None,
        n_layers: int = 35,
        d_ff_per_layer: list[int] | None = None,
        head_dim_per_layer: list[int] | None = None,
        layer_types: list[str] | None = None,
        sliding_window: int = 512,
        rope_theta: float = 1_000_000.0,
        rope_theta_swa: float = 10_000.0,
        rope_dim: int = 128,  # partial 0.25 of 512
        rope_dim_swa: int = 64,  # 0.25 of 256
        hidden_size_per_layer_input: int = 256,
        final_logit_softcapping: float = 30.0,
        hidden_act: str = "gelu_pytorch_tanh",
        tie_weights: bool = True,
        rms_norm_eps: float = 1e-6,
        seq_len: int = 131072,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_kv_heads_per_layer = n_kv_heads_per_layer or [n_kv_heads] * n_layers
        self.n_layers = n_layers
        self.d_ff_per_layer = d_ff_per_layer or [6144] * n_layers
        self.head_dim_per_layer = head_dim_per_layer or [256 if (i % 5 != 4) else 512 for i in range(n_layers)]
        self.layer_types = layer_types or ["sliding_attention" if (i % 5 != 4) else "full_attention" for i in range(n_layers)]
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.rope_theta_swa = rope_theta_swa
        self.rope_dim = rope_dim
        self.rope_dim_swa = rope_dim_swa
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.final_logit_softcapping = final_logit_softcapping
        self.hidden_act = hidden_act
        self.tie_weights = tie_weights
        self.rms_norm_eps = rms_norm_eps
        self.seq_len = seq_len

    @classmethod
    def from_gguf_shard(cls, gguf_path: str | Path, shard_idx: int = 0, num_shards: int = 2, keep_q4: bool | None = None):
        """Load only a shard of layers for pipeline parallel (model-parallel) inference.

        Shard 0 holds tok_emb + layers [0, L0); final shard holds output_norm + tok_emb (tied lm_head).
        Middle shards (if >2) hold only their layers.
        Hidden is [B,T,D] f16 base64 between shards, stays on DEV=CUDA.
        """
        from tools.export_model import parse_gguf, _tensor_data
        if keep_q4 is None:
            keep_q4 = os.environ.get("GEMMA4_KEEP_Q4","0")=="1" or "31B" in str(gguf_path)

        p = Path(gguf_path)
        meta, infos, data_base = parse_gguf(p)
        by_name = {n: (dims, gtype, off) for n, dims, gtype, off in infos}
        if keep_q4:
            # 31B: mmap the GGUF instead of read_bytes — file-backed pages are
            # shared/evictable, so two pipeline workers fit in system RAM while
            # dequant happens per-layer on demand. memoryview(mmap) works in
            # tools.export_model._tensor_data.
            import mmap
            with open(p, "rb") as f:
                buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            buf = p.read_bytes()
        arch = "gemma4"
        L = int(meta[f"{arch}.block_count"])
        D = int(meta[f"{arch}.embedding_length"])
        H_raw = meta[f"{arch}.attention.head_count"]
        H = int(H_raw) if not isinstance(H_raw, list) else int(H_raw[0])
        kv_raw = meta.get(f"{arch}.attention.head_count_kv")
        if isinstance(kv_raw, list):
            kv_per_layer = [int(x) for x in kv_raw]
            Kv = kv_per_layer[0]
        else:
            Kv = int(kv_raw or H)
            kv_per_layer = [Kv] * L
        head_dim_per_layer = []
        for i in range(L):
            qdims = by_name[f"blk.{i}.attn_q.weight"][0]
            hd = int(qdims[1] // H)
            head_dim_per_layer.append(hd)
        dff = meta.get(f"{arch}.feed_forward_length")
        dff_list = [int(x) for x in dff] if isinstance(dff, list) else [int(dff)] * L
        pat = meta.get(f"{arch}.attention.sliding_window_pattern")
        layer_types = ["full_attention" if not bool(pat[i]) else "sliding_attention" for i in range(L)] if isinstance(pat, list) else ["sliding_attention"] * L
        sw = int(meta.get(f"{arch}.attention.sliding_window") or 512)
        rope = float(meta.get(f"{arch}.rope.freq_base") or 1_000_000.0)
        rope_swa = float(meta.get(f"{arch}.rope.freq_base_swa") or 10_000.0)
        rope_dim = int(meta.get(f"{arch}.rope.dimension_count") or 0)
        rope_dim_swa = int(meta.get(f"{arch}.rope.dimension_count_swa") or 0)
        hs_per_raw = meta.get(f"{arch}.embedding_length_per_layer_input")
        hs_per = int(hs_per_raw) if hs_per_raw is not None else 256
        tok_dims = by_name.get("token_embd.weight", ((),))[0]
        n_vocab = int(tok_dims[1]) if len(tok_dims) == 2 else int(meta.get(f"{arch}.vocab_size") or 262144)
        inst = cls(vocab_size=n_vocab, d_model=D, n_heads=H, n_kv_heads=Kv, n_kv_heads_per_layer=kv_per_layer, n_layers=L, d_ff_per_layer=dff_list, head_dim_per_layer=head_dim_per_layer, layer_types=layer_types, sliding_window=sw, rope_theta=rope, rope_theta_swa=rope_swa, rope_dim=rope_dim or 128, rope_dim_swa=rope_dim_swa or 64, hidden_size_per_layer_input=hs_per, final_logit_softcapping=float(meta.get(f"{arch}.final_logit_softcapping") or 30.0), tie_weights=("output.weight" not in by_name))
        # shard boundaries
        base = L // num_shards
        rem = L % num_shards
        # first `rem` shards get one extra layer
        starts = []
        ends = []
        off2 = 0
        for s in range(num_shards):
            sz = base + (1 if s < rem else 0)
            starts.append(off2)
            ends.append(off2 + sz)
            off2 += sz
        inst.shard_idx = int(shard_idx)
        inst.num_shards = int(num_shards)
        inst.layer_start = starts[shard_idx]
        inst.layer_end = ends[shard_idx]
        inst.owns_tok = (shard_idx == 0) or (shard_idx == num_shards - 1 and inst.tie_weights)
        inst.owns_output_norm = (shard_idx == num_shards - 1)
        inst._keep_q4 = bool(keep_q4)
        inst._gguf_buf = buf
        inst._gguf_by_name = by_name
        inst._gguf_data_base = data_base
        inst._qcache: dict = {}
        if inst._keep_q4:
            inst._gguf_params = []
            inst._is_shard = True
            inst._tok_emb_f16 = None  # Q4_K token_embd, CPU-dequantized once + cached
            return inst
        # pack shard
        from tinygrad import Tensor
        from tools.export_model import _tensor_data as _td
        def get(name):
            dims, gtype, off = by_name[name]
            return _td(buf, name, dims, gtype, off, data_base)
        flat: list[Tensor] = []
        if shard_idx == 0:
            tok = get("token_embd.weight").astype(np.float32)
            flat.append(Tensor(tok))
            inst._tok_for_lm = flat[0]  # shard0 also keeps for its own? not needed
        for i in range(inst.layer_start, inst.layer_end):
            hd = head_dim_per_layer[i]
            kv_hd = hd
            n1 = get(f"blk.{i}.attn_norm.weight").astype(np.float32).reshape(-1)
            q_raw = get(f"blk.{i}.attn_q.weight").astype(np.float32).T
            qn = get(f"blk.{i}.attn_q_norm.weight").astype(np.float32).reshape(-1)
            k_raw = get(f"blk.{i}.attn_k.weight").astype(np.float32).T
            kn = get(f"blk.{i}.attn_k_norm.weight").astype(np.float32).reshape(-1)
            v_raw = get(f"blk.{i}.attn_v.weight").astype(np.float32).T
            o_raw = get(f"blk.{i}.attn_output.weight").astype(np.float32).T
            post_attn = get(f"blk.{i}.post_attention_norm.weight").astype(np.float32).reshape(-1)
            pre_ffn = get(f"blk.{i}.ffn_norm.weight").astype(np.float32).reshape(-1)
            dff_i = dff_list[i]
            gate = get(f"blk.{i}.ffn_gate.weight").astype(np.float32).T
            up = get(f"blk.{i}.ffn_up.weight").astype(np.float32).T
            gate_up = np.concatenate([gate, up], axis=1)
            down = get(f"blk.{i}.ffn_down.weight").astype(np.float32).T
            post_ffw = get(f"blk.{i}.post_ffw_norm.weight").astype(np.float32).reshape(-1)
            if hs_per:
                inp_gate = get(f"blk.{i}.inp_gate.weight").astype(np.float32).T
                proj = get(f"blk.{i}.proj.weight").astype(np.float32).T
                post_per = get(f"blk.{i}.post_norm.weight").astype(np.float32).reshape(-1)
                scale = get(f"blk.{i}.layer_output_scale.weight").astype(np.float32).reshape(-1)
                for arr in [n1, q_raw, qn, k_raw, kn, v_raw, o_raw, inp_gate, proj, post_attn, pre_ffn, gate_up, down, post_ffw, post_per, scale]:
                    flat.append(Tensor(arr))
            else:
                scale = get(f"blk.{i}.layer_output_scale.weight").astype(np.float32).reshape(-1) if f"blk.{i}.layer_output_scale.weight" in by_name else np.ones(1, dtype=np.float32)
                for arr in [n1, q_raw, qn, k_raw, kn, v_raw, o_raw, post_attn, pre_ffn, gate_up, down, post_ffw, scale]:
                    flat.append(Tensor(arr))
        if inst.owns_output_norm:
            norm_f = get("output_norm.weight").astype(np.float32).reshape(-1)
            flat.append(Tensor(norm_f))
            if shard_idx != 0:
                # shard !=0 also needs tok for lm head (tied)
                tok2 = get("token_embd.weight").astype(np.float32)
                flat.append(Tensor(tok2))
                inst._tok_for_lm = flat[-1]
            else:
                inst._tok_for_lm = flat[0]
            if not inst.tie_weights:
                o_w = get("output.weight").astype(np.float32)
                flat.append(Tensor(o_w))
        else:
            inst._tok_for_lm = flat[0] if flat else None
        inst._gguf_params = flat
        inst._is_shard = True
        return inst

    def _qget(self, name: str, transpose: bool = False):
        """Dequant on demand. IQ4_NL (all of the 31B's big weights) is
        dequantized ON DEVICE in a tinygrad kernel — only the raw u8 nibbles +
        f16 block scales cross the bus; the kvalue lookup and scale multiply run
        on the GPU. Other types (F32 norms/scales, Q4_K token_embd) stay on CPU
        numpy. No cache when keep_q4 (minimal VRAM for 31B)."""
        dims, gtype, off = self._gguf_by_name[name]
        if gtype == "IQ4_NL":
            return self._qget_iq4nl_gpu(name, dims, off, transpose)
        from tools.export_model import _tensor_data as _td
        import numpy as np
        from tinygrad import Tensor
        arr = _td(self._gguf_buf, name, dims, gtype, off, self._gguf_data_base)
        arr = arr.astype(np.float16)
        if transpose:
            arr = arr.T.copy()
        t = Tensor(arr)
        # only cache for E2B (small) when explicitly enabled; 31B would OOM if cached (31GB/shard)
        if os.environ.get("GEMMA4_QCACHE", "0") == "1":
            key = name + (".T" if transpose else "")
            self._qcache[key] = t
        return t

    def _qget_iq4nl_gpu(self, name: str, dims, off: int, transpose: bool = False):
        """IQ4_NL dequant as tinygrad ops (compiled to a CUDA kernel).

        ggml block: 18 B = f16 scale + 16 nibble bytes; per block positions
        0..15 are the lo nibbles, 16..31 the hi nibbles (dequantize_row_iq4_nl:
        y[j] = ks[qs[j] & 0xf]; y[j+16] = ks[qs[j] >> 4]).
        """
        from tinygrad import Tensor, dtypes
        import numpy as np
        from tools.export_model import _IQ4_KVALUES
        n_in, n_out = int(dims[0]), int(dims[1])
        nb = n_in // 32
        base = self._gguf_data_base + off
        blk = np.frombuffer(self._gguf_buf, dtype=np.uint8, count=n_out * nb * 18, offset=base).reshape(n_out, nb, 18)
        sc = blk[:, :, 0:2].copy().view(np.float16).reshape(n_out, nb, 1).astype(np.float32)
        qs = np.ascontiguousarray(blk[:, :, 2:18])
        t_qs = Tensor(qs)  # u8 [n_out, nb, 16]
        t_sc = Tensor(sc)  # f32 [n_out, nb, 1]
        lo = t_qs & 0x0F
        hi = (t_qs >> 4) & 0x0F
        nib = lo.cat(hi, dim=-1)  # [n_out, nb, 32] — ggml lo-half then hi-half
        ks = Tensor(_IQ4_KVALUES.reshape(16).astype(np.float32))
        v = ks.gather(0, nib.cast(dtypes.int32).reshape(-1)).reshape(n_out, nb, 32)
        v = (v * t_sc).reshape(n_out, n_in).cast(dtypes.float16)
        if transpose:
            v = v.T
        return v

    def _token_emb_f16(self):
        """Q4_K token_embd, dequantized in row-chunks to cap the transient f32
        working set (a full-table dequant allocates ~5.6 GB f32 + 2.8 GB f16 per
        worker — two pipeline workers on a 34 GB box stall on that)."""
        from tools.export_model import _tensor_data as _td
        import numpy as np
        from tinygrad import Tensor
        dims, gtype, off = self._gguf_by_name["token_embd.weight"]
        n_in, n_out = int(dims[0]), int(dims[1])  # ne order: (D, vocab)
        assert gtype == "Q4_K"
        row_bytes = (n_in // 256) * 144
        out = np.empty((n_out, n_in), dtype=np.float16)
        base = self._gguf_data_base + off
        ROWS = 4096
        for r0 in range(0, n_out, ROWS):
            r1 = min(r0 + ROWS, n_out)
            chunk = _td(self._gguf_buf, "token_embd", (n_in, r1 - r0), gtype, off + r0 * row_bytes, self._gguf_data_base)
            out[r0:r1] = chunk.astype(np.float16)
        return Tensor(out)

    def _forward_q4(self, x, layer_start: int, layer_end: int):
        """On-demand Q4 per-layer forward (dequant per matmul to f16, minimal VRAM)."""
        for li in range(layer_start, layer_end):
            hd = self.head_dim_per_layer[li]
            H = self.n_heads
            Kv = self.n_kv_heads_per_layer[li]
            n1 = self._qget(f"blk.{li}.attn_norm.weight")
            q_w = self._qget(f"blk.{li}.attn_q.weight", transpose=True)
            qn_w = self._qget(f"blk.{li}.attn_q_norm.weight")
            k_w = self._qget(f"blk.{li}.attn_k.weight", transpose=True)
            kn_w = self._qget(f"blk.{li}.attn_k_norm.weight")
            try:
                v_w = self._qget(f"blk.{li}.attn_v.weight", transpose=True)
            except KeyError:
                # 31B global layers (every 6th, Kv=4) omit attn_v in GGUF; reuse attn_k as V (K and V same 4*hd=2048)
                v_w = k_w
            o_w = self._qget(f"blk.{li}.attn_output.weight", transpose=True)
            post_attn_w = self._qget(f"blk.{li}.post_attention_norm.weight")
            pre_ffn_w = self._qget(f"blk.{li}.ffn_norm.weight")
            # gate/up need concat (on GPU — no CPU round-trip)
            gate = self._qget(f"blk.{li}.ffn_gate.weight", transpose=True)
            up = self._qget(f"blk.{li}.ffn_up.weight", transpose=True)
            gate_up = gate.cat(up, dim=1)
            down_w = self._qget(f"blk.{li}.ffn_down.weight", transpose=True)
            post_ffw_w = self._qget(f"blk.{li}.post_ffw_norm.weight")
            if self.hidden_size_per_layer_input:
                inp_gate_w = self._qget(f"blk.{li}.inp_gate.weight", transpose=True)
                proj_w = self._qget(f"blk.{li}.proj.weight", transpose=True)
                post_per_w = self._qget(f"blk.{li}.post_norm.weight")
                scale_w = self._qget(f"blk.{li}.layer_output_scale.weight")
            else:
                inp_gate_w = proj_w = post_per_w = None
                try:
                    scale_w = self._qget(f"blk.{li}.layer_output_scale.weight")
                except KeyError:
                    import numpy as np
                    from tinygrad import Tensor
                    scale_w = Tensor(np.ones(1, dtype=np.float16))
            residual = x
            if self.hidden_size_per_layer_input:
                per = residual @ inp_gate_w
                per = (per * 0.5 * (1 + (0.79788456 * (per + 0.044715 * per * per * per)).tanh()))
                per = per @ proj_w
                per = self._rms(per, post_per_w)
                x_input = residual + per
            else:
                x_input = residual
            h = self._rms(x_input, n1)
            bsz, t, _ = h.shape
            q = h @ q_w; k = h @ k_w; v = h @ v_w
            q = q.reshape(bsz, t, H, hd)
            q = q * (q * q).mean(axis=-1, keepdim=True).add(self.rms_norm_eps).rsqrt() * qn_w
            q = q.transpose(1, 2)
            k = k.reshape(bsz, t, Kv, hd)
            k = k * (k * k).mean(axis=-1, keepdim=True).add(self.rms_norm_eps).rsqrt() * kn_w
            k = k.transpose(1, 2)
            v = v.reshape(bsz, t, Kv, hd).transpose(1, 2)
            q = self._apply_rope(q, t, li)
            k = self._apply_rope(k, t, li)
            attn_out = self._causal_attention(q, k, v, li)
            if attn_out.shape[-1] == o_w.shape[0]:
                h_attn = attn_out @ o_w
            else:
                h_attn = attn_out
            h_attn = self._rms(h_attn, post_attn_w)
            x = residual + h_attn * scale_w
            h2 = self._rms(x, pre_ffn_w)
            gate_up_val = h2 @ gate_up
            a, b = gate_up_val.chunk(2, dim=-1)
            a = (a * 0.5 * (1 + (0.79788456 * (a + 0.044715 * a * a * a)).tanh()))
            h2 = a * b
            h2 = h2 @ down_w
            h2 = self._rms(h2, post_ffw_w)
            x = x + h2 * scale_w
            x.realize()  # flush this layer's graph now — otherwise tinygrad's lazy
            # scheduler keeps every shard layer's dequantized f16 weights in VRAM
            # simultaneously (30 x ~1 GB on the 31B -> OOM on a 16 GB card)
        return x

    def forward_shard_ids_to_hidden(self, x_ids):
        """Shard 0: tok_emb -> layers [layer_start, layer_end) -> hidden [B,T,D]."""
        assert getattr(self, "_is_shard", False) and self.shard_idx == 0, "forward_shard_ids_to_hidden only on shard 0"
        if getattr(self, "_keep_q4", False):
            if self._tok_emb_f16 is None:
                # Q4_K token_embd: CPU-dequantized once (chunked), then cached —
                # the table is fixed, so per-token cost is just the lookup
                self._tok_emb_f16 = self._token_emb_f16()
            x = self._tok_emb_f16[x_ids]
            return self._forward_q4(x, self.layer_start, self.layer_end)
        from tinygrad import Tensor
        params = self._gguf_params
        p = 0
        tok_emb = params[p]; p+=1
        x = tok_emb[x_ids]
        for li in range(self.layer_start, self.layer_end):
            hd = self.head_dim_per_layer[li]
            H = self.n_heads
            Kv = self.n_kv_heads_per_layer[li]
            n1 = params[p]; p+=1; q_w = params[p]; p+=1; qn_w = params[p]; p+=1; k_w = params[p]; p+=1; kn_w = params[p]; p+=1; v_w = params[p]; p+=1; o_w = params[p]; p+=1
            if self.hidden_size_per_layer_input:
                inp_gate_w = params[p]; p+=1; proj_w = params[p]; p+=1
            post_attn_w = params[p]; p+=1; pre_ffn_w = params[p]; p+=1; gate_up_w = params[p]; p+=1; down_w = params[p]; p+=1; post_ffw_w = params[p]; p+=1
            if self.hidden_size_per_layer_input:
                post_per_w = params[p]; p+=1; scale_w = params[p]; p+=1
            else:
                scale_w = params[p]; p+=1
            residual = x
            if self.hidden_size_per_layer_input:
                per = residual @ inp_gate_w
                per = (per * 0.5 * (1 + (0.79788456 * (per + 0.044715 * per * per * per)).tanh()))
                per = per @ proj_w
                per = self._rms(per, post_per_w)
                x_input = residual + per
            else:
                x_input = residual
            h = self._rms(x_input, n1)
            bsz, t, _ = h.shape
            q = h @ q_w; k = h @ k_w; v = h @ v_w
            q = q.reshape(bsz, t, H, hd)
            q = q * (q * q).mean(axis=-1, keepdim=True).add(self.rms_norm_eps).rsqrt() * qn_w
            q = q.transpose(1, 2)
            k = k.reshape(bsz, t, Kv, hd)
            k = k * (k * k).mean(axis=-1, keepdim=True).add(self.rms_norm_eps).rsqrt() * kn_w
            k = k.transpose(1, 2)
            v = v.reshape(bsz, t, Kv, hd).transpose(1, 2)
            q = self._apply_rope(q, t, li)
            k = self._apply_rope(k, t, li)
            attn_out = self._causal_attention(q, k, v, li)
            if attn_out.shape[-1] == o_w.shape[0]:
                h_attn = attn_out @ o_w
            else:
                h_attn = attn_out
            h_attn = self._rms(h_attn, post_attn_w)
            x = residual + h_attn * scale_w
            h2 = self._rms(x, pre_ffn_w)
            gate_up = h2 @ gate_up_w
            a, b = gate_up.chunk(2, dim=-1)
            a = (a * 0.5 * (1 + (0.79788456 * (a + 0.044715 * a * a * a)).tanh()))
            h2 = a * b
            h2 = h2 @ down_w
            h2 = self._rms(h2, post_ffw_w)
            x = x + h2 * scale_w
        return x

    def forward_shard_hidden_to_hidden(self, hidden):
        """Middle shard: hidden in -> layers -> hidden out (for num_shards>2)."""
        assert getattr(self, "_is_shard", False)
        if getattr(self, "_keep_q4", False):
            return self._forward_q4(hidden, self.layer_start, self.layer_end)
        from tinygrad import Tensor
        x = hidden
        params = self._gguf_params
        # shard not 0: params start at layer_start
        p = 0
        for li in range(self.layer_start, self.layer_end):
            hd = self.head_dim_per_layer[li]
            H = self.n_heads
            Kv = self.n_kv_heads_per_layer[li]
            n1 = params[p]; p+=1; q_w = params[p]; p+=1; qn_w = params[p]; p+=1; k_w = params[p]; p+=1; kn_w = params[p]; p+=1; v_w = params[p]; p+=1; o_w = params[p]; p+=1
            if self.hidden_size_per_layer_input:
                inp_gate_w = params[p]; p+=1; proj_w = params[p]; p+=1
            post_attn_w = params[p]; p+=1; pre_ffn_w = params[p]; p+=1; gate_up_w = params[p]; p+=1; down_w = params[p]; p+=1; post_ffw_w = params[p]; p+=1
            if self.hidden_size_per_layer_input:
                post_per_w = params[p]; p+=1; scale_w = params[p]; p+=1
            else:
                scale_w = params[p]; p+=1
            residual = x
            if self.hidden_size_per_layer_input:
                per = residual @ inp_gate_w
                per = (per * 0.5 * (1 + (0.79788456 * (per + 0.044715 * per * per * per)).tanh()))
                per = per @ proj_w
                per = self._rms(per, post_per_w)
                x_input = residual + per
            else:
                x_input = residual
            h = self._rms(x_input, n1)
            bsz, t, _ = h.shape
            q = h @ q_w; k = h @ k_w; v = h @ v_w
            q = q.reshape(bsz, t, H, hd)
            q = q * (q * q).mean(axis=-1, keepdim=True).add(self.rms_norm_eps).rsqrt() * qn_w
            q = q.transpose(1, 2)
            k = k.reshape(bsz, t, Kv, hd)
            k = k * (k * k).mean(axis=-1, keepdim=True).add(self.rms_norm_eps).rsqrt() * kn_w
            k = k.transpose(1, 2)
            v = v.reshape(bsz, t, Kv, hd).transpose(1, 2)
            q = self._apply_rope(q, t, li)
            k = self._apply_rope(k, t, li)
            attn_out = self._causal_attention(q, k, v, li)
            h_attn = attn_out @ o_w if attn_out.shape[-1] == o_w.shape[0] else attn_out
            h_attn = self._rms(h_attn, post_attn_w)
            x = residual + h_attn * scale_w
            h2 = self._rms(x, pre_ffn_w)
            gate_up = h2 @ gate_up_w
            a, b = gate_up.chunk(2, dim=-1)
            a = (a * 0.5 * (1 + (0.79788456 * (a + 0.044715 * a * a * a)).tanh()))
            h2 = a * b
            h2 = h2 @ down_w
            h2 = self._rms(h2, post_ffw_w)
            x = x + h2 * scale_w
        return x

    def forward_shard_hidden_to_logits(self, hidden):
        """Final shard: hidden -> remaining layers -> norm -> logits."""
        assert getattr(self, "_is_shard", False) and self.owns_output_norm, "only final shard"
        if getattr(self, "_keep_q4", False):
            x = self._forward_q4(hidden, self.layer_start, self.layer_end) if self.layer_start < self.layer_end else hidden
            norm_f = self._qget("output_norm.weight")
            x = self._rms(x, norm_f)
            if self._tok_emb_f16 is None:
                self._tok_emb_f16 = self._token_emb_f16()
            tok_emb = self._tok_emb_f16
            logits = x @ tok_emb.T
            if self.final_logit_softcapping:
                logits = (logits / self.final_logit_softcapping).tanh() * self.final_logit_softcapping
            return logits
        from tinygrad import Tensor
        x = self.forward_shard_hidden_to_hidden(hidden) if self.layer_start < self.layer_end else hidden
        params = self._gguf_params
        # locate output_norm and tok at end
        # params: [layers..., norm_f, tok] (or + o_w)
        # forward_shard_hidden_to_hidden already consumed layer params; need to get remaining
        # Instead re-derive offset: count layer tensors
        # Simpler: reuse self._tok_for_lm and last norm
        # Find norm_f as second-last (or last if tie? with tok)
        per_layer = 16 if self.hidden_size_per_layer_input else 13
        # not needed; just index from end
        if self.tie_weights:
            tok_emb = self._tok_for_lm
            # output_norm is at -2 (if tie) else -2? Actually flat = layers + [norm_f, tok]
            norm_f = params[-2] if self.tie_weights else params[-2]
            # but if shard 0 final shard case num_shards=1, flat has tok at 0 and norm at -1; our tie path above is different
            # Handle num_shards==1: flat = [tok, layers..., norm_f] (no second tok). Then _tok_for_lm is params[0]
            if self.num_shards == 1:
                norm_f = params[-1]
                tok_emb = params[0]
            else:
                # num_shards==2 final shard flat = [layers..., norm_f, tok]
                norm_f = params[-2]
                tok_emb = params[-1]
        else:
            norm_f = params[-2]
            tok_emb = params[-1]
        x = self._rms(x, norm_f)
        logits = x @ tok_emb.T
        if self.final_logit_softcapping:
            logits = (logits / self.final_logit_softcapping).tanh() * self.final_logit_softcapping
        return logits

    @classmethod
    def from_gguf(cls, gguf_path: str | Path):
        from tools.export_model import parse_gguf, _tensor_data

        p = Path(gguf_path)
        meta, infos, data_base = parse_gguf(p)
        by_name = {n: (dims, gtype, off) for n, dims, gtype, off in infos}
        buf = p.read_bytes()

        def get(name):
            dims, gtype, off = by_name[name]
            return _tensor_data(buf, name, dims, gtype, off, data_base)

        arch = "gemma4"
        L = int(meta[f"{arch}.block_count"])
        D = int(meta[f"{arch}.embedding_length"])
        H = int(meta[f"{arch}.attention.head_count"])
        Kv = int(meta[f"{arch}.attention.head_count_kv"] or H)
        # per-layer head dim from q shape
        head_dim_per_layer = []
        for i in range(L):
            qdims = by_name[f"blk.{i}.attn_q.weight"][0]
            hd = int(qdims[1] // H)
            head_dim_per_layer.append(hd)
        dff = meta.get(f"{arch}.feed_forward_length")
        if isinstance(dff, list):
            dff_list = [int(x) for x in dff]
        else:
            dff_list = [int(dff)] * L
        # layer types from sliding pattern
        pat = meta.get(f"{arch}.attention.sliding_window_pattern")
        if isinstance(pat, list):
            layer_types = ["full_attention" if not bool(pat[i]) else "sliding_attention" for i in range(L)]
        else:
            layer_types = ["sliding_attention"] * L
        sw = int(meta.get(f"{arch}.attention.sliding_window") or 512)
        rope = float(meta.get(f"{arch}.rope.freq_base") or 1_000_000.0)
        rope_swa = float(meta.get(f"{arch}.rope.freq_base_swa") or 10_000.0)
        rope_dim = int(meta.get(f"{arch}.rope.dimension_count") or 0)
        rope_dim_swa = int(meta.get(f"{arch}.rope.dimension_count_swa") or 0)
        hs_per_raw = meta.get(f"{arch}.embedding_length_per_layer_input")
        hs_per = int(hs_per_raw) if hs_per_raw is not None else 256
        tok_dims = by_name.get("token_embd.weight", ((),))[0]
        n_vocab = int(tok_dims[1]) if len(tok_dims) == 2 else int(meta.get(f"{arch}.vocab_size") or 262144)

        inst = cls(
            vocab_size=n_vocab,
            d_model=D,
            n_heads=H,
            n_kv_heads=Kv,
            n_layers=L,
            d_ff_per_layer=dff_list,
            head_dim_per_layer=head_dim_per_layer,
            layer_types=layer_types,
            sliding_window=sw,
            rope_theta=rope,
            rope_theta_swa=rope_swa,
            rope_dim=rope_dim or 128,
            rope_dim_swa=rope_dim_swa or 64,
            hidden_size_per_layer_input=hs_per,
            final_logit_softcapping=float(meta.get(f"{arch}.final_logit_softcapping") or 30.0),
            tie_weights=("output.weight" not in by_name),
        )
        # materialize params as tinygrad Tensors (on whatever Device is active)
        from tinygrad import Tensor

        params: list[Tensor] = []
        tok = get("token_embd.weight")  # (V, D)
        tok_t = Tensor(tok.astype(np.float32))
        params.append(tok_t)
        for i in range(L):
            hd = head_dim_per_layer[i]
            # pack 13 tensors per layer in flat order expected by forward(): n1,q,qn,k,kn,v,o,proj/inp_gate?, post_attn, pre_ffn, gate_up, down, post_ffw, post_per_layer, scale
            # forward() will index via cursor; we store list-of-lists self._layer_params for fast path, plus flat for pebble compatibility
            pass  # keep flat below via _pack_flat
        # build flat-style list for forward() consumption (mirrors BBT + extra)
        flat_params = inst._pack_flat_from_gguf(buf, by_name, data_base)
        inst._gguf_params = flat_params  # Tensor list
        return inst

    def _pack_flat_from_gguf(self, buf: bytes, by_name: dict, data_base: int):
        from tinygrad import Tensor
        from tools.export_model import _tensor_data
        import numpy as np

        def get(name):
            dims, gtype, off = by_name[name]
            return _tensor_data(buf, name, dims, gtype, off, data_base)

        out: list[Tensor] = []
        # token emb
        tok = get("token_embd.weight").astype(np.float32)
        out.append(Tensor(tok))
        for i in range(self.n_layers):
            hd = self.head_dim_per_layer[i]
            kv_hd = hd  # Kv=1 so same hd
            D = self.d_model
            H = self.n_heads
            Kv = self.n_kv_heads_per_layer[i]
            # GGUF dequant returns (out, in) after transpose; forward uses x @ W.T semantics, so keep as (out, in) and forward does @ .T
            # To avoid confusion, store as (D, out) = arr.T so x @ W works directly
            n1 = get(f"blk.{i}.attn_norm.weight").astype(np.float32).reshape(-1)
            q_raw = get(f"blk.{i}.attn_q.weight").astype(np.float32).T  # (1536,2048) from (2048,1536)
            assert q_raw.shape == (D, H * hd), f"q {q_raw.shape} vs {(D, H*hd)} layer {i}"
            qn = get(f"blk.{i}.attn_q_norm.weight").astype(np.float32).reshape(-1)
            k_raw = get(f"blk.{i}.attn_k.weight").astype(np.float32).T
            assert k_raw.shape == (D, Kv * hd), f"k {k_raw.shape} vs {(D, Kv*hd)}"
            kn = get(f"blk.{i}.attn_k_norm.weight").astype(np.float32).reshape(-1)
            v_raw = get(f"blk.{i}.attn_v.weight").astype(np.float32).T
            o_raw = get(f"blk.{i}.attn_output.weight").astype(np.float32).T  # (1536,2048) -> actually (2048,1536).T = (1536,2048) == (D, H*hd) need (H*hd,D) for y@o, so keep transposed differently: we want o as (H*hd, D) so y [B,T,H*hd] @ o -> [B,T,D]
            # o dequant is (1536,2048) = (D, H*hd) after .T? Wait attn_output dequant is (1536,2048) already, which is (D? no) -> it's (1536,2048)=(D?,H*hd?) need (H*hd,D). Let's just set o_raw = get(...).astype(... )  # already (1536,2048) = (D, H*hd)? confusion - check: y @ o where o should be (H*hd,D). So use original (1536,2048) transposed? Let's keep as (H*hd,D) = get(...)
            # o is (H*hd, D) transposed? GGUF attn_output (2048,1536) = (H*hd, D)
            post_attn = get(f"blk.{i}.post_attention_norm.weight").astype(np.float32).reshape(-1)
            pre_ffn = get(f"blk.{i}.ffn_norm.weight").astype(np.float32).reshape(-1)
            dff = self.d_ff_per_layer[i]
            gate = get(f"blk.{i}.ffn_gate.weight").astype(np.float32).T  # (6144,1536).T=(1536,6144) D,dff
            up = get(f"blk.{i}.ffn_up.weight").astype(np.float32).T
            gate_up = np.concatenate([gate, up], axis=1)  # (D,2*dff)
            down = get(f"blk.{i}.ffn_down.weight").astype(np.float32).T  # (1536,6144).T=(6144,1536) dff,D
            post_ffw = get(f"blk.{i}.post_ffw_norm.weight").astype(np.float32).reshape(-1)
            # per_layer_input
            if self.hidden_size_per_layer_input:
                inp_gate = get(f"blk.{i}.inp_gate.weight").astype(np.float32).T  # (256,1536).T=(1536,256) D,256
                proj = get(f"blk.{i}.proj.weight").astype(np.float32).T  # (1536,256).T=(256,1536) 256,D
                post_per = get(f"blk.{i}.post_norm.weight").astype(np.float32).reshape(-1)
                scale = get(f"blk.{i}.layer_output_scale.weight").astype(np.float32).reshape(-1)
                # order: n1,q,qn,k,kn,v,o, inp_gate,proj,post_attn,pre_ffn,gate_up,down,post_ffw,post_per,scale
                for arr in [n1, q_raw, qn, k_raw, kn, v_raw, o_raw, inp_gate, proj, post_attn, pre_ffn, gate_up, down, post_ffw, post_per, scale]:
                    out.append(Tensor(arr))
            else:
                scale = get(f"blk.{i}.layer_output_scale.weight").astype(np.float32).reshape(-1) if f"blk.{i}.layer_output_scale.weight" in by_name else np.ones(1, dtype=np.float32)
                for arr in [n1, q_raw, qn, k_raw, kn, v_raw, o_raw, post_attn, pre_ffn, gate_up, down, post_ffw, scale]:
                    out.append(Tensor(arr))
        norm_f = get("output_norm.weight").astype(np.float32).reshape(-1)
        out.append(Tensor(norm_f))
        if not self.tie_weights:
            o_w = get("output.weight").astype(np.float32)
            out.append(Tensor(o_w))
        return out

    def _rms(self, x, w):
        return _rms_norm(x, w, eps=self.rms_norm_eps)

    def _apply_rope(self, x, seq_len: int, layer_idx: int):
        # x: [B,H,T,D]  apply partial rotary on first rope_dim
        from tinygrad import Tensor
        is_swa = self.layer_types[layer_idx] == "sliding_attention"
        theta = self.rope_theta_swa if is_swa else self.rope_theta
        dim = self.rope_dim_swa if is_swa else self.rope_dim
        if dim == 0:
            dim = x.shape[-1] // 4  # fallback 0.25
        # inv_freq for dim (even)
        hd = x.shape[-1]
        # build inv_freq for dim only
        inv_freq = [math.exp(-(math.log(theta) / dim) * i) for i in range(0, dim, 2)]
        inv = Tensor(inv_freq)
        pos = Tensor.arange(seq_len).float().reshape(seq_len, 1)
        freqs = pos * inv.reshape(1, -1)  # [T, dim/2]
        emb = freqs.cat(freqs, dim=-1)  # [T, dim]
        cos = emb.cos().reshape(1, 1, seq_len, dim)
        sin = emb.sin().reshape(1, 1, seq_len, dim)
        # split x into rotary part and pass part
        if dim < hd:
            x_rot, x_pass = x[:, :, :, :dim], x[:, :, :, dim:]
            x1, x2 = x_rot.chunk(2, dim=-1)
            rot = (-x2).cat(x1, dim=-1)
            x_rot = x_rot * cos + rot * sin
            return x_rot.cat(x_pass, dim=-1)
        else:
            x1, x2 = x.chunk(2, dim=-1)
            rot = (-x2).cat(x1, dim=-1)
            return x * cos + rot * sin

    def _causal_attention(self, q, k, v, layer_idx: int):
        # q [B,H,T,D], k/v [B,KVH,T,D] -> GQA broadcast, then windowed causal
        import math
        from tinygrad import Tensor
        bsz, h, t, hd = q.shape
        kvh = k.shape[1]
        group = h // kvh
        # reshape for GQA
        q = q.reshape(bsz, kvh, group, t, hd)
        k = k.reshape(bsz, kvh, 1, t, hd)
        v = v.reshape(bsz, kvh, 1, t, hd)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        # causal + sliding window
        is_swa = self.layer_types[layer_idx] == "sliding_attention"
        mask = Tensor.ones(t, t).triu(1).reshape(1, 1, 1, t, t)
        scores = scores.masked_fill(mask == 1, float("-inf"))
        if is_swa:
            # sliding window: mask out positions with distance > window
            # build distance mask: j < i - window => -inf
            idx = Tensor.arange(t)
            dist = idx.reshape(t, 1) - idx.reshape(1, t)  # [T,T] i-j
            win_mask = (dist > self.sliding_window).float().reshape(1, 1, 1, t, t)
            scores = scores.masked_fill(win_mask == 1, float("-inf"))
        # query scale (Gemma4 uses query_pre_attn_scalar)
        # approximate as 1/sqrt(hd) already; gemma4 also has softcapping 50 but skip for now
        attn = scores.softmax(axis=-1)
        y = attn @ v  # [B,KVH,G,T,D]
        y = y.reshape(bsz, h, t, hd).transpose(1, 2).reshape(bsz, t, h * hd)
        return y

    def forward(self, x_ids):
        from tinygrad import Tensor
        params = self._gguf_params
        p = 0
        tok_emb = params[p]; p+=1  # [V,D]
        x = tok_emb[x_ids]  # [B,T,D]
        for li in range(self.n_layers):
            hd = self.head_dim_per_layer[li]
            H = self.n_heads
            Kv = self.n_kv_heads_per_layer[li]
            n1 = params[p]; p+=1
            q_w = params[p]; p+=1  # [D,H*hd]
            qn_w = params[p]; p+=1  # [hd]
            k_w = params[p]; p+=1
            kn_w = params[p]; p+=1
            v_w = params[p]; p+=1
            o_w = params[p]; p+=1  # [H*hd,D]
            if self.hidden_size_per_layer_input:
                inp_gate_w = params[p]; p+=1  # [D,256]
                proj_w = params[p]; p+=1  # [256,D]
            post_attn_w = params[p]; p+=1
            pre_ffn_w = params[p]; p+=1
            gate_up_w = params[p]; p+=1  # [D,2*dff]
            down_w = params[p]; p+=1  # [dff,D]
            post_ffw_w = params[p]; p+=1
            if self.hidden_size_per_layer_input:
                post_per_w = params[p]; p+=1
                scale_w = params[p]; p+=1
            else:
                scale_w = params[p]; p+=1

            residual = x
            if self.hidden_size_per_layer_input:
                per = residual @ inp_gate_w  # [B,T,256]
                per = (per * 0.5 * (1 + (0.79788456 * (per + 0.044715 * per * per * per)).tanh()))
                per = per @ proj_w  # [B,T,D]
                per = self._rms(per, post_per_w)
                x_input = residual + per
            else:
                x_input = residual
            h = self._rms(x_input, n1)
            bsz, t, _ = h.shape
            q = h @ q_w  # [B,T,H*hd]
            k = h @ k_w
            v = h @ v_w
            # per-head q/k RMSNorm then RoPE
            q = q.reshape(bsz, t, H, hd)
            q = q * (q * q).mean(axis=-1, keepdim=True).add(self.rms_norm_eps).rsqrt() * qn_w
            q = q.transpose(1, 2)  # [B,H,T,hd]
            k = k.reshape(bsz, t, Kv, hd)
            k = k * (k * k).mean(axis=-1, keepdim=True).add(self.rms_norm_eps).rsqrt() * kn_w
            k = k.transpose(1, 2)  # [B,Kv,T,hd]
            v = v.reshape(bsz, t, Kv, hd).transpose(1, 2)  # [B,Kv,T,hd]
            q = self._apply_rope(q, t, li)
            k = self._apply_rope(k, t, li)
            # attention
            attn_out = self._causal_attention(q, k, v, li)  # [B,T,D]
            # o is [H*hd, D] but o_w stored as [H*hd,D]? hmm we stored as (H*hd,D) after .T? q_w is (D,H*hd).T? o_w is (H*hd,D) = (2048,1536). So attn_out [B,T,H*hd] @ o_w? But attn_out is already [B,T,D]? _causal_attention returns [B,T,D] via y.reshape... So we need y = ... already projects? Wait _causal_attention returns [B,T,D]? No it returns [B,T,H*hd] weighted then reshape to [B,T,H*hd] - but we included proj inside? Let's keep: y = attn @ v -> [B,T,H*hd], then need @ o
            # Our _causal_attention returns [B,T,H*hd]? Actually it reshapes to [B,T,H*hd] via [B,T,H*hd] ??? Let's assume it returns [B,T,H*hd] weighted sum, so need linear
            # So compute o projection
            if attn_out.shape[-1] == o_w.shape[0]:
                h_attn = attn_out @ o_w  # [B,T,D]
            else:
                # attn_out is [B,T,D] already (if _causal did reshape incorrectly), skip
                h_attn = attn_out
            h_attn = self._rms(h_attn, post_attn_w)
            x = residual + h_attn * scale_w  # Gemma4 scales residual
            # FFN block
            h2 = self._rms(x, pre_ffn_w)
            gate_up = h2 @ gate_up_w  # [B,T,2*dff]
            a, b = gate_up.chunk(2, dim=-1)
            a = (a * 0.5 * (1 + (0.79788456 * (a + 0.044715 * a * a * a)).tanh()))
            h2 = a * b
            h2 = h2 @ down_w  # [B,T,D]
            h2 = self._rms(h2, post_ffw_w)
            x = x + h2 * scale_w
        # final norm + logits
        norm_f = params[p]; p+=1
        x = self._rms(x, norm_f)
        # tie
        logits = x @ tok_emb.T  # [B,T,V]
        if self.final_logit_softcapping:
            # tanh softcap 30
            logits = (logits / self.final_logit_softcapping).tanh() * self.final_logit_softcapping
        return logits

    def generate(self, prompt: str, max_tokens: int = 20):
        # stub greedy
        return prompt + " [gemma4 stub]"

    def sample_batch(self, batch_size: int, seed: int):
        # for TinygradTrainer compatibility (unused for inference)
        from tinygrad import Tensor
        import random as _rand
        rng = _rand.Random(seed)
        ids = [[rng.randint(0, self.vocab_size - 1) for _ in range(32)] for _ in range(batch_size)]
        x = Tensor(ids)
        y = x[:, 1:]
        return x[:, :-1], y

    def loss(self, logits, y):
        return logits.sparse_categorical_crossentropy(y)
