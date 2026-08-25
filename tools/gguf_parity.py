#!/usr/bin/env python3
"""Real-GGUF parity check: numpy pure-matmul reference vs the browser JS model.

Builds a llama-arch forward in numpy (embedding, RMSNorm, RoPE, GQA attention,
SwiGLU, final norm, lm_head) from the dequantized weights of a real GGUF, runs it
on fixed token ids, then runs the SAME wire-format weights through the JS engine
(node + gabionLoader) and compares logits. The Python adapter is a BitLinear
(ternary) transformer and cannot serve as a float-model reference.

Both sides emit [B*(T-1), V] -- the last position is dropped -- and EVERY logit is
compared, so an error at t>0 (RoPE, causal mask, GQA grouping) cannot hide behind a
truncated sample.

Usage: python tools/gguf_parity.py <model.gguf> [--tokens 0 1 2 3 ...]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "gabion")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.export_model import export_gguf, export_hf, f16_decode  # noqa: E402


def build_reference(out: dict, tokens: list[int], B: int = 1, T: int = 8):
    """Pure-matmul llama forward in numpy; returns (logits_np, flat_weights)."""
    cfg = out["config"]
    V, D, H, kvH, L, dFF = (cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                            cfg["n_kv_heads"], cfg["n_layers"], cfg["d_ff"])
    hd = D // H
    group = H // kvH
    kvD = kvH * hd
    eps = 1e-6
    inv_freq = np.array([math.exp(-(math.log(cfg.get("rope_base", 10000.0)) / hd) * (i * 2))
                         for i in range(hd // 2)], dtype=np.float32)

    def rope(x):  # x: [B, Hr, T, hd]
        B_, Hr, T_, _ = x.shape
        pos = np.arange(T_, dtype=np.float32).reshape(T_, 1)
        freqs = pos * inv_freq.reshape(1, -1)  # [T, hd/2]
        emb = np.concatenate([freqs, freqs], axis=-1)  # [T, hd]
        cos = np.cos(emb).reshape(1, 1, T_, hd)
        sin = np.sin(emb).reshape(1, 1, T_, hd)
        x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
        rot = np.concatenate([-x2, x1], axis=-1)
        return x * cos + rot * sin

    def rms(x, w):
        norm = np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps)
        return x / norm * w

    flat = f16_decode(out["weights_b64"])
    cur = 0

    def take(shape):
        nonlocal cur
        n = int(np.prod(shape))
        a = flat[cur:cur + n].reshape(shape).copy()
        cur += n
        return a

    tok_emb = take((V, D))
    per = 2 * D * D + 2 * D * kvD + 2 * D + 2 * D * dFF + dFF * D
    assert len(flat) == V * D + L * per + D + (0 if cfg["tie_weights"] else D * V), len(flat)
    x = np.asarray(tok_emb[tokens], dtype=np.float32).reshape(B, T, D)  # [B,T,D]
    qbias = out.get("q_bias") or [None] * L
    kbias = out.get("k_bias") or [None] * L
    vbias = out.get("v_bias") or [None] * L
    for l in range(L):
        qb = np.asarray(qbias[l], dtype=np.float32) if qbias[l] is not None else None
        kb = np.asarray(kbias[l], dtype=np.float32) if kbias[l] is not None else None
        vb = np.asarray(vbias[l], dtype=np.float32) if vbias[l] is not None else None
        q_w, k_w, v_w, o_w, n1_w, gate_up_w, n2_w, down_w = (
            take((D, D)), take((D, kvD)), take((D, kvD)), take((D, D)),
            take((D,)), take((D, 2 * dFF)), take((D,)), take((dFF, D)))
        # attention
        h = rms(x, n1_w)
        q = h @ q_w  # [B,T,D]
        k = h @ k_w
        v = h @ v_w
        if qb is not None: q = q + qb.reshape(1, 1, -1)
        if kb is not None: k = k + kb.reshape(1, 1, -1)
        if vb is not None: v = v + vb.reshape(1, 1, -1)
        q = q.reshape(B, T, H, hd).transpose(0, 2, 1, 3)  # [B,H,T,hd]
        k = k.reshape(B, T, kvH, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, kvH, hd).transpose(0, 2, 1, 3)
        q = rope(q)
        k = rope(k)
        q = q.reshape(B, kvH, group, T, hd)
        k = k.reshape(B, kvH, 1, T, hd)
        v = v.reshape(B, kvH, 1, T, hd)
        scores = (q @ k.transpose(0, 1, 2, 4, 3)) / math.sqrt(hd)  # [B,KVH,G,T,T]
        mask = np.triu(np.ones((T, T), dtype=np.float32), 1).reshape(1, 1, 1, T, T)
        scores = np.where(mask == 1, -np.inf, scores)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        y = attn @ v  # [B,KVH,G,T,hd]
        y = y.reshape(B, H, T, hd).transpose(0, 2, 1, 3).reshape(B, T, D)
        y = y @ o_w
        x = x + y
        # ffn (SwiGLU, mirroring the JS _swiGLU): silu(h@gate) * (h@up) @ down
        h = rms(x, n2_w)
        gate, up = gate_up_w[:, :dFF], gate_up_w[:, dFF:]
        h_g = h @ gate
        act = h_g * (1.0 / (1.0 + np.exp(-h_g))) * (h @ up)
        x = x + (act @ down_w)
    x = rms(x, take((D,)))
    if cfg["tie_weights"]:
        logits = x @ tok_emb.T
    else:
        logits = x @ take((D, V))
    assert cur == len(flat), (cur, len(flat))
    # slice: predict x[1..T] from x[0..T-1]
    return logits[:, :T - 1, :].reshape(B * (T - 1), V), flat


def run_js(flat: np.ndarray, out: dict, tokens: list[int], B: int, T: int) -> dict:
    """Run the same weights through the JS engine in node.

    The wire base64 JSON is ~1.7GB for this model — past Node's string limit — so
    the parity harness writes the flat f32 weights as a binary file and loads them
    directly (the f16 wire codec is exercised separately by the small-model tests).
    """
    cfg = out["config"]
    bin_path = Path(ROOT) / ".gguf_parity_flat.f32"
    logits_path = Path(ROOT) / ".gguf_parity_logits.f32"
    bin_path.write_bytes(np.asarray(flat, dtype="<f4").tobytes())
    src = f"""
const fs = require('fs'), vm = require('vm');
const cfg = {json.dumps(cfg)};
const tokens = {json.dumps(tokens)};
// readFileSync refuses >2GiB; read the flat f32 weights via a loop
const fd = fs.openSync({json.dumps(str(bin_path))}, 'r');
const stat = fs.fstatSync(fd);
const buf = Buffer.allocUnsafe(stat.size);
let off = 0;
while (off < stat.size) {{
  const chunk = Math.min(stat.size - off, 0x7fffffff);
  off += fs.readSync(fd, buf, off, chunk, off);
}}
fs.closeSync(fd);
const weights = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
const sandbox = {{ window: {{}}, console, Float32Array, Int32Array, Uint32Array, Math, Error,
  btoa: (s) => Buffer.from(s,'binary').toString('base64'), atob: (s) => Buffer.from(s,'base64').toString('binary'),
  TextDecoder, TextEncoder }};
sandbox.window = sandbox; sandbox.globalThis = sandbox;
vm.createContext(sandbox);
for (const rel of ['gabion/web/tinygrad_v0.js','gabion/web/bbt_forward.js','gabion/web/tokenizer.js','gabion/web/model_loader.js'])
  vm.runInContext(fs.readFileSync(rel,'utf8'), sandbox, {{filename: rel}});
(async () => {{
  const tg = sandbox.window.tinygradV0;
  const model = new tg.BBTTransformer({{
    vocabSize: cfg.vocab_size, dModel: cfg.d_model, nHeads: cfg.n_heads,
    kvHeads: cfg.n_kv_heads || cfg.n_heads, nLayers: cfg.n_layers,
    seqLen: cfg.seq_len, dFF: cfg.d_ff,
    tieWeights: cfg.tie_weights, actQuant: cfg.act_quant, ropeBase: cfg.rope_base,
    qBiases: {json.dumps(out.get("q_bias"))} ? {json.dumps(out.get("q_bias"))}.map(a => new Float32Array(a)) : null,
    kBiases: {json.dumps(out.get("k_bias"))} ? {json.dumps(out.get("k_bias"))}.map(a => new Float32Array(a)) : null,
    vBiases: {json.dumps(out.get("v_bias"))} ? {json.dumps(out.get("v_bias"))}.map(a => new Float32Array(a)) : null,
  }});
  const consumed = model.loadFlatWeights(weights, false);
  if (consumed !== weights.length) throw new Error(`cursor ${{consumed}} != ${{weights.length}}`);
  const xFlat = Int32Array.from(tokens);
  const logits = await model.forward(xFlat, {B}, {T}, false);
  // Write every logit as binary; JSON.stringify of ~1e6 floats is slow and the
  // old `.slice(0, 5000)` silently reduced the comparison to a fraction of row 0.
  const d = logits.data;
  fs.writeFileSync({json.dumps(str(logits_path))},
    Buffer.from(d.buffer, d.byteOffset, d.byteLength));
  console.log(JSON.stringify({{ n: d.length }}));
}})().catch(e => {{ console.error(e); process.exit(1); }});
"""
    tmp = Path(ROOT) / ".gguf_parity_tmp.cjs"
    tmp.write_text(src, encoding="utf-8")
    try:
        proc = subprocess.run(["node", str(tmp)], cwd=str(ROOT), capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            raise RuntimeError(f"node failed: {proc.stderr[:2000]}")
        lines = proc.stdout.strip().splitlines()
        if not lines:
            raise RuntimeError(f"node produced no stdout; stderr: {proc.stderr[:2000]}")
        res = json.loads(lines[-1])
        res["data"] = np.fromfile(logits_path, dtype="<f4")
    finally:
        tmp.unlink(missing_ok=True)
        bin_path.unlink(missing_ok=True)
        logits_path.unlink(missing_ok=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("gguf", type=Path, nargs="?", help="GGUF model file")
    src.add_argument("--from-hf", type=Path, metavar="DIR",
                     help="HuggingFace checkpoint dir (config.json + model.safetensors)")
    ap.add_argument("--tokens", type=int, nargs="*", default=None)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    if args.from_hf:
        print(f"[1/3] exporting HF dir {args.from_hf} ...")
        out = export_hf(args.from_hf)
    else:
        print(f"[1/3] exporting {args.gguf} ...")
        out = export_gguf(args.gguf)
    cfg = out["config"]
    print("     config:", {k: cfg[k] for k in ("vocab_size", "d_model", "n_heads", "n_kv_heads", "n_layers", "d_ff", "tie_weights", "rope_base")})
    if args.out_json:
        args.out_json.write_text(json.dumps(out))
        print(f"     wire json -> {args.out_json}")

    T = min(args.seq, cfg["seq_len"])
    if T < 2:
        raise SystemExit("--seq must be >= 2: the last position is dropped from the logits")
    # cycle rather than truncate, so --seq beyond the supplied/default token list works
    base = list(args.tokens) if args.tokens else [0, 1, 2, 3, 4, 5, 6, 7]
    if not base:
        raise SystemExit("--tokens given but empty")
    bad = [t for t in base if not 0 <= t < cfg["vocab_size"]]
    if bad:
        raise SystemExit(f"token ids out of range for vocab_size={cfg['vocab_size']}: {bad}")
    tokens = [base[i % len(base)] for i in range(T)]
    print(f"[2/3] numpy reference forward (T={T}, tokens={tokens}) ...")
    ref, flat = build_reference(out, tokens, B=1, T=T)
    print(f"     ref logits [{ref.shape[0]}x{ref.shape[1]}], finite={np.isfinite(ref).all()}, std={np.std(ref):.4f}")

    print("[3/3] JS engine forward ...")
    js = run_js(flat, out, tokens, 1, T)
    ref_flat = np.asarray(ref).reshape(-1).astype(np.float64)
    js_data = np.asarray(js["data"], dtype=np.float64)
    # Both sides are [B*(T-1), V]; a length mismatch is itself a parity failure and
    # must not be papered over by comparing a common prefix.
    if len(js_data) != len(ref_flat):
        print(f"     LENGTH MISMATCH js={len(js_data)} ref={len(ref_flat)}")
        print("verdict FAIL")
        return 2
    d = np.abs(ref_flat - js_data)
    worst = int(np.argmax(d))
    print(f"     logits compared: {len(d)} (all of [{ref.shape[0]}x{ref.shape[1]}])")
    print(f"     max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
          f"(worst at position {worst // ref.shape[1]}, vocab {worst % ref.shape[1]})")
    # per-position worst, so an error confined to t>0 is visible rather than averaged away
    per_pos = d.reshape(ref.shape).max(axis=1)
    print("     per-position max_abs: " + " ".join(f"{v:.2e}" for v in per_pos))
    ok = d.max() < 1e-3 and math.isfinite(float(np.mean(js_data)))
    print("verdict", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
