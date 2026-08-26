# LLM Runner Baseline — 2026-08-26

## JS bundle (raw / gzip)

| file | raw | gz |
|------|-----|----|
| tinygrad_v0.js | 148849 | 28247 |
| bbt_forward.js | 77542 | 18577 |
| webgpu_backend.js | 64898 | 9178 |
| tokenizer.js | 5839 | 2162 |
| model_loader.js | 7072 | 2400 |
| **JS total** | 304200 | ~59192 (cat *.js \| gzip) |
| kernels/*.wgsl (40 files) | 79493 | 15206 (cat \| gzip) |
| **total static** | 383693 raw | ~74400 gz |

## Weights

| asset | size | format |
|-------|------|--------|
| qwen2.5-0.5b.json (base64?) | 8.4M | json with config only? actually 8.4M is json part, 943M f16 binary already split |
| qwen2.5-0.5b.f16 | 943M | raw f16 |
| base64 equivalent would be | ~1257M | +33% (4/3) |
| shards | 0.9-4B shards present | already binary sharded |

Wire format: `tools/export_model.py` still defaults to `weights_b64` base64 JSON, but `model_loader.js` has `loadBBTModelBin` (binary wire) and `demo.html` conditionally uses it when url ends .gabion.json. Shards already use binary (model.json + coord.f16 + shard_*.f16).

## Sync audit (bbt_forward.js)

`grep -n toCPU` = 20 hits:
- forward(): 2 (normF, logitsFlat)
- _block(): 4 (norm1, attn, norm2, ffn) — 4 per layer
- _causalSelfAttention(): 3 (q,k,v) + 2 (out)
- _swiGLU(): 2-3
- qwen35 linear: 4 matmuls with .toCPU() per token
- lfm2 conv: 2

Per token with L=24: ~4*L + extras ≈ 100 GPU↔CPU round-trips. Each is a `mapAsync` stall.

## Kernel inventory

- 40 wgsl files, ~15 inference-only (matmul, batched_matmul, transpose, embedding, rmsnorm, layernorm, fused_attention, kv_*, softmax, silu_mul, reduce, affine, etc)
- ~25 training/backward only (adam, conv_bwd, lstm_bwd, batchnorm_bwd, dropout_bwd, etc) — dead for inference.

## Next (size phase)

- Task 2: make binary wire default everywhere (save 33% on non-sharded models)
- Task 3: inference-only bundle (drop ~25 wgsl + optimizer code → -40KB raw)
- Task 4: brotli/gzip precompress
- Task 5: terser minify
