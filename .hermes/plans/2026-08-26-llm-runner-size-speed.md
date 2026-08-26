# LLM Runner Optimization Plan — Size then Speed

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Cut the browser LLM runner's download size (JS + weights) and inference latency, with size first (cheapest wins, no correctness risk) then speed (GPU residency, fewer syncs).

**Architecture:** JS bundle is ~300 KB raw (~60 KB gz) + kernels ~80 KB raw (~36 KB gz) + weights 0.9–4 GB f16. Size wins come from wire format + dead-code elimination + compression; speed wins come from removing 8× per-layer CPU↔GPU round-trips in `bbt_forward.js` and tiling the hot `matmul` kernel.

**Tech Stack:** `gabion/web/{tinygrad_v0.js,bbt_forward.js,webgpu_backend.js,kernels/*.wgsl}`, `tools/export_model.py` (wire format), `gabion/web/{model_loader.js,tokenizer.js}`, `tests/js_bbt_smoke.py` + `tests/js_wgsl_kernel_smoke.mjs` for parity.

---

## Baseline (measure before changing)

Run these now and record:
```bash
wc -c gabion/web/*.js gabion/web/kernels/*.wgsl
gzip -c gabion/web/tinygrad_v0.js | wc -c; gzip -c gabion/web/bbt_forward.js | wc -c; gzip -c gabion/web/webgpu_backend.js | wc -c; gzip -c gabion/web/kernels/*.wgsl | wc -c
ls -lh gabion/web/models/*.f16 gabion/web/models/*.json 2>&1 | head
grep -c "toCPU" gabion/web/bbt_forward.js  # expect 8
grep -n "onGPU.*toCPU\|await.*toCPU" gabion/web/bbt_forward.js
```
Current (2026-08-26): `tinygrad 149K/28K gz, bbt_forward 77K/18K gz, webgpu_backend 65K/9K gz, kernels 80K/36K gz, tokenizer 5.8K, model_loader 7K` → JS total ~300K raw / ~60K gz. Weights: `qwen2.5-0.5b.f16 943M` + shards. Each `await x.toCPU()` before `RMSNorm` forces a GPU readback + re-upload. `matmul.wgsl` is naive global-memory (no tiling).

---

### Task 1: Record size/speed baseline + wire-format audit

**Objective:** Capture before numbers so gains are verifiable.

**Files:**
- Create: `.hermes/plans/baseline-llm-runner.md` (or `docs/plans/`)
- Modify: none

**Step 1: Measure JS + kernel + weight sizes**
```bash
echo "raw"; wc -c gabion/web/*.js gabion/web/kernels/*.wgsl | tail -1
echo "gz JS"; gzip -c gabion/web/tinygrad_v0.js gabion/web/bbt_forward.js gabion/web/webgpu_backend.js gabion/web/tokenizer.js gabion/web/model_loader.js | wc -c
echo "gz kernels"; cat gabion/web/kernels/*.wgsl | gzip -c | wc -c
ls -lh gabion/web/models/ | head -20
ls -lh web/models/ 2>&1 | head -20
```

**Step 2: Audit wire usage**
```bash
grep -rn "weights_b64\|loadBBTModel\|loadBBTModelBin\|loadCoordinator\|loadShard" gabion/web/ tools/ | head -30
grep -rn "weights_b64" gabion/web/models/*.json 2>&1 | head
cat gabion/web/demo.html | grep -n "loadBBT" | head
cat gabion/web/shards.html | grep -n "load" | head
```

**Step 3: Count sync points**
```bash
grep -n "toCPU\|onGPU" gabion/web/bbt_forward.js | head -30
grep -n "createBufferFromData\|toGPU\|_dirty" gabion/web/tinygrad_v0.js | head -20
```

**Step 4: Commit baseline note**
```bash
git add .hermes/plans/baseline-llm-runner.md
git commit -m "docs: LLM runner baseline — size + sync audit"
```
Expected: file with numbers; no code change.

---

### Task 2: Make binary wire the default (size: -33% weight download)

**Objective:** Stop shipping base64 JSON weights; use `model.json + weights.f16` (already implemented in `model_loader.js`) everywhere.

**Files:**
- Modify: `gabion/web/demo.html:1-40` (fetch call)
- Modify: `gabion/web/shards.html:1-40` (coord + shard fetch)
- Modify: `tools/export_model.py` help text / default `--out` handling
- Test: manual `python tools/export_model.py --from-gguf … --out-bin` produces pair

**Step 1: Write failing check — demo still uses base64**
```bash
grep -n "loadBBTModel[^B]" gabion/web/demo.html
# expect hit if still base64 path
```

**Step 2: Switch demo.html to loadBBTModelBin**
```js
// before: const model = await gabionLoader.loadBBTModel("models/qwen2.5-0.5b.json")
// after:
const model = await gabionLoader.loadBBTModelBin("models/qwen2.5-0.5b.json", "models/qwen2.5-0.5b.f16");
```
Same for `shards.html`: coordinator `loadCoordinator(manifest, "coord.f16")`, shard `loadShard(manifest, f16Url, i)`.

**Step 3: Add export helper**
```bash
python tools/export_model.py --help | grep -A2 out-bin
# ensure --out-bin writes model.json + weights.f16 (already exists per loader)
```

**Step 4: Verify**
```bash
ls -lh gabion/web/models/*.json gabion/web/models/*.f16 2>&1 | head
# json ~8M, f16 ~943M vs single .json ~1.2G (base64 33% larger) — confirm json no longer contains weights_b64
python -c "import json; d=json.load(open('gabion/web/models/qwen2.5-0.5b.json')); print('has weights_b64:', 'weights_b64' in d)"
# expect False for new export
```

**Step 5: Commit**
```bash
git add gabion/web/demo.html gabion/web/shards.html tools/export_model.py
git commit -m "feat(web): binary wire default — model.json + weights.f16 (33% smaller than base64)"
```

---

### Task 3: Ship an inference-only JS bundle (size: -40–50 KB raw)

**Objective:** Training kernels are shipped to every browser but never used for inference. Provide `tinygrad_infer.js` + `kernels/infer/` that strips backward + optimizer kernels.

**Files:**
- Create: `tools/build_infer_bundle.py`
- Create: `gabion/web/tinygrad_infer.js` (generated)
- Modify: `gabion/web/webgpu_backend.js:1-20` add `INFER_ONLY` guard (optional)
- Test: `tests/js_bbt_smoke.py` with infer bundle

**Step 1: List kernels only needed for inference**
```
keep: matmul, batched_matmul, batched_transpose, embedding_forward, rmsnorm_forward, layernorm, fused_attention, kv_attention_scores/apply, kv_group_sum, softmax, silu_mul, reduce
drop: *bwd*, *backward*, adam*, conv*_bwd*, lstm*_bwd*, dropout_bwd (unless training demo), batchnorm_bwd*
```

**Step 2: Build script — copy subset + stub getPipeline for missing**
```python
# tools/build_infer_bundle.py
# - reads gabion/web/tinygrad_v0.js, strips Optimizer/Backward methods via AST or string filter
# - copies only keep-list wgsl files to gabion/web/kernels/infer/
# - writes gabion/web/tinygrad_infer.js with window.tinygradV0 infer flag
```

**Step 3: Verify**
```bash
python tools/build_infer_bundle.py
wc -c gabion/web/tinygrad_v0.js gabion/web/tinygrad_infer.js
ls -1 gabion/web/kernels/infer/ | wc -l  # expect ~15 vs 40
node --check gabion/web/tinygrad_infer.js
DEV=CPU C:/Users/aryan/miniconda3/envs/p311/python.exe tests/js_bbt_smoke.py 2>&1 | grep verdict
# must still PASS (logits/loss/train may skip, but decode/wire/loader PASS)
```

**Step 4: Commit**
```bash
git add tools/build_infer_bundle.py gabion/web/tinygrad_infer.js
git commit -m "feat(web): inference-only bundle — strip backward/optim kernels"
```

---

### Task 4: Enable brotli + gzip precompression for static assets

**Objective:** Serve `.js` + `.wgsl` + `.f16` with `Content-Encoding: br` (or at least document nginx/caddy config). Cheap size win: br is ~15% better than gz on JS.

**Files:**
- Create: `tools/compress_assets.py`
- Modify: `README.md` / `TINYGRAD_RUNTIME.md` — add serving notes
- Test: local check `brotli -c gabion/web/tinygrad_v0.js | wc -c` vs gzip

**Step 1: Script**
```python
# compress_assets.py: for f in web/**/*.js web/**/*.wgsl web/**/*.f16: write f.br + f.gz if missing
```

**Step 2: Verify**
```bash
python tools/compress_assets.py
ls -lh gabion/web/*.js.br gabion/web/*.js.gz 2>&1 | head
brotli -c gabion/web/tinygrad_v0.js | wc -c; gzip -c gabion/web/tinygrad_v0.js | wc -c
```

**Step 3: Commit**
```bash
git add tools/compress_assets.py README.md
git commit -m "feat(web): brotli/gzip precompressed assets + serving docs"
```

---

### Task 5: Minify JS bundles (size: -20–30% post-gz)

**Objective:** Terser/minify `tinygrad_v0.js` + `bbt_forward.js` + `webgpu_backend.js` for release, keep source maps for debug.

**Files:**
- Create: `tools/minify.py` or use `npx terser`
- Modify: `.gitignore` to allow `*.min.js` + `*.map`
- Test: `node --check` + smoke still passes with `.min.js`

**Step 1: Try terser**
```bash
npx terser gabion/web/tinygrad_v0.js --compress --mangle --output gabion/web/tinygrad_v0.min.js --source-map
ls -lh gabion/web/*.min.js
gzip -c gabion/web/tinygrad_v0.min.js | wc -c  # expect < 28K
```

**Step 2: Switch demo to .min.js when ?min=1 or prod flag**
```html
<script src="tinygrad_v0.min.js"></script>
```

**Step 3: Commit**
```bash
git add tools/minify.py gabion/web/*.min.js
git commit -m "feat(web): minified bundles with source maps"
```

---

### Task 6: Verify size wins (gate)

**Objective:** Prove size reduction with numbers.

**Files:**
- Modify: `.hermes/plans/baseline-llm-runner.md` add "after" table

**Steps:**
```bash
echo "before gz JS"; cat .hermes/plans/baseline-llm-runner.md | grep gz
echo "after"; gzip -c gabion/web/tinygrad_infer.min.js gabion/web/bbt_forward.js gabion/web/webgpu_backend.js | wc -c
ls -lh gabion/web/models/*.json gabion/web/models/*.f16
# assert: JS gz < 45K (vs 60K), weight JSON no base64, assets have .br
pytest tests/js_bbt_smoke.py -v 2>&1 | grep verdict
node tests/js_wgsl_kernel_smoke.mjs 2>&1 | tail -5
```
Commit docs.

---

### Task 7: Profile the hot path — count and time CPU syncs (speed baseline)

**Objective:** Quantify the 8× per-layer `toCPU()` stalls before fixing them.

**Files:**
- Create: `tools/profile_bbt.html` or `gabion/web/test_gpu_head.html` instrumentation
- Modify: `gabion/web/bbt_forward.js` add `console.time` around block (non-breaking)

**Steps:**
```bash
grep -n "toCPU" gabion/web/bbt_forward.js
# instrument: before/after each await x.toCPU() log elapsed
# run shards.html with ?profile=1, copy console output
```
Expected: each `_block` does 4 readbacks, each `forward` does 2 more. Total ~4*L+2 stalls per token.

---

### Task 8: Keep RMSNorm on GPU (speed: -4*L stalls/token)

**Objective:** Port `RMSNorm.forward` to GPU so `x` stays resident; remove `await x.toCPU()` in `_block`.

**Files:**
- Modify: `gabion/web/kernels/rmsnorm_forward.wgsl` (ensure exists, else create)
- Modify: `gabion/web/webgpu_backend.js` add `rmsnorm(xBuf, rows, dim, eps, weightBuf)`
- Modify: `gabion/web/tinygrad_v0.js` class RMSNorm — add GPU path
- Modify: `gabion/web/bbt_forward.js:_block` remove 2× `toCPU` before `norm1/norm2`

**Step 1: Verify kernel exists**
```bash
ls gabion/web/kernels/rmsnorm*.wgsl
cat gabion/web/kernels/rmsnorm_forward.wgsl 2>&1 | head -20
```

**Step 2: Implement GPU rmsnorm in backend + tinygrad**
```js
// tinygrad_v0.js RMSNorm.forward:
// if (backend && this.onGPU) return new Tensor(backend.rmsnorm(...), shape, ...)
```

**Step 3: Remove syncs in bbt_forward.js**
```js
// before: if (x.onGPU) await x.toCPU(); h = bl.norm1.forward(x)
// after: h = bl.norm1.forward(x)  // stays on GPU
```

**Step 4: Verify**
```bash
node --check gabion/web/tinygrad_v0.js && node --check gabion/web/bbt_forward.js
grep -n "toCPU" gabion/web/bbt_forward.js  # expect 4 left (down from 8)
DEV=CPU python tests/js_bbt_smoke.py 2>&1 | grep gqa  # must PASS
```

**Step 5: Commit**
```bash
git add gabion/web/kernels/rmsnorm_forward.wgsl gabion/web/webgpu_backend.js gabion/web/tinygrad_v0.js gabion/web/bbt_forward.js
git commit -m "perf(web): GPU-resident RMSNorm — remove 4*L CPU stalls"
```

---

### Task 9: Fuse RoPE into attention matmuls (speed: fewer dispatches)

**Objective:** RoPE cos/sin tables are uploaded but Q/K still reshaped on CPU before attention. Add a fused `rope` kernel or keep Q/K on GPU through `_causalSelfAttention`.

**Files:**
- Modify: `gabion/web/kernels/rope.wgsl` (create, or extend `fused_attention.wgsl`)
- Modify: `gabion/web/webgpu_backend.js` add `rope(qBuf, ...)` dispatch
- Modify: `gabion/web/bbt_forward.js:_causalSelfAttention` keep `q,k` on GPU, avoid `toCPU` reshape

**Verification:**
```bash
grep -n "toCPU" gabion/web/bbt_forward.js  # expect 2 left
python tests/js_wgsl_kernel_smoke.mjs 2>&1 | grep -A2 rmsnorm
```

---

### Task 10: Keep KV cache on GPU across decode steps (speed: no per-token re-upload)

**Objective:** `kCache/vCache` are JS `Float32Array`s today; each decode copies them. Move to GPU buffers with `kv_attention_scores` / `kv_attention_apply` already GPU-accelerated.

**Files:**
- Modify: `gabion/web/bbt_forward.js` — add `this._kvCacheGPU` buffers, update in place via `writeBuffer`
- Modify: `gabion/web/webgpu_backend.js` ensure `kv_attention_*` accept GPU cache

**Verification:** shards.html decode 32 tokens, measure ms/token before/after (console.time).

---

### Task 11: Tile matmul kernel (speed: 2–4× on large D)

**Objective:** `matmul.wgsl` does 1 thread per output element with naive global reads. Tile 16×16 with `var<workgroup>` shared memory, like `batched_matmul.wgsl` but for 2D.

**Files:**
- Modify: `gabion/web/kernels/matmul.wgsl`
- Modify: `gabion/web/kernels/batched_matmul.wgsl` (reference tile)
- Test: `tests/js_wgsl_kernel_smoke.mjs` matmul parity

**Steps:**
```bash
cp gabion/web/kernels/matmul.wgsl gabion/web/kernels/matmul.wgsl.bak
# replace with tiled version: workgroup_size 16x16, shared A/B tiles, loop over K tiles
node tests/js_wgsl_kernel_smoke.mjs 2>&1 | grep matmul
DEV=CPU python tests/js_bbt_smoke.py 2>&1 | grep verdict
```

---

### Task 12: Verify speed + size gates (final)

**Objective:** Prove both wins without regression.

**Files:**
- Create: `docs/plans/llm-runner-optim-results.md`

**Steps:**
```bash
# size
gzip -c gabion/web/tinygrad_infer.min.js | wc -c
cat gabion/web/kernels/*.wgsl | gzip -c | wc -c
ls -lh gabion/web/models/*.f16 | head

# speed — run shards.html?profile=1 or node smoke with timers
node tests/js_bbt_smoke.py 2>&1 | grep verdict
node tests/js_wgsl_kernel_smoke.mjs 2>&1 | grep -E "matmul|kv|r rmsnorm"

# commit
git add docs/plans/llm-runner-optim-results.md
git commit -m "docs: LLM runner size+speed results"
git push origin HEAD
```

---

## Review Checklist
- [ ] Tasks are 2–5 min, exact paths, copy-paste commands
- [ ] Size phase before speed (no speed work starts until Task 6 passes)
- [ ] Every kernel change has CPU parity check (`js_wgsl_kernel_smoke`)
- [ ] Every JS change has `node --check` + `js_bbt_smoke` PASS
- [ ] No weight correctness change (f16 round-trip)
- [ ] Docs updated for binary wire + compression serving
