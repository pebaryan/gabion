// CPU smoke: load browser IIFEs in Node and compare against a Python fixture.
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const fixturePath = process.argv[2] || path.join(here, "_js_bbt_fixture.json");

const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
const sandbox = {
  window: {},
  WebGPUBackend: undefined,
  console,
  Float32Array,
  Int32Array,
  Uint32Array,
  Math,
  Error,
  btoa: (s) => Buffer.from(s, "binary").toString("base64"),
  atob: (s) => Buffer.from(s, "base64").toString("binary"),
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
// fetch shim: gabionLoader.loadBBTModel fetches the wire JSON
sandbox.fetch = async (url) => ({ ok: true, json: async () => fixture });
vm.createContext(sandbox);
for (const rel of ["gabion/web/tinygrad_v0.js", "gabion/web/bbt_forward.js", "gabion/web/tokenizer.js", "gabion/web/model_loader.js"]) {
  const src = fs.readFileSync(path.join(root, rel), "utf8");
  vm.runInContext(src, sandbox, { filename: rel });
}

const tg = sandbox.window.tinygradV0;
if (!tg || !tg.BBTTransformer) {
  throw new Error("tinygradV0.BBTTransformer not installed");
}

const cfg = fixture.config;
const model = new tg.BBTTransformer({
  vocabSize: cfg.vocab_size,
  dModel: cfg.d_model,
  nHeads: cfg.n_heads,
  kvHeads: cfg.n_kv_heads || cfg.n_heads,
  nLayers: cfg.n_layers,
  seqLen: cfg.seq_len,
  dFF: cfg.d_ff,
  tieWeights: cfg.tie_weights,
  actQuant: cfg.act_quant,
});
const weights = Float32Array.from(fixture.weights);
const consumed = model.loadFlatWeights(weights, false);
if (consumed !== weights.length) {
  throw new Error(`weight cursor ${consumed} != ${weights.length} (paramCount=${model.paramCount()})`);
}

const B = fixture.batch_size;
const T = cfg.seq_len;
const xFlat = Int32Array.from(fixture.x_flat);
const yFlat = Int32Array.from(fixture.y_flat);

// Cross-language wire codec check: Python f16 base64 -> JS decode must match
// the flat weights fixture (this is the model_loader.js contract).
const wireWeights = tg.f16Base64ToWeights(fixture.weights_b64);
let wireMaxAbs = 0;
for (let i = 0; i < wireWeights.length; i++) {
  wireMaxAbs = Math.max(wireMaxAbs, Math.abs(wireWeights[i] - fixture.weights[i]));
}

const logits = await model.forward(xFlat, B, T, !!fixture.ternarize);
const loss = await tg.Tensor.crossEntropy(logits, yFlat);

// GQA parity: 4 query heads, 2 KV heads. JS must reproduce the Python adapter
// forward exactly (grouped k/v projections + contiguous h//group expansion).
// ternarize=true matches Python's default act_quant path (STE bitlinear), same
// convention as the main model check.
const gqaCfg = fixture.gqa.config;
const gqaModel = new tg.BBTTransformer({
  vocabSize: gqaCfg.vocab_size,
  dModel: gqaCfg.d_model,
  nHeads: gqaCfg.n_heads,
  kvHeads: gqaCfg.n_kv_heads,
  nLayers: gqaCfg.n_layers,
  seqLen: gqaCfg.seq_len,
  dFF: gqaCfg.d_ff,
  tieWeights: gqaCfg.tie_weights,
  actQuant: gqaCfg.act_quant,
});
{
  const w = Float32Array.from(fixture.gqa.weights);
  const consumed = gqaModel.loadFlatWeights(w, false);
  if (consumed !== w.length) throw new Error(`gqa weight cursor ${consumed} != ${w.length}`);
}
const gqaLogits = await gqaModel.forward(xFlat, B, T, true);
const gqaRef = fixture.gqa.logits_flat;
let gqaMaxAbs = 0;
for (let i = 0; i < gqaLogits.data.length; i++) {
  gqaMaxAbs = Math.max(gqaMaxAbs, Math.abs(gqaLogits.data[i] - gqaRef[i]));
}

// GQA backward: finite-difference check on a k-weight element validates the
// group-sum expansion backward (ternarize=false = continuous plain matmuls).
let gqaBwd = null;
{
  const mod = gqaModel;
  const params = mod.parameters();
  const lossOf = async () => {
    const lg = await mod.forward(xFlat, B, T, false);
    return Number((await tg.Tensor.crossEntropy(lg, yFlat)).data[0]);
  };
  // autograd grad
  for (const p of params) p.grad = null;
  {
    const lg = await mod.forward(xFlat, B, T, false);
    const loss = await tg.Tensor.crossEntropy(lg, yFlat);
    loss.backward();
  }
  const kw = mod.layers[0].k.weight, vw = mod.layers[0].v.weight, qw = mod.layers[0].q.weight;
  const idx = 7; // arbitrary element of the grouped k projection
  const auto = kw.grad ? kw.grad[idx] : NaN;
  // finite difference
  const eps = 1e-3;
  const orig = kw.data[idx];
  kw.data[idx] = orig + eps;
  const lp = await lossOf();
  kw.data[idx] = orig - eps;
  const lm = await lossOf();
  kw.data[idx] = orig;
  const fd = (lp - lm) / (2 * eps);
  const relErr = Math.abs(fd - auto) / (1 + Math.abs(fd));
  const gradsPresent = !!kw.grad && !!vw.grad && !!qw.grad &&
    kw.grad.some((v) => v !== 0) && vw.grad.some((v) => v !== 0) && qw.grad.some((v) => v !== 0);
  gqaBwd = { relErr, gradsPresent, auto, fd };
}

// End-to-end loader: gabionLoader.loadBBTModel(wire JSON) must produce a model
// with identical weights/params (f16 codec + loadFlatWeights + tokenizer attach).
const loaded = await sandbox.gabionLoader.loadBBTModel(fixturePath);
const loadedOk = loaded.paramCount() === model.paramCount() && !!loaded.tokenize;
let loadedWeightsMax = 0;
{
  const a = loaded.tokEmb.weight.data, b = model.tokEmb.weight.data;
  for (let i = 0; i < a.length; i++) loadedWeightsMax = Math.max(loadedWeightsMax, Math.abs(a[i] - b[i]));
}

const jsLogits = Array.from(logits.data);
const jsLoss = Number(loss.data[0]);
const pyLogits = fixture.logits_flat;

// Decode check: CPU-fallback decodeStep must reproduce forward logits row-by-row
// (row pos predicts token pos+1 while attending the prefix 0..pos — identical to
// the KV-cache GPU path's semantics, which can't run in Node without WebGPU).
const outT = T - 1;
const st = model.initKVCache();
let decodeMax = 0;
for (let pos = 0; pos < outT; pos++) {
  const step = await model.decodeStep(xFlat[0 * T + pos], st, !!fixture.ternarize);
  const refBase = (0 * outT + pos) * cfg.vocab_size;
  for (let i = 0; i < step.logits.length; i++) {
    decodeMax = Math.max(decodeMax, Math.abs(step.logits[i] - logits.data[refBase + i]));
  }
}

if (jsLogits.length !== pyLogits.length) {
  throw new Error(`logit length js=${jsLogits.length} py=${pyLogits.length}`);
}
let maxAbs = 0;
let sumAbs = 0;
for (let i = 0; i < jsLogits.length; i++) {
  const d = Math.abs(jsLogits[i] - pyLogits[i]);
  if (d > maxAbs) maxAbs = d;
  sumAbs += d;
}
const meanAbs = sumAbs / jsLogits.length;
const lossAbs = Math.abs(jsLoss - fixture.loss);

// One CPU train step must finish and keep the vector length.
const trained = await tg.trainLocalV1(weights, {
  lr: 1e-3,
  epochs: 1,
  batchSize: B,
  seed: 1,
  vocabSize: cfg.vocab_size,
  dModel: cfg.d_model,
  nHeads: cfg.n_heads,
  nLayers: cfg.n_layers,
  seqLen: cfg.seq_len,
  dFF: cfg.d_ff,
  tieWeights: cfg.tie_weights,
  ternarize: !!fixture.ternarize,
  sequences: fixture.sequences,
  optimizer: "sgd",
  gradClipNorm: 0,
  warmupSteps: 1,
});
const updated = trained.updated || trained;
const updatedLen = updated.length || trained.weights_count || 0;

const report = {
  ok_load: consumed === weights.length,
  param_count: model.paramCount(),
  js_loss: jsLoss,
  py_loss: fixture.loss,
  loss_abs: lossAbs,
  logits_max_abs: maxAbs,
  logits_mean_abs: meanAbs,
  decode_max_abs: decodeMax,
  wire_max_abs: wireMaxAbs,
  loader_ok: loadedOk,
  loader_weight_max_abs: loadedWeightsMax,
  gqa_max_abs: gqaMaxAbs,
  gqa_bwd_rel_err: gqaBwd.relErr,
  gqa_bwd_grads: gqaBwd.gradsPresent,
  train_mode: trained.mode || null,
  train_loss: Number(trained.loss),
  train_updated_len: updatedLen,
  finite: Number.isFinite(jsLoss) && Number.isFinite(Number(trained.loss)),
};
const outPath = fixturePath.replace(/\.json$/, "_js.json");
fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
console.log(`wrote ${outPath}`);
