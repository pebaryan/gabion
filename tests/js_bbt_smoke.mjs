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
vm.createContext(sandbox);
for (const rel of ["gabion/web/tinygrad_v0.js", "gabion/web/bbt_forward.js"]) {
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
const logits = await model.forward(xFlat, B, T, !!fixture.ternarize);
const loss = await tg.Tensor.crossEntropy(logits, yFlat);

const jsLogits = Array.from(logits.data);
const jsLoss = Number(loss.data[0]);
const pyLogits = fixture.logits_flat;
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
  train_mode: trained.mode || null,
  train_loss: Number(trained.loss),
  train_updated_len: updatedLen,
  finite: Number.isFinite(jsLoss) && Number.isFinite(Number(trained.loss)),
};
const outPath = fixturePath.replace(/\.json$/, "_js.json");
fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
console.log(`wrote ${outPath}`);
