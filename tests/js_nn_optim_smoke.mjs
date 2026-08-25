// Compare new nn/optim coverage against a Python tinygrad fixture.
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const fixturePath = process.argv[2] || path.join(here, "_js_nn_optim_fixture.json");
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
vm.runInContext(
  fs.readFileSync(path.join(root, "gabion/web/tinygrad_v0.js"), "utf8"),
  sandbox,
  { filename: "tinygrad_v0.js" },
);

const tg = sandbox.window.tinygradV0;
const Tensor = tg.Tensor;

function maxAbs(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

const x = Tensor.fromArray(fixture.x, fixture.x_shape, false);
const relu = x.relu();
const gelu = x.gelu();
const ln = new tg.nn.LayerNorm(fixture.ln_dim, { eps: 1e-5 });
ln.weight.data.set(Float32Array.from(fixture.ln_weight));
ln.bias.data.set(Float32Array.from(fixture.ln_bias));
const lnOut = ln.forward(Tensor.fromArray(fixture.ln_x, fixture.ln_x_shape, false));

const dropOff = Tensor.fromArray(fixture.x, fixture.x_shape, false).dropout(0.3);
tg.training = true;
const dropAll = Tensor.fromArray(fixture.x, fixture.x_shape, false).dropout(1.0);
tg.training = false;

const p = Tensor.fromArray(fixture.p, [fixture.p.length], true);
p.grad = Float32Array.from(fixture.g);
const opt = tg.AdamW([p], {
  lr: fixture.lr,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
  weightDecay: fixture.wd,
  warmupSteps: 1,
  gradClipNorm: 0,
});
await opt.step();

const report = {
  nn_modules: Object.keys(tg.nn).sort(),
  optim_exports: Object.keys(tg.optim).sort(),
  relu_max_abs: maxAbs(relu.data, fixture.relu),
  gelu_max_abs: maxAbs(gelu.data, fixture.gelu),
  ln_max_abs: maxAbs(lnOut.data, fixture.ln_y),
  dropout_identity_ok: maxAbs(dropOff.data, fixture.x) === 0,
  dropout_p1_zero: Array.from(dropAll.data).every((v) => v === 0),
  adamw_max_abs: maxAbs(p.data, fixture.p_after),
};
const outPath = fixturePath.replace(/\.json$/, "_js.json");
fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
