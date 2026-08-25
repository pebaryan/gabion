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

const conv = new tg.nn.Conv2d(1, 2, 3, { padding: 1, bias: true });
conv.weight.data.set(Float32Array.from(fixture.conv_w));
conv.bias.data.set(Float32Array.from(fixture.conv_b));
const convY = conv.forward(Tensor.fromArray(fixture.conv_x, fixture.conv_x_shape, false));

const gn = new tg.nn.GroupNorm(2, 4, { eps: 1e-5 });
gn.weight.data.set(Float32Array.from(fixture.gn_w));
gn.bias.data.set(Float32Array.from(fixture.gn_b));
const gnY = gn.forward(Tensor.fromArray(fixture.gn_x, fixture.gn_x_shape, false));

tg.training = true;
const bn = new tg.nn.BatchNorm(3, { eps: 1e-5, momentum: 0.1 });
bn.weight.data.set(Float32Array.from(fixture.bn_w));
bn.bias.data.set(Float32Array.from(fixture.bn_b));
const bnY = bn.forward(Tensor.fromArray(fixture.bn_x, fixture.bn_x_shape, false));
tg.training = false;

const lstm = new tg.nn.LSTMCell(3, 4, { bias: true });
lstm.weightIh.data.set(Float32Array.from(fixture.lstm_wih));
lstm.weightHh.data.set(Float32Array.from(fixture.lstm_whh));
lstm.biasIh.data.set(Float32Array.from(fixture.lstm_bih));
lstm.biasHh.data.set(Float32Array.from(fixture.lstm_bhh));
const [lh, lc] = lstm.forward(Tensor.fromArray(fixture.lstm_x, [2, 3], false));

const lp = Tensor.fromArray(fixture.lamb_p, [fixture.lamb_p.length], true);
lp.grad = Float32Array.from(fixture.lamb_g);
const lamb = tg.LAMB([lp], {
  lr: fixture.lr, beta1: 0.9, beta2: 0.999, eps: 1e-8,
  weightDecay: fixture.wd, warmupSteps: 1, gradClipNorm: 0,
});
await lamb.step();

const convg = new tg.nn.Conv2d(4, 4, 3, { groups: 2, padding: 1, bias: true });
convg.weight.data.set(Float32Array.from(fixture.convg_w));
convg.bias.data.set(Float32Array.from(fixture.convg_b));
const convgY = convg.forward(Tensor.fromArray(fixture.convg_x, fixture.convg_x_shape, false));

const convd = new tg.nn.Conv2d(1, 1, 3, { dilation: 2, padding: 2, bias: false });
convd.weight.data.set(Float32Array.from(fixture.convd_w));
const convdY = convd.forward(Tensor.fromArray(fixture.convd_x, fixture.convd_x_shape, false));

const ct = new tg.nn.ConvTranspose2d(1, 1, 3, { bias: true });
ct.weight.data.set(Float32Array.from(fixture.ct_w));
ct.bias.data.set(Float32Array.from(fixture.ct_b));
const ctY = ct.forward(Tensor.fromArray(fixture.ct_x, fixture.ct_x_shape, false));

// ConvTranspose2d backward, simple (stride 1, padding 0): drive closure with fixture gout
const ctx = Tensor.fromArray(fixture.ct_x, fixture.ct_x_shape, true);
const ctYg = ctx.convTranspose2d(ct.weight, ct.bias, 1, 1, 1, 0, 0);
ctYg.grad = Float32Array.from(fixture.ct_g);
ctYg._backward(ctYg.grad);

// ConvTranspose2d backward, hard (groups 2, stride 2, padding 1, output_padding 1)
const ct2 = new tg.nn.ConvTranspose2d(2, 4, 3, { stride: 2, padding: 1, outputPadding: 1, groups: 2, bias: true });
ct2.weight.data.set(Float32Array.from(fixture.ct2_w));
ct2.bias.data.set(Float32Array.from(fixture.ct2_b));
const ct2Yf = ct2.forward(Tensor.fromArray(fixture.ct2_x, fixture.ct2_x_shape, false));
const ct2x = Tensor.fromArray(fixture.ct2_x, fixture.ct2_x_shape, true);
const ct2Yg = ct2x.convTranspose2d(ct2.weight, ct2.bias, 2, 2, 1, 1, 1);
ct2Yg.grad = Float32Array.from(fixture.ct2_g);
ct2Yg._backward(ct2Yg.grad);

const mp = Tensor.fromArray(fixture.muon_p, [4, 4], true);
mp.grad = Float32Array.from(fixture.muon_g);
const muon = tg.Muon([mp], { lr: 1e-3, warmupSteps: 1, gradClipNorm: 0 });
await muon.step();

const report = {
  nn_modules: Object.keys(tg.nn).sort(),
  optim_exports: Object.keys(tg.optim).sort(),
  relu_max_abs: maxAbs(relu.data, fixture.relu),
  gelu_max_abs: maxAbs(gelu.data, fixture.gelu),
  ln_max_abs: maxAbs(lnOut.data, fixture.ln_y),
  dropout_identity_ok: maxAbs(dropOff.data, fixture.x) === 0,
  dropout_p1_zero: Array.from(dropAll.data).every((v) => v === 0),
  adamw_max_abs: maxAbs(p.data, fixture.p_after),
  conv_max_abs: maxAbs(convY.data, fixture.conv_y),
  gn_max_abs: maxAbs(gnY.data, fixture.gn_y),
  bn_max_abs: maxAbs(bnY.data, fixture.bn_y),
  lstm_h_max_abs: maxAbs(lh.data, fixture.lstm_h),
  lstm_c_max_abs: maxAbs(lc.data, fixture.lstm_c),
  lamb_max_abs: maxAbs(lp.data, fixture.lamb_after),
  convg_max_abs: maxAbs(convgY.data, fixture.convg_y),
  convd_max_abs: maxAbs(convdY.data, fixture.convd_y),
  ct_max_abs: maxAbs(ctY.data, fixture.ct_y),
  ct_grad_x: maxAbs(ctx.grad, fixture.ct_x_grad),
  ct_grad_w: maxAbs(ct.weight.grad, fixture.ct_w_grad),
  ct_grad_b: maxAbs(ct.bias.grad, fixture.ct_b_grad),
  ct2_max_abs: maxAbs(ct2Yf.data, fixture.ct2_y),
  ct2_grad_x: maxAbs(ct2x.grad, fixture.ct2_x_grad),
  ct2_grad_w: maxAbs(ct2.weight.grad, fixture.ct2_w_grad),
  ct2_grad_b: maxAbs(ct2.bias.grad, fixture.ct2_b_grad),
  muon_max_abs: maxAbs(mp.data, fixture.muon_after),
};
const outPath = fixturePath.replace(/\.json$/, "_js.json");
fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
