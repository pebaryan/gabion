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

// Tensor sum/mean reductions
const redx = Tensor.fromArray(fixture.red_x, fixture.red_x_shape, true);
const sumY = redx.sum([2, 3]);
sumY.grad = Float32Array.from(fixture.sum_g);
sumY._backward(sumY.grad);
const meanX = Tensor.fromArray(fixture.red_x, fixture.red_x_shape, true);
const meanY = meanX.mean([1]);
meanY.grad = Float32Array.from(fixture.mean_g);
meanY._backward(meanY.grad);
const meanSumNode = meanY._parents[0]; // sum -> scale composition
meanSumNode._backward(meanSumNode.grad);

// Tensor pad/concat
const padX = Tensor.fromArray(fixture.pad_x, fixture.pad_x_shape, true);
const padY = padX.pad(fixture.pad_pads);
padY.grad = Float32Array.from(fixture.pad_g);
padY._backward(padY.grad);
const pad2X = Tensor.fromArray(fixture.pad2_x, fixture.pad2_x_shape, true);
const pad2Y = pad2X.pad(fixture.pad2_pads);
pad2Y.grad = Float32Array.from(fixture.pad2_g);
pad2Y._backward(pad2Y.grad);
const catA = Tensor.fromArray(fixture.cat_a, fixture.cat_a_shape, true);
const catB = Tensor.fromArray(fixture.cat_b, fixture.cat_b_shape, true);
const catY = catA.concat(catB, fixture.cat_axis);
catY.grad = Float32Array.from(fixture.cat_g);
catY._backward(catY.grad);
const cat2A = Tensor.fromArray(fixture.cat2_a, fixture.cat2_a_shape, true);
const cat2B = Tensor.fromArray(fixture.cat2_b, fixture.cat2_b_shape, true);
const cat2Y = cat2A.concat(cat2B, fixture.cat2_axis);
cat2Y.grad = Float32Array.from(fixture.cat2_g);
cat2Y._backward(cat2Y.grad);

// SinusoidalTimestep
const ste = new tg.nn.SinusoidalTimestep(fixture.ste_dim).forward(fixture.timesteps);

// ResBlock fwd + backward
const rb = new tg.nn.ResBlock(8, 12, 16, { numGroups: 4 });
rb.norm1.weight.data.set(Float32Array.from(fixture.rb_gn1w));
rb.norm1.bias.data.set(Float32Array.from(fixture.rb_gn1b));
rb.conv1.weight.data.set(Float32Array.from(fixture.rb_conv1w));
rb.norm2.weight.data.set(Float32Array.from(fixture.rb_gn2w));
rb.norm2.bias.data.set(Float32Array.from(fixture.rb_gn2b));
rb.conv2.weight.data.set(Float32Array.from(fixture.rb_conv2w));
rb.timeMlp.weight.data.set(Float32Array.from(fixture.rb_mlpw));
rb.skip.weight.data.set(Float32Array.from(fixture.rb_skipw));
const rbX = Tensor.fromArray(fixture.rb_x, fixture.rb_x_shape, true);
const rbT = Tensor.fromArray(fixture.rb_t, [2, 16], false);
const rbY = rb.forward(rbX, rbT);
rbY.grad = Float32Array.from(fixture.rb_gout);
// deep graph: full topo walk like backward() but seeded with gout
{
  const topo = [];
  const seen = new Set();
  const build = (t) => {
    if (seen.has(t)) return;
    seen.add(t);
    for (const p of t._parents) build(p);
    topo.push(t);
  };
  build(rbY);
  for (let i = topo.length - 1; i >= 0; i--) {
    const t = topo[i];
    if (t.grad === null || t.grad === undefined) continue;
    t._backward(t.grad, t._gradGPUBuf || t._pendingGradBuf || null);
  }
}

// SpatialAttention fwd + backward
const sa = new tg.nn.SpatialAttention(4, { numGroups: 2 });
sa.norm.weight.data.set(Float32Array.from(fixture.sa_gnw));
sa.norm.bias.data.set(Float32Array.from(fixture.sa_gnb));
sa.qkv.weight.data.set(Float32Array.from(fixture.sa_qkvw));
sa.proj.weight.data.set(Float32Array.from(fixture.sa_projw));
const saX = Tensor.fromArray(fixture.sa_x, fixture.sa_x_shape, true);
const saY = sa.forward(saX);
saY.grad = Float32Array.from(fixture.sa_gout);
{
  const topo = [];
  const seen = new Set();
  const build = (tt) => { if (seen.has(tt)) return; seen.add(tt); for (const q of tt._parents) build(q); topo.push(tt); };
  build(saY);
  for (let i = topo.length - 1; i >= 0; i--) { const tt = topo[i]; if (tt.grad === null || tt.grad === undefined) continue; tt._backward(tt.grad, tt._gradGPUBuf || tt._pendingGradBuf || null); }
}

// UNet tiny forward
const unet = new tg.nn.UNet(2, 4, 16, { chMults: [1, 2], numGroups: 2 });
unet.stem.weight.data.set(Float32Array.from(fixture.unet_stem_w));
unet.downBlocks[0].rb.norm1.weight.data.set(Float32Array.from(fixture.unet_d0_gn1w));
unet.downBlocks[0].rb.norm1.bias.data.set(Float32Array.from(fixture.unet_d0_gn1b));
unet.downBlocks[0].rb.conv1.weight.data.set(Float32Array.from(fixture.unet_d0_c1w));
unet.downBlocks[0].rb.norm2.weight.data.set(Float32Array.from(fixture.unet_d0_gn2w));
unet.downBlocks[0].rb.norm2.bias.data.set(Float32Array.from(fixture.unet_d0_gn2b));
unet.downBlocks[0].rb.conv2.weight.data.set(Float32Array.from(fixture.unet_d0_c2w));
unet.downBlocks[0].rb.timeMlp.weight.data.set(Float32Array.from(fixture.unet_d0_mlp));
unet.downBlocks[0].down.weight.data.set(Float32Array.from(fixture.unet_down0w));
unet.downBlocks[1].rb.norm1.weight.data.set(Float32Array.from(fixture.unet_d1_gn1w));
unet.downBlocks[1].rb.norm1.bias.data.set(Float32Array.from(fixture.unet_d1_gn1b));
unet.downBlocks[1].rb.conv1.weight.data.set(Float32Array.from(fixture.unet_d1_c1w));
unet.downBlocks[1].rb.norm2.weight.data.set(Float32Array.from(fixture.unet_d1_gn2w));
unet.downBlocks[1].rb.norm2.bias.data.set(Float32Array.from(fixture.unet_d1_gn2b));
unet.downBlocks[1].rb.conv2.weight.data.set(Float32Array.from(fixture.unet_d1_c2w));
unet.downBlocks[1].rb.timeMlp.weight.data.set(Float32Array.from(fixture.unet_d1_mlp));
unet.mid1.norm1.weight.data.set(Float32Array.from(fixture.unet_m1_gn1w));
unet.mid1.norm1.bias.data.set(Float32Array.from(fixture.unet_m1_gn1b));
unet.mid1.conv1.weight.data.set(Float32Array.from(fixture.unet_m1_c1w));
unet.mid1.norm2.weight.data.set(Float32Array.from(fixture.unet_m1_gn2w));
unet.mid1.norm2.bias.data.set(Float32Array.from(fixture.unet_m1_gn2b));
unet.mid1.conv2.weight.data.set(Float32Array.from(fixture.unet_m1_c2w));
unet.mid1.timeMlp.weight.data.set(Float32Array.from(fixture.unet_m1_mlp));
unet.midAttn.norm.weight.data.set(Float32Array.from(fixture.unet_mid_gnw));
unet.midAttn.norm.bias.data.set(Float32Array.from(fixture.unet_mid_gnb));
unet.midAttn.qkv.weight.data.set(Float32Array.from(fixture.unet_mid_qkv));
unet.midAttn.proj.weight.data.set(Float32Array.from(fixture.unet_mid_proj));
unet.mid2.norm1.weight.data.set(Float32Array.from(fixture.unet_m2_gn1w));
unet.mid2.norm1.bias.data.set(Float32Array.from(fixture.unet_m2_gn1b));
unet.mid2.conv1.weight.data.set(Float32Array.from(fixture.unet_m2_c1w));
unet.mid2.norm2.weight.data.set(Float32Array.from(fixture.unet_m2_gn2w));
unet.mid2.norm2.bias.data.set(Float32Array.from(fixture.unet_m2_gn2b));
unet.mid2.conv2.weight.data.set(Float32Array.from(fixture.unet_m2_c2w));
unet.mid2.timeMlp.weight.data.set(Float32Array.from(fixture.unet_m2_mlp));
unet.upBlocks[0].rb.norm1.weight.data.set(Float32Array.from(fixture.unet_u0_gn1w));
unet.upBlocks[0].rb.norm1.bias.data.set(Float32Array.from(fixture.unet_u0_gn1b));
unet.upBlocks[0].rb.conv1.weight.data.set(Float32Array.from(fixture.unet_u0_c1w));
unet.upBlocks[0].rb.norm2.weight.data.set(Float32Array.from(fixture.unet_u0_gn2w));
unet.upBlocks[0].rb.norm2.bias.data.set(Float32Array.from(fixture.unet_u0_gn2b));
unet.upBlocks[0].rb.conv2.weight.data.set(Float32Array.from(fixture.unet_u0_c2w));
unet.upBlocks[0].rb.timeMlp.weight.data.set(Float32Array.from(fixture.unet_u0_mlp));
unet.upBlocks[0].rb.skip.weight.data.set(Float32Array.from(fixture.unet_u0_skipw));
unet.upBlocks[0].up.weight.data.set(Float32Array.from(fixture.unet_up0w));
unet.upBlocks[1].rb.norm1.weight.data.set(Float32Array.from(fixture.unet_u1_gn1w));
unet.upBlocks[1].rb.norm1.bias.data.set(Float32Array.from(fixture.unet_u1_gn1b));
unet.upBlocks[1].rb.conv1.weight.data.set(Float32Array.from(fixture.unet_u1_c1w));
unet.upBlocks[1].rb.norm2.weight.data.set(Float32Array.from(fixture.unet_u1_gn2w));
unet.upBlocks[1].rb.norm2.bias.data.set(Float32Array.from(fixture.unet_u1_gn2b));
unet.upBlocks[1].rb.conv2.weight.data.set(Float32Array.from(fixture.unet_u1_c2w));
unet.upBlocks[1].rb.timeMlp.weight.data.set(Float32Array.from(fixture.unet_u1_mlp));
unet.upBlocks[1].rb.skip.weight.data.set(Float32Array.from(fixture.unet_u1_skipw));
unet.outNorm.weight.data.set(Float32Array.from(fixture.unet_out_gnw));
unet.outNorm.bias.data.set(Float32Array.from(fixture.unet_out_gnb));
unet.outConv.weight.data.set(Float32Array.from(fixture.unet_out_cw));
unet.outConv.bias.data.set(Float32Array.from(fixture.unet_out_cb));
const unetX = Tensor.fromArray(fixture.unet_x, fixture.unet_x_shape, true);
const unetT = Tensor.fromArray(fixture.unet_t, [2, 16], false);
const unetY = unet.forward(unetX, unetT);
const unetNoise = Tensor.fromArray(fixture.unet_noise, [2,2,8,8], false);
const unetDiff = unetY.sub(unetNoise);
const unetLoss = unetDiff.mul(unetDiff).mean();
unetLoss.grad = new Float32Array([1]);
{
  const topo=[];const seen=new Set();const build=(tt)=>{if(seen.has(tt))return;seen.add(tt);for(const q of tt._parents) build(q);topo.push(tt);};
  build(unetLoss);
  for(let i=topo.length-1;i>=0;i--){const tt=topo[i];if(tt.grad===null&&!tt._gradGPUBuf&&!tt._pendingGradBuf) continue;const g=tt.grad||new Float32Array(tt.numel).fill(0);tt._backward(g, tt._gradGPUBuf||tt._pendingGradBuf||null);}
}

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
  unet_fwd: maxAbs(unetY.data, fixture.unet_y),
  unet_loss: unetLoss.data[0],
  unet_loss_err: Math.abs(unetLoss.data[0] - fixture.unet_loss),
  unet_loss_g_stem: maxAbs(unet.stem.weight.grad, fixture.unet_loss_g_stem),
  unet_loss_g_out: maxAbs(unet.outConv.weight.grad, fixture.unet_loss_g_out),
  muon_max_abs: maxAbs(mp.data, fixture.muon_after),
  sum_fwd: maxAbs(sumY.data, fixture.sum_y),
  sum_grad: maxAbs(redx.grad.slice(), fixture.sum_x_grad),
  mean_fwd: maxAbs(meanY.data, fixture.mean_y),
  mean_grad: maxAbs(meanX.grad.slice(), fixture.mean_x_grad),
  pad_fwd: maxAbs(padY.data, fixture.pad_y),
  pad_grad: maxAbs(padX.grad.slice(), fixture.pad_x_grad),
  pad2_fwd: maxAbs(pad2Y.data, fixture.pad2_y),
  pad2_grad: maxAbs(pad2X.grad.slice(), fixture.pad2_x_grad),
  cat_fwd: maxAbs(catY.data, fixture.cat_y),
  cat_a_grad: maxAbs(catA.grad.slice(), fixture.cat_a_grad),
  cat_b_grad: maxAbs(catB.grad.slice(), fixture.cat_b_grad),
  cat2_fwd: maxAbs(cat2Y.data, fixture.cat2_y),
  cat2_a_grad: maxAbs(cat2A.grad.slice(), fixture.cat2_a_grad),
  cat2_b_grad: maxAbs(cat2B.grad.slice(), fixture.cat2_b_grad),
  ste: maxAbs(ste.data, fixture.ste_ref),
  rb_fwd: maxAbs(rbY.data, fixture.rb_y),
  rb_grad_x: maxAbs(rbX.grad.slice(), fixture.rb_x_grad),
  rb_grad_gn1w: maxAbs(rb.norm1.weight.grad, fixture.rb_gn1w_grad),
  rb_grad_gn1b: maxAbs(rb.norm1.bias.grad, fixture.rb_gn1b_grad),
  rb_grad_conv1w: maxAbs(rb.conv1.weight.grad, fixture.rb_conv1w_grad),
  rb_grad_gn2w: maxAbs(rb.norm2.weight.grad, fixture.rb_gn2w_grad),
  rb_grad_gn2b: maxAbs(rb.norm2.bias.grad, fixture.rb_gn2b_grad),
  rb_grad_conv2w: maxAbs(rb.conv2.weight.grad, fixture.rb_conv2w_grad),
  rb_grad_mlpw: maxAbs(rb.timeMlp.weight.grad, fixture.rb_mlp_grad),
  rb_grad_skipw: maxAbs(rb.skip.weight.grad, fixture.rb_skipw_grad),
  sa_fwd: maxAbs(saY.data, fixture.sa_y),
  sa_grad_x: maxAbs(saX.grad.slice(), fixture.sa_x_grad),
  sa_gnw: maxAbs(sa.norm.weight.grad, fixture.sa_gnw_grad),
  sa_gnb: maxAbs(sa.norm.bias.grad, fixture.sa_gnb_grad),
  sa_qkv: maxAbs(sa.qkv.weight.grad, fixture.sa_qkv_grad),
  sa_proj: maxAbs(sa.proj.weight.grad, fixture.sa_proj_grad),
};
const outPath = fixturePath.replace(/\.json$/, "_js.json");
fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
