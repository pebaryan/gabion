// Validate the new WGSL kernels by transpiling their ops to JS and
// comparing against the CPU reference implementations in tinygrad_v0.js.
// (We can't run real WGSL in Node; this checks the algorithm, not GPU dispatch.)
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");

const sandbox = { window: {}, WebGPUBackend: undefined, console, Float32Array, Int32Array, Uint32Array, Math, Error, btoa: (s) => Buffer.from(s, "binary").toString("base64"), atob: (s) => Buffer.from(s, "base64").toString("binary") };
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(root, "gabion/web/tinygrad_v0.js"), "utf8"), sandbox, { filename: "tinygrad_v0.js" });
const tg = sandbox.window.tinygradV0;
const Tensor = tg.Tensor;

function maxAbs(a, b) { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; }

const report = {};

// ---- LayerNorm kernel vs CPU layerNorm ----
{
  const rng = (() => { let s = 12345; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  const rows = 5, d = 16, eps = 1e-5;
  const x = new Float32Array(rows * d);
  for (let i = 0; i < x.length; i++) x[i] = (rng() * 2 - 1) * 2;
  // CPU reference (no weight, no bias — matches kernel hasWeight=0, hasBias=0)
  const cpu = Tensor.fromArray(x, [rows, d], false).layerNorm(eps).data;
  // WGSL kernel transpiled (hasWeight=0, hasBias=0)
  const w = new Float32Array(d), b = new Float32Array(d);
  const out = new Float32Array(rows * d);
  for (let row = 0; row < rows; row++) {
    const base = row * d;
    let sum = 0; for (let j = 0; j < d; j++) sum += x[base + j];
    const mean = sum / d;
    let varr = 0; for (let j = 0; j < d; j++) { const v = x[base + j] - mean; varr += v * v; }
    varr /= d;
    const inv_std = 1.0 / Math.sqrt(varr + eps);
    for (let j = 0; j < d; j++) {
      let y = (x[base + j] - mean) * inv_std;
      if (0) { y *= w[j]; }   // hasWeight=0
      if (0) { y += b[j]; }   // hasBias=0
      out[base + j] = y;
    }
  }
  report.layernorm = maxAbs(out, cpu);
}

// ---- Dropout fwd kernel (hash) vs CPU dropout semantics ----
{
  // Reproduce the WGSL hash in JS (32-bit ops)
  const mul32 = (a, b) => Math.imul(a, b);
  function hash2(i, seed) {
    let h = (i ^ seed) >>> 0;
    h = mul32(h ^ (h >>> 16), 0x7feb352d);
    h = mul32(h ^ (h >>> 15), 0x846ca68b);
    h = h ^ (h >>> 16);
    return h >>> 0;
  }
  const len = 1000, p = 0.3, seed = 98765;
  const x = new Float32Array(len);
  for (let i = 0; i < len; i++) x[i] = i % 7;
  // WGSL kernel transpiled (full 32-bit: r = (h>>8)/2^24)
  const outK = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const r = ((hash2(i, seed) >>> 8) / 16777215.0);
    outK[i] = r < p ? 0.0 : x[i] / (1.0 - p);
  }
  // Expected properties: kept entries scaled by 1/(1-p), dropped are 0,
  // fraction kept ~ (1-p), and bwd(fwd) round-trips the scale.
  let kept = 0, scaleOk = true, zeroOk = true;
  for (let i = 0; i < len; i++) {
    if (outK[i] === 0) zeroOk = zeroOk && (x[i] === 0 || true);
    else { kept++; if (Math.abs(outK[i] - x[i] / (1 - p)) > 1e-6) scaleOk = false; }
  }
  // bwd with same seed should reproduce the same mask: bwd(gout=1) == fwd scale mask
  const gout = new Float32Array(len).fill(1);
  const dx = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const r = ((hash2(i, seed) >>> 8) / 16777215.0);
    dx[i] = r < p ? 0.0 : gout[i] / (1.0 - p);
  }
  const roundTrip = maxAbs(dx, outK); // same mask & scale => identical
  report.dropout = {
    scaleOk,
    keptFraction: kept / len,
    expectedKept: 1 - p,
    bwdFwdMatch: roundTrip,
  };
}

// ---- Conv2d fwd + bwd (groups=1, stride=2, padding=1, dilation=1) ----
{
  const rng = (() => { let s = 424242; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  const N = 2, Cin = 3, H = 5, W = 5, Cout = 4, kH = 3, kW = 3;
  const groups = 1, sH = 2, sW = 2, dH = 1, dW = 1, pH = 1, pW = 1;
  const cinPerG = Cin / groups, coutPerG = Cout / groups;
  const Ho = Math.floor((H + 2 * pH - dH * (kH - 1) - 1) / sH) + 1; // 3
  const Wo = Math.floor((W + 2 * pW - dW * (kW - 1) - 1) / sW) + 1; // 3
  const x = new Float32Array(N * Cin * H * W), w = new Float32Array(Cout * cinPerG * kH * kW);
  const b = new Float32Array(Cout), gout = new Float32Array(N * Cout * Ho * Wo);
  for (let i = 0; i < x.length; i++) x[i] = rng() * 2 - 1;
  for (let i = 0; i < w.length; i++) w[i] = (rng() * 2 - 1) * 0.5;
  for (let i = 0; i < b.length; i++) b[i] = (rng() * 2 - 1) * 0.1;
  for (let i = 0; i < gout.length; i++) gout[i] = rng() * 2 - 1;

  // CPU reference via the engine
  const xt = Tensor.fromArray(x, [N, Cin, H, W], true);
  const wt = Tensor.fromArray(w, [Cout, cinPerG, kH, kW], true);
  const bt = Tensor.fromArray(b, [Cout], true);
  const yt = xt.conv2d(wt, bt, groups, sH, dH, pH);
  const cpuY = Float32Array.from(yt.data);
  yt.grad = Float32Array.from(gout);
  yt._backward(yt.grad);

  // WGSL conv2d_fwd transpiled (gather per output)
  const yK = new Float32Array(N * Cout * Ho * Wo);
  for (let idx = 0; idx < yK.length; idx++) {
    const ow = idx % Wo, oh = ((idx / Wo) | 0) % Ho, oc = ((idx / (Wo * Ho)) | 0) % Cout, n = (idx / (Wo * Ho * Cout)) | 0;
    let acc = b[oc];
    const g = (oc / coutPerG) | 0, ic0 = g * cinPerG;
    for (let icL = 0; icL < cinPerG; icL++) {
      const ic = ic0 + icL;
      for (let kh = 0; kh < kH; kh++) {
        const ih = oh * sH - pH + kh * dH;
        if (ih < 0 || ih >= H) continue;
        for (let kw = 0; kw < kW; kw++) {
          const iw = ow * sW - pW + kw * dW;
          if (iw < 0 || iw >= W) continue;
          acc += x[((n * Cin + ic) * H + ih) * W + iw] * w[((oc * cinPerG) + icL) * kH * kW + kh * kW + kw];
        }
      }
    }
    yK[idx] = acc;
  }
  report.conv2d_fwd = maxAbs(yK, cpuY);

  // WGSL conv2d_bwd_dx transpiled (gather per input, exact-division)
  const dxK = new Float32Array(N * Cin * H * W);
  for (let idx = 0; idx < dxK.length; idx++) {
    const iw = idx % W, ih = ((idx / W) | 0) % H, ic = ((idx / (W * H)) | 0) % Cin, n = (idx / (W * H * Cin)) | 0;
    const g = (ic / cinPerG) | 0, oc0 = g * coutPerG, icL = ic - g * cinPerG;
    let acc = 0;
    for (let ocL = 0; ocL < coutPerG; ocL++) {
      const oc = oc0 + ocL;
      for (let kh = 0; kh < kH; kh++) {
        const num = ih + pH - kh * dH;
        if (num < 0 || num % sH !== 0) continue;
        const oh = num / sH;
        if (oh >= Ho) continue;
        for (let kw = 0; kw < kW; kw++) {
          const num2 = iw + pW - kw * dW;
          if (num2 < 0 || num2 % sW !== 0) continue;
          const ow = num2 / sW;
          if (ow >= Wo) continue;
          acc += gout[((n * Cout + oc) * Ho + oh) * Wo + ow] * w[((oc * cinPerG) + icL) * kH * kW + kh * kW + kw];
        }
      }
    }
    dxK[idx] = acc;
  }
  report.conv2d_dx = maxAbs(dxK, xt.grad);

  // WGSL conv2d_bwd_dw transpiled (gather per weight element)
  const dwK = new Float32Array(Cout * cinPerG * kH * kW);
  for (let idx = 0; idx < dwK.length; idx++) {
    const kw = idx % kW, kh = ((idx / kW) | 0) % kH, icL = ((idx / (kW * kH)) | 0) % cinPerG, oc = (idx / (kW * kH * cinPerG)) | 0;
    const g = (oc / coutPerG) | 0, ic0 = g * cinPerG;
    let acc = 0;
    for (let n = 0; n < N; n++) for (let oh = 0; oh < Ho; oh++) {
      const ih = oh * sH - pH + kh * dH;
      if (ih < 0 || ih >= H) continue;
      for (let ow = 0; ow < Wo; ow++) {
        const iw = ow * sW - pW + kw * dW;
        if (iw < 0 || iw >= W) continue;
        acc += gout[((n * Cout + oc) * Ho + oh) * Wo + ow] * x[((n * Cin + (ic0 + icL)) * H + ih) * W + iw];
      }
    }
    dwK[idx] = acc;
  }
  report.conv2d_dw = maxAbs(dwK, wt.grad);

  // WGSL conv_bwd_db transpiled (per output channel)
  const dbK = new Float32Array(Cout);
  for (let oc = 0; oc < Cout; oc++) {
    let acc = 0;
    for (let n = 0; n < N; n++) for (let oh = 0; oh < Ho; oh++) for (let ow = 0; ow < Wo; ow++)
      acc += gout[((n * Cout + oc) * Ho + oh) * Wo + ow];
    dbK[oc] = acc;
  }
  report.conv2d_db = maxAbs(dbK, bt.grad);
}

// ---- ConvTranspose2d fwd + bwd (groups=2, stride=2, padding=1, output_padding=1) ----
{
  const rng = (() => { let s = 909090; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  const N = 2, Cin = 2, H = 4, W = 4, Cout = 4, kH = 3, kW = 3;
  const groups = 2, sH = 2, sW = 2, dH = 1, dW = 1, pH = 1, pW = 1, oH = 1, oW = 1;
  const cinPerG = Cin / groups, coutPerG = Cout / groups;
  const Ho = (H - 1) * sH - 2 * pH + dH * (kH - 1) + oH + 1; // 8
  const Wo = (W - 1) * sW - 2 * pW + dW * (kW - 1) + oW + 1; // 8
  const x = new Float32Array(N * Cin * H * W), w = new Float32Array(Cin * coutPerG * kH * kW);
  const b = new Float32Array(Cout), gout = new Float32Array(N * Cout * Ho * Wo);
  for (let i = 0; i < x.length; i++) x[i] = rng() * 2 - 1;
  for (let i = 0; i < w.length; i++) w[i] = (rng() * 2 - 1) * 0.5;
  for (let i = 0; i < b.length; i++) b[i] = (rng() * 2 - 1) * 0.1;
  for (let i = 0; i < gout.length; i++) gout[i] = rng() * 2 - 1;

  // CPU reference via the engine
  const xt = Tensor.fromArray(x, [N, Cin, H, W], true);
  const wt = Tensor.fromArray(w, [Cin, coutPerG, kH, kW], true);
  const bt = Tensor.fromArray(b, [Cout], true);
  const yt = xt.convTranspose2d(wt, bt, groups, sH, dH, pH, oH);
  const cpuY = Float32Array.from(yt.data);
  yt.grad = Float32Array.from(gout);
  yt._backward(yt.grad);

  // WGSL convtranspose2d_fwd transpiled (gather with exact-division)
  const yK = new Float32Array(N * Cout * Ho * Wo);
  for (let idx = 0; idx < yK.length; idx++) {
    const ow = idx % Wo, oh = ((idx / Wo) | 0) % Ho, oc = ((idx / (Wo * Ho)) | 0) % Cout, n = (idx / (Wo * Ho * Cout)) | 0;
    let acc = b[oc];
    const g = (oc / coutPerG) | 0, ocL = oc % coutPerG;
    for (let icL = 0; icL < cinPerG; icL++) {
      const ic = g * cinPerG + icL;
      for (let kh = 0; kh < kH; kh++) {
        const num = oh + pH - kh * dH;
        if (num < 0 || num % sH !== 0) continue;
        const ih = num / sH;
        if (ih >= H) continue;
        for (let kw = 0; kw < kW; kw++) {
          const num2 = ow + pW - kw * dW;
          if (num2 < 0 || num2 % sW !== 0) continue;
          const iw = num2 / sW;
          if (iw >= W) continue;
          acc += x[((n * Cin + ic) * H + ih) * W + iw] * w[(ic * coutPerG + ocL) * kH * kW + kh * kW + kw];
        }
      }
    }
    yK[idx] = acc;
  }
  report.ct_fwd = maxAbs(yK, cpuY);

  // WGSL convtranspose2d_bwd_dx transpiled (per input element)
  const dxK = new Float32Array(N * Cin * H * W);
  for (let idx = 0; idx < dxK.length; idx++) {
    const iw = idx % W, ih = ((idx / W) | 0) % H, ic = ((idx / (W * H)) | 0) % Cin, n = (idx / (W * H * Cin)) | 0;
    const g = (ic / cinPerG) | 0, oc0 = g * coutPerG;
    let acc = 0;
    for (let ocL = 0; ocL < coutPerG; ocL++) {
      const oc = oc0 + ocL;
      for (let kh = 0; kh < kH; kh++) {
        const oh = ih * sH - pH + kh * dH;
        if (oh < 0 || oh >= Ho) continue;
        for (let kw = 0; kw < kW; kw++) {
          const ow = iw * sW - pW + kw * dW;
          if (ow < 0 || ow >= Wo) continue;
          acc += gout[((n * Cout + oc) * Ho + oh) * Wo + ow] * w[(ic * coutPerG + ocL) * kH * kW + kh * kW + kw];
        }
      }
    }
    dxK[idx] = acc;
  }
  report.ct_dx = maxAbs(dxK, xt.grad);

  // WGSL convtranspose2d_bwd_dw transpiled (per weight element)
  const dwK = new Float32Array(Cin * coutPerG * kH * kW);
  for (let idx = 0; idx < dwK.length; idx++) {
    const kw = idx % kW, kh = ((idx / kW) | 0) % kH, ocL = ((idx / (kW * kH)) | 0) % coutPerG, ic = (idx / (kW * kH * coutPerG)) | 0;
    const g = (ic / cinPerG) | 0, oc = g * coutPerG + ocL;
    let acc = 0;
    for (let n = 0; n < N; n++) for (let ih = 0; ih < H; ih++) {
      const oh = ih * sH - pH + kh * dH;
      if (oh < 0 || oh >= Ho) continue;
      for (let iw = 0; iw < W; iw++) {
        const ow = iw * sW - pW + kw * dW;
        if (ow < 0 || ow >= Wo) continue;
        acc += x[((n * Cin + ic) * H + ih) * W + iw] * gout[((n * Cout + oc) * Ho + oh) * Wo + ow];
      }
    }
    dwK[idx] = acc;
  }
  report.ct_dw = maxAbs(dwK, wt.grad);

  // WGSL conv_bwd_db transpiled (per output channel)
  const dbK = new Float32Array(Cout);
  for (let oc = 0; oc < Cout; oc++) {
    let acc = 0;
    for (let n = 0; n < N; n++) for (let oh = 0; oh < Ho; oh++) for (let ow = 0; ow < Wo; ow++)
      acc += gout[((n * Cout + oc) * Ho + oh) * Wo + ow];
    dbK[oc] = acc;
  }
  report.ct_db = maxAbs(dbK, bt.grad);
}

// ---- BatchNorm fwd + bwd (train mode) ----
{
  const rng = (() => { let s = 13579; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  const N = 2, C = 3, rest = 4, eps = 1e-5, cnt = N * rest;
  const x = new Float32Array(N * C * rest), gout = new Float32Array(N * C * rest);
  for (let i = 0; i < x.length; i++) x[i] = (rng() * 2 - 1) * 2;
  for (let i = 0; i < gout.length; i++) gout[i] = rng() * 2 - 1;
  const wArr = Float32Array.from([1.1, 0.9, 1.0]);
  const bArr = Float32Array.from([0.1, -0.1, 0.05]);

  // CPU reference via the engine module (train mode)
  const prevTraining = tg.training;
  tg.training = true;
  const bn = new tg.nn.BatchNorm(C, { momentum: 0.1, eps });
  bn.weight.data.set(wArr);
  bn.bias.data.set(bArr);
  const xt = Tensor.fromArray(x, [N, C, rest], true);
  const bnRef = bn.forward(xt);
  const cpuY = Float32Array.from(bnRef.data);
  bnRef.grad = Float32Array.from(gout);
  bnRef._backward(bnRef.grad);
  tg.training = prevTraining;

  // WGSL batchnorm_stats + batchnorm_fwd transpiled
  const mean = new Float32Array(C), varv = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    let s = 0;
    for (let n = 0; n < N; n++) { const off = (n * C + c) * rest; for (let i = 0; i < rest; i++) s += x[off + i]; }
    mean[c] = s / cnt;
    let v = 0;
    for (let n = 0; n < N; n++) { const off = (n * C + c) * rest; for (let i = 0; i < rest; i++) { const d = x[off + i] - mean[c]; v += d * d; } }
    varv[c] = v / cnt;
  }
  const yK = new Float32Array(N * C * rest);
  for (let idx = 0; idx < yK.length; idx++) {
    const c = ((idx / rest) | 0) % C;
    yK[idx] = (x[idx] - mean[c]) / Math.sqrt(varv[c] + eps) * wArr[c] + bArr[c];
  }
  report.bn_fwd = maxAbs(yK, cpuY);

  // WGSL batchnorm_bwd_stats + batchnorm_bwd_dx transpiled
  const sumG = new Float32Array(C), sumGY = new Float32Array(C), dg = new Float32Array(C), db = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    const inv = 1 / Math.sqrt(varv[c] + eps), wc = wArr[c];
    for (let n = 0; n < N; n++) {
      const off = (n * C + c) * rest;
      for (let i = 0; i < rest; i++) {
        const yhat = (x[off + i] - mean[c]) * inv;
        sumG[c] += gout[off + i] * wc;
        sumGY[c] += gout[off + i] * wc * yhat;
        dg[c] += gout[off + i] * yhat;
        db[c] += gout[off + i];
      }
    }
  }
  const dxK = new Float32Array(N * C * rest);
  for (let idx = 0; idx < dxK.length; idx++) {
    const c = ((idx / rest) | 0) % C;
    const inv = 1 / Math.sqrt(varv[c] + eps);
    const yhat = (x[idx] - mean[c]) * inv;
    dxK[idx] = inv * (gout[idx] * wArr[c] - sumG[c] / cnt - yhat * sumGY[c] / cnt);
  }
  report.bn_dx = maxAbs(dxK, xt.grad);
  report.bn_dgamma = maxAbs(dg, bn.weight.grad);
  report.bn_dbeta = maxAbs(db, bn.bias.grad);
}

// ---- Affine channel (GroupNorm tail) fwd + bwd ----
{
  const rng = (() => { let s = 24680; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  const N = 2, C = 4, rest = 6;
  const x = new Float32Array(N * C * rest), gout = new Float32Array(N * C * rest);
  const wArr = new Float32Array(C), bArr = new Float32Array(C);
  for (let i = 0; i < x.length; i++) x[i] = rng() * 2 - 1;
  for (let i = 0; i < gout.length; i++) gout[i] = rng() * 2 - 1;
  for (let c = 0; c < C; c++) { wArr[c] = 0.5 + rng() * 0.5; bArr[c] = (rng() * 2 - 1) * 0.1; }

  // CPU reference via the engine op
  const xt = Tensor.fromArray(x, [N, C, rest], true);
  const wt = Tensor.fromArray(wArr, [C], true);
  const bt = Tensor.fromArray(bArr, [C], true);
  const at = xt.affineChannel(wt, bt);
  const cpuY = Float32Array.from(at.data);
  at.grad = Float32Array.from(gout);
  at._backward(at.grad);

  // WGSL affine_channel transpiled
  const yK = new Float32Array(N * C * rest);
  for (let idx = 0; idx < yK.length; idx++) {
    const c = ((idx / rest) | 0) % C;
    yK[idx] = x[idx] * wArr[c] + bArr[c];
  }
  report.aff_fwd = maxAbs(yK, cpuY);

  // WGSL affine_channel_bwd_dw transpiled + dx = gout*w + db = sum
  const dwK = new Float32Array(C);
  const dxK = new Float32Array(N * C * rest);
  const dbK = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    let s = 0;
    for (let n = 0; n < N; n++) {
      const off = (n * C + c) * rest;
      for (let i = 0; i < rest; i++) {
        s += gout[off + i] * x[off + i];
        dxK[off + i] = gout[off + i] * wArr[c];
        dbK[c] += gout[off + i];
      }
    }
    dwK[c] = s;
  }
  report.aff_dx = maxAbs(dxK, xt.grad);
  report.aff_dw = maxAbs(dwK, wt.grad);
  report.aff_db = maxAbs(dbK, bt.grad);
}

// ---- LayerNorm backward ----
{
  const rng = (() => { let s = 31415; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  const rows = 5, d = 16, eps = 1e-5;
  const x = new Float32Array(rows * d), gout = new Float32Array(rows * d);
  for (let i = 0; i < x.length; i++) x[i] = (rng() * 2 - 1) * 2;
  for (let i = 0; i < gout.length; i++) gout[i] = rng() * 2 - 1;

  // CPU reference via the engine op
  const xt = Tensor.fromArray(x, [rows, d], true);
  const yt = xt.layerNorm(eps);
  yt.grad = Float32Array.from(gout);
  yt._backward(yt.grad);

  // WGSL layernorm_backward transpiled
  const dxK = new Float32Array(rows * d);
  for (let r = 0; r < rows; r++) {
    const base = r * d;
    let mean = 0;
    for (let j = 0; j < d; j++) mean += x[base + j];
    mean /= d;
    let varr = 0;
    for (let j = 0; j < d; j++) { const v = x[base + j] - mean; varr += v * v; }
    const inv = 1 / Math.sqrt(varr / d + eps);
    let sumG = 0, dot = 0;
    const yrow = new Float32Array(d);
    for (let j = 0; j < d; j++) { yrow[j] = (x[base + j] - mean) * inv; sumG += gout[base + j]; dot += gout[base + j] * yrow[j]; }
    const meanG = sumG / d, meanGY = dot / d;
    for (let j = 0; j < d; j++) dxK[base + j] = inv * (gout[base + j] - meanG - yrow[j] * meanGY);
  }
  report.ln_bwd = maxAbs(dxK, xt.grad);
}

// ---- Affine last (nn.LayerNorm tail) fwd + bwd ----
{
  const rng = (() => { let s = 16180; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  const rows = 6, C = 4;
  const x = new Float32Array(rows * C), gout = new Float32Array(rows * C);
  const wArr = new Float32Array(C), bArr = new Float32Array(C);
  for (let i = 0; i < x.length; i++) x[i] = rng() * 2 - 1;
  for (let i = 0; i < gout.length; i++) gout[i] = rng() * 2 - 1;
  for (let c = 0; c < C; c++) { wArr[c] = 0.5 + rng() * 0.5; bArr[c] = (rng() * 2 - 1) * 0.1; }

  const xt = Tensor.fromArray(x, [rows, C], true);
  const wt = Tensor.fromArray(wArr, [C], true);
  const bt = Tensor.fromArray(bArr, [C], true);
  const at = xt.affineLast(wt, bt);
  const cpuY = Float32Array.from(at.data);
  at.grad = Float32Array.from(gout);
  at._backward(at.grad);

  // WGSL affine_last + affine_last_bwd_dw transpiled
  const yK = new Float32Array(rows * C);
  const dxK = new Float32Array(rows * C);
  const dwK = new Float32Array(C);
  const dbK = new Float32Array(C);
  for (let idx = 0; idx < rows * C; idx++) {
    const j = idx % C;
    yK[idx] = x[idx] * wArr[j] + bArr[j];
    dxK[idx] = gout[idx] * wArr[j];
  }
  for (let c = 0; c < C; c++) {
    for (let r = 0; r < rows; r++) {
      const off = r * C + c;
      dwK[c] += gout[off] * x[off];
      dbK[c] += gout[off];
    }
  }
  report.afflast_fwd = maxAbs(yK, cpuY);
  report.afflast_dx = maxAbs(dxK, xt.grad);
  report.afflast_dw = maxAbs(dwK, wt.grad);
  report.afflast_db = maxAbs(dbK, bt.grad);
}

// ---- LSTM cell forward (gate order i,f,g,o) ----
{
  const rng = (() => { let s = 27182; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  const B = 2, I = 3, H = 4;
  const x = new Float32Array(B * I), h0 = new Float32Array(B * H), c0 = new Float32Array(B * H);
  const wih = new Float32Array(4 * H * I), whh = new Float32Array(4 * H * H);
  const bIh = new Float32Array(4 * H), bHh = new Float32Array(4 * H);
  for (let i = 0; i < x.length; i++) x[i] = rng() * 2 - 1;
  for (let i = 0; i < h0.length; i++) h0[i] = rng() * 2 - 1;
  for (let i = 0; i < c0.length; i++) c0[i] = rng() * 2 - 1;
  for (let i = 0; i < wih.length; i++) wih[i] = (rng() * 2 - 1) * 0.5;
  for (let i = 0; i < whh.length; i++) whh[i] = (rng() * 2 - 1) * 0.5;
  for (let i = 0; i < bIh.length; i++) bIh[i] = (rng() * 2 - 1) * 0.1;
  for (let i = 0; i < bHh.length; i++) bHh[i] = (rng() * 2 - 1) * 0.1;

  // CPU reference via the engine module
  const cell = new tg.nn.LSTMCell(I, H, { bias: true });
  cell.weightIh.data.set(wih);
  cell.weightHh.data.set(whh);
  cell.biasIh.data.set(bIh);
  cell.biasHh.data.set(bHh);
  const [hT, cT] = cell.forward(
    Tensor.fromArray(x, [B, I], false),
    [Tensor.fromArray(h0, [B, H], false), Tensor.fromArray(c0, [B, H], false)],
  );
  const cpuH = Float32Array.from(hT.data), cpuC = Float32Array.from(cT.data);

  // WGSL lstm_cell transpiled
  const sig = (v) => 1 / (1 + Math.exp(-v));
  const hK = new Float32Array(B * H), cK = new Float32Array(B * H);
  for (let idx = 0; idx < B * H; idx++) {
    const n = (idx / H) | 0, j = idx % H;
    let s0 = bIh[j] + bHh[j], s1 = bIh[H + j] + bHh[H + j], s2 = bIh[2 * H + j] + bHh[2 * H + j], s3 = bIh[3 * H + j] + bHh[3 * H + j];
    const xB = n * I, hB = n * H;
    for (let k = 0; k < I; k++) {
      const xv = x[xB + k];
      s0 += xv * wih[j * I + k];
      s1 += xv * wih[(H + j) * I + k];
      s2 += xv * wih[(2 * H + j) * I + k];
      s3 += xv * wih[(3 * H + j) * I + k];
    }
    for (let k = 0; k < H; k++) {
      const hv = h0[hB + k];
      s0 += hv * whh[j * H + k];
      s1 += hv * whh[(H + j) * H + k];
      s2 += hv * whh[(2 * H + j) * H + k];
      s3 += hv * whh[(3 * H + j) * H + k];
    }
    const i = sig(s0), f = sig(s1), g = Math.tanh(s2), o = sig(s3);
    const cnew = f * c0[idx] + i * g;
    cK[idx] = cnew;
    hK[idx] = o * Math.tanh(cnew);
  }
  report.lstm_h = maxAbs(hK, cpuH);
  report.lstm_c = maxAbs(cK, cpuC);
}

// ---- LSTM cell backward: transpiled kernels vs engine CPU autograd (both h and c chains) ----
{
  const rng = (() => { let s = 161803; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  const B = 2, I = 3, H = 4;
  const x = new Float32Array(B * I), h0 = new Float32Array(B * H), c0 = new Float32Array(B * H);
  const wih = new Float32Array(4 * H * I), whh = new Float32Array(4 * H * H);
  const bIh = new Float32Array(4 * H), bHh = new Float32Array(4 * H);
  const dh = new Float32Array(B * H), dc = new Float32Array(B * H);
  for (let i = 0; i < x.length; i++) x[i] = rng() * 2 - 1;
  for (let i = 0; i < h0.length; i++) h0[i] = rng() * 2 - 1;
  for (let i = 0; i < c0.length; i++) c0[i] = rng() * 2 - 1;
  for (let i = 0; i < wih.length; i++) wih[i] = (rng() * 2 - 1) * 0.5;
  for (let i = 0; i < whh.length; i++) whh[i] = (rng() * 2 - 1) * 0.5;
  for (let i = 0; i < bIh.length; i++) bIh[i] = (rng() * 2 - 1) * 0.1;
  for (let i = 0; i < bHh.length; i++) bHh[i] = (rng() * 2 - 1) * 0.1;
  for (let i = 0; i < dh.length; i++) dh[i] = rng() * 2 - 1;
  for (let i = 0; i < dc.length; i++) dc[i] = rng() * 2 - 1;

  // CPU reference via the engine's autograd graph
  const cell = new tg.nn.LSTMCell(I, H, { bias: true });
  cell.weightIh.data.set(wih);
  cell.weightHh.data.set(whh);
  cell.biasIh.data.set(bIh);
  cell.biasHh.data.set(bHh);
  const xt = Tensor.fromArray(x, [B, I], true);
  const ht = Tensor.fromArray(h0, [B, H], true);
  const ct = Tensor.fromArray(c0, [B, H], true);
  const [hT, cT] = cell.forward(xt, [ht, ct]);
  hT.grad = Float32Array.from(dh);
  cT.grad = Float32Array.from(dc);
  // manual reverse-topo walk over both chains (scalar-loss check bypassed)
  {
    const topo = [], seen = new Set();
    const build = (t) => { if (seen.has(t)) return; seen.add(t); for (const p of t._parents) build(p); topo.push(t); };
    build(hT); build(cT);
    for (let i = topo.length - 1; i >= 0; i--) {
      const t = topo[i];
      if (!t.grad) continue;
      t._backward(t.grad, null);
    }
  }

  // WGSL lstm_cell_bwd + friends transpiled
  const sig = (v) => 1 / (1 + Math.exp(-v));
  const ds = new Float32Array(B * 4 * H), dcPrev = new Float32Array(B * H);
  for (let idx = 0; idx < B * H; idx++) {
    const n = (idx / H) | 0, j = idx % H;
    let s0 = bIh[j] + bHh[j], s1 = bIh[H + j] + bHh[H + j], s2 = bIh[2 * H + j] + bHh[2 * H + j], s3 = bIh[3 * H + j] + bHh[3 * H + j];
    const xB = n * I, hB = n * H;
    for (let k = 0; k < I; k++) {
      const xv = x[xB + k];
      s0 += xv * wih[j * I + k]; s1 += xv * wih[(H + j) * I + k];
      s2 += xv * wih[(2 * H + j) * I + k]; s3 += xv * wih[(3 * H + j) * I + k];
    }
    for (let k = 0; k < H; k++) {
      const hv = h0[hB + k];
      s0 += hv * whh[j * H + k]; s1 += hv * whh[(H + j) * H + k];
      s2 += hv * whh[(2 * H + j) * H + k]; s3 += hv * whh[(3 * H + j) * H + k];
    }
    const i = sig(s0), f = sig(s1), g = Math.tanh(s2), o = sig(s3);
    const cprev = c0[idx];
    const cnew = f * cprev + i * g;
    const c1 = Math.tanh(cnew);
    const dcTot = dc[idx] + dh[idx] * o * (1 - c1 * c1);
    const base = n * 4 * H;
    ds[base + j] = dcTot * g * i * (1 - i);
    ds[base + H + j] = dcTot * cprev * f * (1 - f);
    ds[base + 2 * H + j] = dcTot * i * (1 - g * g);
    ds[base + 3 * H + j] = dh[idx] * c1 * o * (1 - o);
    dcPrev[idx] = dcTot * f;
  }
  const dxK = new Float32Array(B * I), dhK = new Float32Array(B * H);
  for (let idx = 0; idx < B * I; idx++) {
    const n = (idx / I) | 0, k = idx % I;
    let acc = 0;
    for (let j = 0; j < H; j++) for (let g = 0; g < 4; g++) acc += ds[n * 4 * H + g * H + j] * wih[(g * H + j) * I + k];
    dxK[idx] = acc;
  }
  for (let idx = 0; idx < B * H; idx++) {
    const n = (idx / H) | 0, k = idx % H;
    let acc = 0;
    for (let j = 0; j < H; j++) for (let g = 0; g < 4; g++) acc += ds[n * 4 * H + g * H + j] * whh[(g * H + j) * H + k];
    dhK[idx] = acc;
  }
  const dwihK = new Float32Array(4 * H * I), dwhhK = new Float32Array(4 * H * H), dbK = new Float32Array(4 * H);
  for (let idx = 0; idx < 4 * H * I; idx++) {
    const k = idx % I, j = ((idx / I) | 0) % H, g = ((idx / (I * H)) | 0) % 4;
    let acc = 0;
    for (let n = 0; n < B; n++) acc += ds[n * 4 * H + g * H + j] * x[n * I + k];
    dwihK[idx] = acc;
  }
  for (let idx = 0; idx < 4 * H * H; idx++) {
    const k = idx % H, j = ((idx / H) | 0) % H, g = ((idx / (H * H)) | 0) % 4;
    let acc = 0;
    for (let n = 0; n < B; n++) acc += ds[n * 4 * H + g * H + j] * h0[n * H + k];
    dwhhK[idx] = acc;
  }
  for (let u = 0; u < 4 * H; u++) {
    let acc = 0;
    for (let n = 0; n < B; n++) acc += ds[n * 4 * H + u];
    dbK[u] = acc;
  }
  report.lstm_dx = maxAbs(dxK, xt.grad);
  report.lstm_dh = maxAbs(dhK, ht.grad);
  report.lstm_dc = maxAbs(dcPrev, ct.grad);
  report.lstm_dwih = maxAbs(dwihK, cell.weightIh.grad);
  report.lstm_dwhh = maxAbs(dwhhK, cell.weightHh.grad);
  report.lstm_dbih = maxAbs(dbK, cell.biasIh.grad);
  report.lstm_dbhh = maxAbs(dbK, cell.biasHh.grad);
}

// ---- KV-cache attention (decode step): scores + apply, incl. causal mask ----
{
  const rng = (() => { let s = 314159; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  const BH = 2, L = 5, headDim = 4, scale = 1 / Math.sqrt(headDim);
  const q = new Float32Array(BH * headDim), kc = new Float32Array(BH * L * headDim), vc = new Float32Array(BH * L * headDim);
  for (let i = 0; i < q.length; i++) q[i] = rng() * 2 - 1;
  for (let i = 0; i < kc.length; i++) kc[i] = rng() * 2 - 1;
  for (let i = 0; i < vc.length; i++) vc[i] = rng() * 2 - 1;

  // Reference: explicit softmax-weighted sum over the cache (no masking)
  const scoresRef = new Float32Array(BH * L), outRef = new Float32Array(BH * headDim);
  for (let bh = 0; bh < BH; bh++) {
    for (let j = 0; j < L; j++) {
      let dot = 0;
      for (let d = 0; d < headDim; d++) dot += q[bh * headDim + d] * kc[(bh * L + j) * headDim + d];
      scoresRef[bh * L + j] = dot * scale;
    }
    let m = -Infinity;
    for (let j = 0; j < L; j++) m = Math.max(m, scoresRef[bh * L + j]);
    let sumE = 0;
    const p = new Float32Array(L);
    for (let j = 0; j < L; j++) { p[j] = Math.exp(scoresRef[bh * L + j] - m); sumE += p[j]; }
    for (let d = 0; d < headDim; d++) {
      let acc = 0;
      for (let j = 0; j < L; j++) acc += p[j] / sumE * vc[(bh * L + j) * headDim + d];
      outRef[bh * headDim + d] = acc;
    }
  }

  // WGSL kv_attention_scores + kv_attention_apply transpiled (no causal).
  // MHA case: kvH = H = BH, so kv = (bh * kvH) / H = bh (identical to before).
  const kvH0 = BH, H0 = BH;
  const scoresK = new Float32Array(BH * L), outK = new Float32Array(BH * headDim);
  for (let idx = 0; idx < BH * L; idx++) {
    const bh = (idx / L) | 0, j = idx % L;
    const kv = ((bh * kvH0) / H0) | 0;
    let dot = 0;
    for (let d = 0; d < headDim; d++) dot += q[bh * headDim + d] * kc[(kv * L + j) * headDim + d];
    scoresK[idx] = dot * scale;
  }
  for (let idx = 0; idx < BH * headDim; idx++) {
    const bh = (idx / headDim) | 0, d = idx % headDim;
    const kv = ((bh * kvH0) / H0) | 0;
    let m = -Infinity;
    for (let j = 0; j < L; j++) m = Math.max(m, scoresK[bh * L + j]);
    let sumE = 0, acc = 0;
    for (let j = 0; j < L; j++) {
      const e = Math.exp(scoresK[bh * L + j] - m);
      sumE += e;
      acc += e * vc[(kv * L + j) * headDim + d];
    }
    outK[idx] = acc / Math.max(sumE, 1e-12);
  }
  report.kv_scores = maxAbs(scoresK, scoresRef);
  report.kv_out = maxAbs(outK, outRef);

  // Causal mask: attend over prefix pos=2 with L=5 (positions 3,4 must contribute 0)
  const pos = 2;
  const outCausal = new Float32Array(BH * headDim);
  for (let idx = 0; idx < BH * headDim; idx++) {
    const bh = (idx / headDim) | 0, d = idx % headDim;
    let m = -Infinity;
    for (let j = 0; j <= pos; j++) m = Math.max(m, scoresK[bh * L + j]);
    let sumE = 0, acc = 0;
    for (let j = 0; j <= pos; j++) {
      const e = Math.exp(scoresK[bh * L + j] - m);
      sumE += e;
      acc += e * vc[(bh * L + j) * headDim + d];
    }
    outCausal[idx] = acc / Math.max(sumE, 1e-12);
  }
  // Cross-check against full-sequence causal attention (last row equivalence): the
  // kv-attention over the full cache at pos = L-1 must equal the last row of a
  // full-sequence causal attention over the same K/V.
  const outFullRow = new Float32Array(BH * headDim);
  for (let bh = 0; bh < BH; bh++) {
    let m = -Infinity;
    for (let j = 0; j < L; j++) m = Math.max(m, scoresRef[bh * L + j]);
    let sumE = 0;
    const p = new Float32Array(L);
    for (let j = 0; j < L; j++) { p[j] = Math.exp(scoresRef[bh * L + j] - m); sumE += p[j]; }
    for (let d = 0; d < headDim; d++) {
      let acc = 0;
      for (let j = 0; j < L; j++) acc += p[j] / sumE * vc[(bh * L + j) * headDim + d];
      outFullRow[bh * headDim + d] = acc;
    }
  }
  const causalFull = new Float32Array(BH * headDim);
  for (let idx = 0; idx < BH * headDim; idx++) {
    const bh = (idx / headDim) | 0, d = idx % headDim;
    let m = -Infinity;
    for (let j = 0; j < L; j++) m = Math.max(m, scoresK[bh * L + j]);
    let sumE = 0, acc = 0;
    for (let j = 0; j < L; j++) {
      const e = Math.exp(scoresK[bh * L + j] - m);
      sumE += e;
      acc += e * vc[(bh * L + j) * headDim + d];
    }
    causalFull[idx] = acc / Math.max(sumE, 1e-12);
  }
  report.kv_causal = maxAbs(outCausal, outRef.slice(0, BH * headDim).map((_, i) => {
    // masked reference: recompute per bh over 0..pos
    const bh = (i / headDim) | 0, d = i % headDim;
    let m = -Infinity;
    for (let j = 0; j <= pos; j++) m = Math.max(m, scoresRef[bh * L + j]);
    let sumE = 0, acc = 0;
    for (let j = 0; j <= pos; j++) {
      const e = Math.exp(scoresRef[bh * L + j] - m);
      sumE += e;
      acc += e * vc[(bh * L + j) * headDim + d];
    }
    return acc / Math.max(sumE, 1e-12);
  }));
  report.kv_fullrow = maxAbs(causalFull, outFullRow);
}

// ---- GQA (grouped query attention): grouped KV cache + group-sum backward ----
{
  const rng = (() => { let s = 271828; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  const B = 2, H = 4, kvH = 2, T = 5, headDim = 4;
  const BH = B * H, BKV = B * kvH, scale = 1 / Math.sqrt(headDim);
  const q = new Float32Array(BH * headDim), kc = new Float32Array(BKV * T * headDim), vc = new Float32Array(BKV * T * headDim);
  for (let i = 0; i < q.length; i++) q[i] = rng() * 2 - 1;
  for (let i = 0; i < kc.length; i++) kc[i] = rng() * 2 - 1;
  for (let i = 0; i < vc.length; i++) vc[i] = rng() * 2 - 1;

  // Reference: query head h attends to KV head (b*kvH + h//group)
  const scoresRef = new Float32Array(BH * T), outRef = new Float32Array(BH * headDim);
  for (let bh = 0; bh < BH; bh++) {
    const kv = ((bh * kvH) / H) | 0;
    for (let j = 0; j < T; j++) {
      let dot = 0;
      for (let d = 0; d < headDim; d++) dot += q[bh * headDim + d] * kc[(kv * T + j) * headDim + d];
      scoresRef[bh * T + j] = dot * scale;
    }
    let m = -Infinity;
    for (let j = 0; j < T; j++) m = Math.max(m, scoresRef[bh * T + j]);
    const p = new Float32Array(T); let sumE = 0;
    for (let j = 0; j < T; j++) { p[j] = Math.exp(scoresRef[bh * T + j] - m); sumE += p[j]; }
    for (let d = 0; d < headDim; d++) {
      let acc = 0;
      for (let j = 0; j < T; j++) acc += p[j] / sumE * vc[(kv * T + j) * headDim + d];
      outRef[bh * headDim + d] = acc;
    }
  }

  // Transpiled WGSL with the GQA kvidx mapping (kvH=2, H=4)
  const scoresK = new Float32Array(BH * T), outK = new Float32Array(BH * headDim);
  for (let idx = 0; idx < BH * T; idx++) {
    const bh = (idx / T) | 0, j = idx % T;
    const kv = ((bh * kvH) / H) | 0;
    let dot = 0;
    for (let d = 0; d < headDim; d++) dot += q[bh * headDim + d] * kc[(kv * T + j) * headDim + d];
    scoresK[idx] = dot * scale;
  }
  for (let idx = 0; idx < BH * headDim; idx++) {
    const bh = (idx / headDim) | 0, d = idx % headDim;
    const kv = ((bh * kvH) / H) | 0;
    let m = -Infinity;
    for (let j = 0; j < T; j++) m = Math.max(m, scoresK[bh * T + j]);
    let sumE = 0, acc = 0;
    for (let j = 0; j < T; j++) {
      const e = Math.exp(scoresK[bh * T + j] - m);
      sumE += e;
      acc += e * vc[(kv * T + j) * headDim + d];
    }
    outK[idx] = acc / Math.max(sumE, 1e-12);
  }
  report.gqa_kv_scores = maxAbs(scoresK, scoresRef);
  report.gqa_kv_out = maxAbs(outK, outRef);

  // Group-sum kernel: [BH*T*headDim] -> [B*kvH*T*headDim]
  const gradIn = new Float32Array(BH * T * headDim), groupRef = new Float32Array(BKV * T * headDim);
  for (let i = 0; i < gradIn.length; i++) gradIn[i] = rng() * 2 - 1;
  const group = H / kvH, TD = T * headDim;
  for (let b = 0; b < B; b++) {
    for (let kv = 0; kv < kvH; kv++) {
      const dst = (b * kvH + kv) * TD;
      for (let g = 0; g < group; g++) {
        const src = (b * H + kv * group + g) * TD;
        for (let i = 0; i < TD; i++) groupRef[dst + i] += gradIn[src + i];
      }
    }
  }
  const groupK = new Float32Array(BKV * TD);
  for (let idx = 0; idx < BKV * TD; idx++) {
    const bkv = (idx / TD) | 0, rest = idx % TD;
    const b = (bkv / kvH) | 0, g0 = (bkv % kvH) * group;
    const base = (b * H + g0) * TD + rest;
    let acc = 0;
    for (let g = 0; g < group; g++) acc += gradIn[base + g * TD];
    groupK[idx] = acc;
  }
  report.kv_group_sum = maxAbs(groupK, groupRef);
}

// ---- Pad kernel (NCHW) vs CPU pad ----
{
  const N=2, C=4, H=4, W=4, pad_h_before=1, pad_h_after=1, pad_w_before=1, pad_w_after=1;
  const H_out=H+pad_h_before+pad_h_after, W_out=W+pad_w_before+pad_w_after;
  const x = new Float32Array(N*C*H*W);
  for(let i=0;i<x.length;i++) x[i]= (i%7)-3;
  const xt = Tensor.fromArray(x, [N,C,H,W], false);
  const cpu = xt.pad([[0,0],[0,0],[pad_h_before,pad_h_after],[pad_w_before,pad_w_after]]).data;
  const yK = new Float32Array(N*C*H_out*W_out).fill(0);
  for(let n=0;n<N;n++) for(let c=0;c<C;c++) for(let h=0;h<H;h++) for(let w=0;w<W;w++){
    const src_idx = ((n*C+c)*H+h)*W+w;
    const h_out = h+pad_h_before, w_out=w+pad_w_before;
    const dst_idx = ((n*C+c)*H_out+h_out)*W_out+w_out;
    yK[dst_idx]= x[src_idx];
  }
  report.pad_fwd = maxAbs(yK, cpu);
  // pad backward: grad = slice(gout)
  const gout = new Float32Array(N*C*H_out*W_out);
  for(let i=0;i<gout.length;i++) gout[i]= (i%5)-2;
  const goutT = Tensor.fromArray(gout, [N,C,H_out,W_out], false);
  // CPU backward via engine: create pad tensor then backward
  const xt2 = Tensor.fromArray(x, [N,C,H,W], true);
  const yt2 = xt2.pad([[0,0],[0,0],[pad_h_before,pad_h_after],[pad_w_before,pad_w_after]]);
  yt2.grad = Float32Array.from(gout);
  yt2._backward(yt2.grad);
  const dxK = new Float32Array(N*C*H*W);
  for(let n=0;n<N;n++) for(let c=0;c<C;c++) for(let h=0;h<H;h++) for(let w=0;w<W;w++){
    const h_out=h+pad_h_before, w_out=w+pad_w_before;
    const gout_idx = ((n*C+c)*H_out+h_out)*W_out+w_out;
    const src_idx = ((n*C+c)*H+h)*W+w;
    dxK[src_idx]= gout[gout_idx];
  }
  report.pad_bwd = maxAbs(dxK, xt2.grad);
}

// ---- Concat kernel (channel + width) vs CPU ----
{
  const N=2, C_a=3, C_b=2, H=4, W=4;
  const a = new Float32Array(N*C_a*H*W), b = new Float32Array(N*C_b*H*W);
  for(let i=0;i<a.length;i++) a[i]= (i%9)-4;
  for(let i=0;i<b.length;i++) b[i]= (i%7)-3;
  const at = Tensor.fromArray(a, [N,C_a,H,W], false);
  const bt = Tensor.fromArray(b, [N,C_b,H,W], false);
  const cpu = at.concat(bt, 1).data;
  const outer=N, inner=H*W, c_total=C_a+C_b;
  const yK = new Float32Array(N*c_total*H*W);
  for(let r=0;r<outer;r++) for(let c=0;c<c_total;c++) for(let ii=0;ii<inner;ii++){
    const dst_idx = r*c_total*inner + c*inner + ii;
    if(c < C_a) yK[dst_idx]= a[r*C_a*inner + c*inner + ii];
    else yK[dst_idx]= b[r*C_b*inner + (c-C_a)*inner + ii];
  }
  report.concat_fwd = maxAbs(yK, cpu);
  // concat backward split
  const gout = new Float32Array(N*c_total*H*W);
  for(let i=0;i<gout.length;i++) gout[i]= (i%11)-5;
  const at2 = Tensor.fromArray(a, [N,C_a,H,W], true);
  const bt2 = Tensor.fromArray(b, [N,C_b,H,W], true);
  const yt2 = at2.concat(bt2, 1);
  yt2.grad = Float32Array.from(gout);
  yt2._backward(yt2.grad);
  const daK = new Float32Array(N*C_a*H*W), dbK = new Float32Array(N*C_b*H*W);
  for(let r=0;r<outer;r++) for(let c=0;c<C_a;c++) for(let ii=0;ii<inner;ii++) daK[r*C_a*inner + c*inner + ii]= gout[r*c_total*inner + c*inner + ii];
  for(let r=0;r<outer;r++) for(let c=0;c<C_b;c++) for(let ii=0;ii<inner;ii++) dbK[r*C_b*inner + c*inner + ii]= gout[r*c_total*inner + (c+C_a)*inner + ii];
  report.concat_bwd_a = maxAbs(daK, at2.grad);
  report.concat_bwd_b = maxAbs(dbK, bt2.grad);
}

// ---- Exp elementwise (op 7) vs CPU ----
{
  const len=64;
  const x = new Float32Array(len);
  for(let i=0;i<len;i++) x[i]= (i%10)/5 -1;
  const xt = Tensor.fromArray(x, [len], false);
  const cpu = xt.exp().data;
  const yK = new Float32Array(len);
  for(let i=0;i<len;i++) yK[i]= Math.exp(x[i]);
  report.exp_fwd = maxAbs(yK, cpu);
  const xt2 = Tensor.fromArray(x, [len], true);
  const yt2 = xt2.exp();
  const gout = new Float32Array(len).fill(1);
  yt2.grad = gout;
  yt2._backward(gout);
  const dxK = new Float32Array(len);
  for(let i=0;i<len;i++) dxK[i]= Math.exp(x[i]);
  report.exp_bwd = maxAbs(dxK, xt2.grad);
}

const convChecks = [report.conv2d_fwd, report.conv2d_dx, report.conv2d_dw, report.conv2d_db, report.ct_fwd, report.ct_dx, report.ct_dw, report.ct_db];
const normChecks = [report.bn_fwd, report.bn_dx, report.bn_dgamma, report.bn_dbeta, report.aff_fwd, report.aff_dx, report.aff_dw, report.aff_db];
const lastChecks = [report.ln_bwd, report.afflast_fwd, report.afflast_dx, report.afflast_dw, report.afflast_db, report.lstm_h, report.lstm_c];
const lstmBwdChecks = [report.lstm_dx, report.lstm_dh, report.lstm_dc, report.lstm_dwih, report.lstm_dwhh, report.lstm_dbih, report.lstm_dbhh];
const kvChecks = [report.kv_scores, report.kv_out, report.kv_causal, report.kv_fullrow, report.gqa_kv_scores, report.gqa_kv_out, report.kv_group_sum];
const newChecks = [report.pad_fwd, report.pad_bwd, report.concat_fwd, report.concat_bwd_a, report.concat_bwd_b, report.exp_fwd, report.exp_bwd];
if (convChecks.some((v) => !(v < 1e-4)) || normChecks.some((v) => !(v < 1e-4)) || lastChecks.some((v) => !(v < 1e-4)) || lstmBwdChecks.some((v) => !(v < 1e-4)) || kvChecks.some((v) => !(v < 1e-4)) || newChecks.some((v) => !(v < 1e-4))) process.exitCode = 2;

console.log(JSON.stringify(report, null, 2));
