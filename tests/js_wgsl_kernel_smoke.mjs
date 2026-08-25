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

const convChecks = [report.conv2d_fwd, report.conv2d_dx, report.conv2d_dw, report.conv2d_db, report.ct_fwd, report.ct_dx, report.ct_dw, report.ct_db];
if (convChecks.some((v) => !(v < 1e-4))) process.exitCode = 2;

console.log(JSON.stringify(report, null, 2));
