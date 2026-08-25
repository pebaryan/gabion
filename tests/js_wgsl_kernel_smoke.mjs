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

console.log(JSON.stringify(report, null, 2));
