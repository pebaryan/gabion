// batchnorm_fwd.wgsl — per-element apply, one thread per element.
// y[idx] = (x[idx] - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c], c = (idx/rest)%C.
// Works for train (mean/var from batchnorm_stats) and eval (mean/var = running stats).
// gamma/beta optional (hasWeight/hasBias).

struct Params {
  N: u32, C: u32, rest: u32, eps: f32, hasWeight: u32, hasBias: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> meanBuf: array<f32>;
@group(0) @binding(3) var<storage, read> varBuf: array<f32>;
@group(0) @binding(4) var<storage, read> gamma: array<f32>;
@group(0) @binding(5) var<storage, read> beta: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.N * params.C * params.rest;
  if (idx >= total) { return; }
  let c = (idx / params.rest) % params.C;
  let inv = 1.0 / sqrt(varBuf[c] + params.eps);
  var y: f32 = (x[idx] - meanBuf[c]) * inv;
  if (params.hasWeight != 0u) { y *= gamma[c]; }
  if (params.hasBias != 0u) { y += beta[c]; }
  out[idx] = y;
}
