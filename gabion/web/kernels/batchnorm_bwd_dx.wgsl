// batchnorm_bwd_dx.wgsl — per-element input gradient, one thread per element.
// dx[idx] = inv[c] * (gout[idx]*gamma_c - sumG[c]/cnt - xhat*sumGY[c]/cnt),
// c = (idx/rest)%C, xhat = (x[idx]-mean[c])*inv[c], cnt = N*rest.
// sumG/sumGY come from batchnorm_bwd_stats.

struct Params {
  N: u32, C: u32, rest: u32, eps: f32, hasWeight: u32, hasBias: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gout: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> meanBuf: array<f32>;
@group(0) @binding(5) var<storage, read> varBuf: array<f32>;
@group(0) @binding(6) var<storage, read> sumGBuf: array<f32>;
@group(0) @binding(7) var<storage, read> sumGYBuf: array<f32>;
@group(0) @binding(8) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.N * params.C * params.rest;
  if (idx >= total) { return; }
  let c = (idx / params.rest) % params.C;
  let inv = 1.0 / sqrt(varBuf[c] + params.eps);
  let xhat = (x[idx] - meanBuf[c]) * inv;
  let wc = params.hasWeight != 0u ? gamma[c] : 1.0;
  let cnt = f32(params.N * params.rest);
  dx[idx] = inv * (gout[idx] * wc - sumGBuf[c] / cnt - xhat * sumGYBuf[c] / cnt);
}
