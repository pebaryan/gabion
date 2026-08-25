// batchnorm_bwd_stats.wgsl — per-channel backward stats, one thread per channel.
// Computes, for each channel c:
//   sumG   = sum(gout * gamma_c)            (unscaled x-grad sum)
//   sumGY  = sum(gout * gamma_c * xhat)     (xhat-weighted x-grad sum)
//   dgamma = sum(gout * xhat)
//   dbeta  = sum(gout)
// over all (n, i). xhat = (x - mean[c]) * inv[c], inv[c] = 1/sqrt(var[c]+eps).

struct Params {
  N: u32, C: u32, rest: u32, eps: f32, hasWeight: u32, hasBias: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gout: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> meanBuf: array<f32>;
@group(0) @binding(5) var<storage, read> varBuf: array<f32>;
@group(0) @binding(6) var<storage, read_write> sumGBuf: array<f32>;
@group(0) @binding(7) var<storage, read_write> sumGYBuf: array<f32>;
@group(0) @binding(8) var<storage, read_write> dgammaBuf: array<f32>;
@group(0) @binding(9) var<storage, read_write> dbetaBuf: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= params.C) { return; }
  let inv = 1.0 / sqrt(varBuf[c] + params.eps);
  let mean = meanBuf[c];
  let wc = params.hasWeight != 0u ? gamma[c] : 1.0;

  var sumG: f32 = 0.0;
  var sumGY: f32 = 0.0;
  var dg: f32 = 0.0;
  var db: f32 = 0.0;
  for (var n = 0u; n < params.N; n++) {
    let off = (n * params.C + c) * params.rest;
    for (var i = 0u; i < params.rest; i++) {
      let xhat = (x[off + i] - mean) * inv;
      let gv = gout[off + i];
      sumG += gv * wc;
      sumGY += gv * wc * xhat;
      dg += gv * xhat;
      db += gv;
    }
  }
  sumGBuf[c] = sumG;
  sumGYBuf[c] = sumGY;
  dgammaBuf[c] = dg;
  dbetaBuf[c] = db;
}
