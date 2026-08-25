// affine_channel.wgsl — per-element NCHW channel affine, one thread per element.
// y[idx] = x[idx] * w[c] + b[c], c = (idx/rest)%C. Used by GroupNorm.
// b optional (hasBias).

struct Params {
  N: u32, C: u32, rest: u32, hasBias: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.N * params.C * params.rest;
  if (idx >= total) { return; }
  let c = (idx / params.rest) % params.C;
  var y: f32 = x[idx] * w[c];
  if (params.hasBias != 0u) { y += b[c]; }
  out[idx] = y;
}
