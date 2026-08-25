// affine_last.wgsl — per-element last-dim affine, one thread per element.
// y[idx] = x[idx] * w[j] + b[j], j = idx % C. Input [rows, C]. Used by nn.LayerNorm.
// b optional (hasBias).

struct Params {
  rows: u32, C: u32, hasBias: u32, _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.rows * params.C;
  if (idx >= total) { return; }
  let j = idx % params.C;
  var y: f32 = x[idx] * w[j];
  if (params.hasBias != 0u) { y += b[j]; }
  out[idx] = y;
}
