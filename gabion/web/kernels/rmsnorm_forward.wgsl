// rmsnorm_forward.wgsl — RMSNorm with optional weight
struct Params {
  rows: u32,
  d: u32,
  eps: f32,
  hasWeight: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) { return; }
  let d = params.d;
  let base = row * d;
  var sum: f32 = 0.0;
  for (var j: u32 = 0u; j < d; j++) {
    let v = x[base + j];
    sum += v * v;
  }
  let inv = 1.0 / sqrt(sum / f32(d) + params.eps);
  for (var j: u32 = 0u; j < d; j++) {
    var v = x[base + j] * inv;
    if (params.hasWeight == 1u) {
      v *= w[j];
    }
    out[base + j] = v;
  }
}
