// affine_last_bwd_dw.wgsl — last-dim affine weight gradient, one thread per channel.
// dw[c] = sum over rows r of gout[r*C + c] * x[r*C + c]. Pure gather — no write races.

struct Params {
  rows: u32, C: u32, _pad: u32, _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gout: array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= params.C) { return; }
  var acc: f32 = 0.0;
  for (var r = 0u; r < params.rows; r++) {
    let off = r * params.C + c;
    acc += gout[off] * x[off];
  }
  dw[c] = acc;
}
