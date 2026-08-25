// affine_channel_bwd_dw.wgsl — channel-affine weight gradient, one thread per channel.
// dw[c] = sum over (n, i) of gout[(n*C+c)*rest + i] * x[(n*C+c)*rest + i].
// Pure gather — no write races.

struct Params {
  N: u32, C: u32, rest: u32, hasBias: u32,
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
  for (var n = 0u; n < params.N; n++) {
    let off = (n * params.C + c) * params.rest;
    for (var i = 0u; i < params.rest; i++) {
      acc += gout[off + i] * x[off + i];
    }
  }
  dw[c] = acc;
}
