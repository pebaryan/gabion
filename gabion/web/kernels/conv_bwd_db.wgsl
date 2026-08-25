// conv_bwd_db.wgsl — conv / conv_transpose bias gradient, one thread per output channel.
// db[oc] = sum_{n,oh,ow} gout[n,oc,oh,ow]. Pure gather — no write races.
// Shared by conv2d and convtranspose2d (identical math).

struct Params {
  N: u32, Cin: u32, H: u32, W: u32,
  Cout: u32, Ho: u32, Wo: u32, kH: u32,
  kW: u32, groups: u32, cinPerG: u32, coutPerG: u32,
  sH: u32, sW: u32, dH: u32, dW: u32,
  pH: u32, pW: u32, hasBias: u32, _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gout: array<f32>;
@group(0) @binding(2) var<storage, read_write> db: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let oc = gid.x;
  if (oc >= params.Cout) { return; }
  let HoWo = params.Ho * params.Wo;
  var acc: f32 = 0.0;
  for (var n = 0u; n < params.N; n++) {
    let gN = (n * params.Cout + oc) * HoWo;
    for (var i = 0u; i < HoWo; i++) {
      acc += gout[gN + i];
    }
  }
  db[oc] = acc;
}
