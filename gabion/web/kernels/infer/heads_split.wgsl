// heads_split.wgsl — [B*T, D] -> [B*H_out, T, headDim] with optional bias and GQA expand
// D = H_in*headDim, groups = H_out/H_in
struct Params {
  B: u32,
  T: u32,
  H_in: u32,
  H_out: u32,
  headDim: u32,
  hasBias: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.B * params.H_out * params.T * params.headDim;
  let idx = gid.x;
  if (idx >= total) { return; }
  let Hd = params.headDim;
  let T = params.T;
  let H_out = params.H_out;
  let H_in = params.H_in;
  let groups = H_out / H_in;
  let d = idx % Hd;
  let t = (idx / Hd) % T;
  let bh_out = idx / (T * Hd);
  let b = bh_out / H_out;
  let h_out = bh_out % H_out;
  let h_in = h_out / groups;
  let D = H_in * Hd;
  let srcOff = (b * T + t) * D + h_in * Hd + d;
  var v = src[srcOff];
  if (params.hasBias == 1u) {
    v += bias[h_in * Hd + d];
  }
  dst[idx] = v;
}
