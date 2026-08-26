// heads_combine.wgsl — [B*H, T, headDim] -> [B*T, D]
struct Params {
  B: u32,
  T: u32,
  H: u32,
  headDim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.B * params.T * params.H * params.headDim;
  let idx = gid.x;
  if (idx >= total) { return; }
  let Hd = params.headDim;
  let H = params.H;
  let T = params.T;
  let d = idx % Hd;
  let h = (idx / Hd) % H;
  let t = (idx / (Hd * H)) % T;
  let b = idx / (Hd * H * T);
  let D = H * Hd;
  let srcOff = (b * H + h) * T * Hd + t * Hd + d;
  let dstOff = (b * T + t) * D + h * Hd + d;
  dst[dstOff] = src[srcOff];
}
