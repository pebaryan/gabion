// pad_bwd.wgsl — backward of pad: grad = slice(gout)
@group(0) @binding(0) var<storage, read> gout: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad: array<f32>;
struct Params { N: u32, C: u32, H: u32, W: u32, H_out: u32, W_out: u32, pad_h_before: u32, pad_w_before: u32 }
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = p.N * p.C * p.H * p.W;
  if (idx >= total) { return; }
  let w = p.W;
  let hw = p.H * w;
  let chw = p.C * hw;
  let n = idx / chw;
  let rem1 = idx % chw;
  let c = rem1 / hw;
  let rem2 = rem1 % hw;
  let h = rem2 / w;
  let w_c = rem2 % w;
  let h_out = h + p.pad_h_before;
  let w_out = w_c + p.pad_w_before;
  let gout_idx = ((n * p.C + c) * p.H_out + h_out) * p.W_out + w_out;
  grad[idx] = gout[gout_idx];
}
