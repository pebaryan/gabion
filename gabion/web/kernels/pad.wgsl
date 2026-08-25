// pad.wgsl — zero-pad for 4D NCHW (N,C unchanged, pad H/W)
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
struct Params { N: u32, C: u32, H: u32, W: u32, H_out: u32, W_out: u32, pad_h_before: u32, pad_w_before: u32 }
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = p.N * p.C * p.H_out * p.W_out;
  if (idx >= total) { return; }
  let w_out = p.W_out;
  let hw_out = p.H_out * w_out;
  let chw_out = p.C * hw_out;
  let n = idx / chw_out;
  let rem1 = idx % chw_out;
  let c = rem1 / hw_out;
  let rem2 = rem1 % hw_out;
  let h_out = rem2 / w_out;
  let w_out_c = rem2 % w_out;
  let h_in = i32(h_out) - i32(p.pad_h_before);
  let w_in = i32(w_out_c) - i32(p.pad_w_before);
  if (h_in < 0 || h_in >= i32(p.H) || w_in < 0 || w_in >= i32(p.W)) {
    dst[idx] = 0.0;
  } else {
    let src_idx = ((n * p.C + c) * p.H + u32(h_in)) * p.W + u32(w_in);
    dst[idx] = src[src_idx];
  }
}
