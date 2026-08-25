// pad.wgsl — zero-pad for NCHW tensors (CPU fallback active; WGSL stub)
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
struct Params { n: u32, c: u32, h: u32, w: u32, pad_h_before: u32, pad_h_after: u32, pad_w_before: u32, pad_w_after: u32, }
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.n * p.c * (p.h + p.pad_h_before + p.pad_h_after) * (p.w + p.pad_w_before + p.pad_w_after)) { return; }
  // stub: actual pad is CPU-side for now (8x8 tiny); this kernel is reserved for future dispatch
  dst[idx] = 0.0;
}
