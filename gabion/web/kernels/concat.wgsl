// concat.wgsl — channel concat for NCHW (CPU fallback active; WGSL stub)
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
struct Params { n: u32, c_a: u32, c_b: u32, hw: u32, }
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.n * (p.c_a + p.c_b) * p.hw) { return; }
  dst[idx] = 0.0;
}
