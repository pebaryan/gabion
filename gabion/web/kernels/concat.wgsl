// concat.wgsl — generic concat along one axis (outer * (c_a+c_b) * inner)
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
struct Params { outer: u32, c_a: u32, c_b: u32, inner: u32 }
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = p.outer * (p.c_a + p.c_b) * p.inner;
  if (idx >= total) { return; }
  let inner = p.inner;
  let c_total = p.c_a + p.c_b;
  let rem = idx % (c_total * inner);
  let c = rem / inner;
  let inner_idx = rem % inner;
  let outer_idx = idx / (c_total * inner);
  if (c < p.c_a) {
    let src_idx = outer_idx * p.c_a * inner + c * inner + inner_idx;
    dst[idx] = a[src_idx];
  } else {
    let c_b = c - p.c_a;
    let src_idx = outer_idx * p.c_b * inner + c_b * inner + inner_idx;
    dst[idx] = b[src_idx];
  }
}
