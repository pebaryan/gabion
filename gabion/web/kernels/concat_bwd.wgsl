// concat_bwd.wgsl — backward of concat: split gout into a_grad or b_grad
// is_a = 1 => grad = a part, is_a = 0 => grad = b part
@group(0) @binding(0) var<storage, read> gout: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad: array<f32>;
struct Params { outer: u32, c_a: u32, c_b: u32, inner: u32, is_a: u32 }
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total_a = p.outer * p.c_a * p.inner;
  let total_b = p.outer * p.c_b * p.inner;
  let total = select(total_b, total_a, p.is_a == 1u);
  if (idx >= total) { return; }
  let inner = p.inner;
  let outer_idx = idx / (select(p.c_b, p.c_a, p.is_a == 1u) * inner);
  let rem = idx % (select(p.c_b, p.c_a, p.is_a == 1u) * inner);
  let c = rem / inner;
  let inner_idx = rem % inner;
  let c_total = p.c_a + p.c_b;
  let gout_c = select(c + p.c_a, c, p.is_a == 1u);
  let gout_idx = outer_idx * c_total * inner + gout_c * inner + inner_idx;
  grad[idx] = gout[gout_idx];
}
