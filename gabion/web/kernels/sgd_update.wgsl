// SGD optimizer update kernel.
// Per-element: param -= lr * grad

struct SGDParams {
  len: u32,
  _pad: u32,
  lr: f32,
  _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: SGDParams;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> param: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) { return; }
  param[i] -= params.lr * grad[i];
}
