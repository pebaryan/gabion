// Fused Adam optimizer update kernel.
// Per-element: updates m, v (moments) and param (weights) in-place.
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g * g
//   param -= effLr * (m / bc1) / (sqrt(v / bc2) + eps)

struct AdamParams {
  len:   u32,
  _pad:  u32,
  beta1: f32,
  beta2: f32,
  effLr: f32,
  bc1:   f32,
  bc2:   f32,
  eps:   f32,
}

@group(0) @binding(0) var<uniform> params: AdamParams;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> param: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) { return; }

  let g = grad[i];
  let mi = params.beta1 * m[i] + (1.0 - params.beta1) * g;
  let vi = params.beta2 * v[i] + (1.0 - params.beta2) * g * g;
  m[i] = mi;
  v[i] = vi;
  param[i] -= params.effLr * (mi / params.bc1) / (sqrt(vi / params.bc2) + params.eps);
}
