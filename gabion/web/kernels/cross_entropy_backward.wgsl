// Cross-entropy backward: dLogits = (probs - one_hot(targets)) * scale
// Input: probs[N*V], targets[N]
// Output: dLogits[N*V]
// Elementwise per-element kernel.

struct Params {
  N: u32,
  V: u32,
  scale: f32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> dLogits: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = params.N * params.V;
  if (i >= total) { return; }

  let row = i / params.V;
  let col = i % params.V;
  let tgt = min(targets[row], params.V - 1u);
  var grad = probs[i] * params.scale;
  if (col == tgt) {
    grad -= params.scale;
  }
  dLogits[i] = grad;
}
