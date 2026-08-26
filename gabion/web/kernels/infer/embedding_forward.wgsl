// Embedding forward: gather rows from embedding table.
// out[row * D + col] = emb[indices[row] * D + col]
// Total elements: BT * D

struct Params {
  BT: u32,
  D: u32,
  V: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> emb: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = params.BT * params.D;
  if (i >= total) { return; }

  let row = i / params.D;
  let col = i % params.D;
  let idx = min(indices[row], params.V - 1u);
  output[i] = emb[idx * params.D + col];
}
