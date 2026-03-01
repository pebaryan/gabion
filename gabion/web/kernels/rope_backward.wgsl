// rope_backward.wgsl — Backward pass for Rotary Position Embedding
// Given: dOut[BH * T * headDim] (upstream grad through RoPE output)
// Compute: dX[BH * T * headDim] (grad w.r.t. RoPE input)
//
// RoPE forward:
//   out[..., i]         = x[..., i] * cos - x[..., halfDim+i] * sin
//   out[..., halfDim+i] = x[..., halfDim+i] * cos + x[..., i] * sin
//
// RoPE backward (transpose of rotation matrix):
//   dX[..., i]         = dOut[..., i] * cos + dOut[..., halfDim+i] * sin
//   dX[..., halfDim+i] = -dOut[..., i] * sin + dOut[..., halfDim+i] * cos

struct Params {
  BH: u32,
  T: u32,
  headDim: u32,
  halfDim: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> dOut: array<f32>;
@group(0) @binding(2) var<storage, read> freq_cos: array<f32>;
@group(0) @binding(3) var<storage, read> freq_sin: array<f32>;
@group(0) @binding(4) var<storage, read_write> dX: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.BH * params.T * params.halfDim;
  if (idx >= total) { return; }

  let halfDim = params.halfDim;
  let headDim = params.headDim;
  let T = params.T;

  // Decompose idx -> (bh, t, i)
  let i = idx % halfDim;
  let rem = idx / halfDim;
  let t = rem % T;
  let bh = rem / T;

  let off = (bh * T + t) * headDim;
  let tOff = t * headDim;

  let dO1 = dOut[off + i];
  let dO2 = dOut[off + halfDim + i];
  let c = freq_cos[tOff + i];
  let s = freq_sin[tOff + i];

  // Transpose of rotation: [cos, -sin; sin, cos]^T = [cos, sin; -sin, cos]
  dX[off + i] = dO1 * c + dO2 * s;
  dX[off + halfDim + i] = -dO1 * s + dO2 * c;
}
