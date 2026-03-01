// rope.wgsl — Rotary Position Embedding
// Input:  x[BH * T * headDim]  (query or key tensor, shape [BH, T, headDim])
// Output: out[BH * T * headDim]
// Freq:   freq_cos[T * headDim], freq_sin[T * headDim]  (precomputed cos/sin tables)
// Uniform: BH (u32), T (u32), headDim (u32), halfDim (u32)
//
// For each (bh, t, i) where i < halfDim:
//   out[..., i]          = x[..., i] * cos[t, i] - x[..., halfDim+i] * sin[t, i]
//   out[..., halfDim+i]  = x[..., halfDim+i] * cos[t, i] + x[..., i] * sin[t, i]

struct Params {
  BH: u32,
  T: u32,
  headDim: u32,
  halfDim: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> freq_cos: array<f32>;
@group(0) @binding(3) var<storage, read> freq_sin: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  // Global thread index maps to (bh, t, i) where i < halfDim
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

  let x1 = x[off + i];
  let x2 = x[off + halfDim + i];
  let c = freq_cos[tOff + i];
  let s = freq_sin[tOff + i];

  out[off + i] = x1 * c - x2 * s;
  out[off + halfDim + i] = x2 * c + x1 * s;
}
