// kv_attention_apply.wgsl — softmax-weighted sum over a KV cache, one thread per (bh, headDim element).
// o[bh,d] = sum_j softmax(scores[bh,j]) * VCache[bh,j,d] over j in [0, L).
// Each thread scans scores twice (max, then sum-of-exp + weighted sum) and V once —
// no shared-memory score buffer, so L is unbounded. Pure gather — no write races.
// Masked positions carry -FLT_MAX scores (exp -> 0), so causal masking is inherited
// from kv_attention_scores and needs no flag here.

struct Params {
  BH: u32, L: u32, headDim: u32, scale: f32, causal: u32, pos: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> scores: array<f32>;
@group(0) @binding(2) var<storage, read> VCache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.BH * params.headDim;
  if (idx >= total) { return; }
  let headDim = params.headDim;
  let bh = idx / headDim;
  let d = idx % headDim;
  let L = params.L;
  let sBase = bh * L;

  // Pass 1: row max
  var m: f32 = -3.402823e+38;
  for (var j = 0u; j < L; j++) {
    m = max(m, scores[sBase + j]);
  }

  // Pass 2: sum(exp) and weighted sum of V
  var sumE: f32 = 0.0;
  var acc: f32 = 0.0;
  let vBase = (bh * L) * headDim;
  for (var j = 0u; j < L; j++) {
    let e = exp(scores[sBase + j] - m);
    sumE += e;
    acc += e * VCache[vBase + j * headDim + d];
  }
  out[idx] = acc / max(sumE, 1e-12);
}
