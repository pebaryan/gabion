// kv_attention_scores.wgsl — attention scores over a KV cache, one thread per (bh, cache position).
// For a single query position: score[bh, j] = scale * dot(Q[bh,:], KCache[bh,j,:]).
// Causal masking: positions j > pos get -FLT_MAX (softmax -> 0). Pure gather — no write races.
// Unlike fused_attention this is O(BH*L) threads and has no T<=256 constraint.

struct Params {
  BH: u32, L: u32, headDim: u32, scale: f32, causal: u32, pos: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> KCache: array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.BH * params.L;
  if (idx >= total) { return; }
  let L = params.L;
  let bh = idx / L;
  let j = idx % L;
  let headDim = params.headDim;

  if (params.causal != 0u && j > params.pos) {
    scores[idx] = -3.402823e+38;
    return;
  }

  let qBase = bh * headDim;
  let kBase = (bh * L + j) * headDim;
  var dot: f32 = 0.0;
  for (var d = 0u; d < headDim; d++) {
    dot += Q[qBase + d] * KCache[kBase + d];
  }
  scores[idx] = dot * params.scale;
}
