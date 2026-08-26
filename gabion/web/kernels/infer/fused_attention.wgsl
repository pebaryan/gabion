// fused_attention.wgsl — Fused Q·K^T scaling + causal mask + softmax
// Replaces: batched_matmul(Q, K, transposeB) + elementwise_scale + causal_softmax
//
// Input:  Q[BH * T * headDim], K[BH * T * headDim]
// Output: attn[BH * T * T]  (attention weights, row-normalized)
//
// One workgroup per output row (BH * T rows total).
// Each thread handles one key column (tid < T).
// Constraint: T <= 256 (workgroup size). Caller must fall back for larger T.
//
// Per row (bh, t_q):
//   score[t_k] = dot(Q[bh,t_q,:], K[bh,t_k,:]) * scale   if t_k <= t_q
//   score[t_k] = -inf                                      if t_k >  t_q  (causal)
//   attn[t_k]  = softmax(score)[t_k]

struct Params {
  BH: u32,
  T: u32,
  headDim: u32,
  scale: f32,   // 1.0 / sqrt(headDim)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read_write> attn: array<f32>;

var<workgroup> qCache: array<f32, 256>;  // shared Q row (headDim <= 256)
var<workgroup> sdata: array<f32, 256>;   // scratch for reductions

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let rowIdx = wid.x;
  let BH = params.BH;
  let T = params.T;
  let headDim = params.headDim;
  let tid = lid.x;

  if (rowIdx >= BH * T) { return; }

  let bh = rowIdx / T;
  let t_q = rowIdx % T;

  // --- Cooperatively load Q[bh, t_q, :] into shared memory ---
  if (tid < headDim) {
    qCache[tid] = Q[(bh * T + t_q) * headDim + tid];
  }
  workgroupBarrier();

  // --- Phase 0: Compute scaled dot product with causal mask ---
  var score: f32 = -3.402823e+38;  // -FLT_MAX
  if (tid < T && tid <= t_q) {
    let kBase = (bh * T + tid) * headDim;
    var dot: f32 = 0.0;
    for (var d: u32 = 0u; d < headDim; d = d + 1u) {
      dot += qCache[d] * K[kBase + d];
    }
    score = dot * params.scale;
  }
  sdata[tid] = score;
  workgroupBarrier();

  // --- Phase 1: Tree reduce for row max ---
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      sdata[tid] = max(sdata[tid], sdata[tid + stride]);
    }
    workgroupBarrier();
  }
  let row_max = sdata[0];
  workgroupBarrier();

  // --- Phase 2: exp(score - max) and partial sum ---
  var e: f32 = 0.0;
  if (tid < T) {
    e = exp(score - row_max);
  }
  sdata[tid] = e;
  workgroupBarrier();

  // Tree reduce for sum
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    workgroupBarrier();
  }
  let row_sum = max(sdata[0], 1e-12);
  workgroupBarrier();

  // --- Phase 3: Normalize and write ---
  if (tid < T) {
    attn[rowIdx * T + tid] = e / row_sum;
  }
}
