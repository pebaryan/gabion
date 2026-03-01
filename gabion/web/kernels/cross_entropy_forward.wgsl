// Fused cross-entropy forward: softmax + NLL loss.
// Input: logits[N, V], targets[N] (u32 indices)
// Output: losses[N] (per-sample -log(prob[target])), probs[N*V] (saved for backward)
// One workgroup per row (sample). Up to 256 vocab entries per workgroup stride.

struct Params {
  N: u32,
  V: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> probs: array<f32>;
@group(0) @binding(4) var<storage, read_write> losses: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row = wid.x;
  if (row >= params.N) { return; }
  let V = params.V;
  let tid = lid.x;
  let base = row * V;

  // Phase 1: find row max
  var local_max: f32 = -3.402823e+38;
  for (var c = tid; c < V; c += 256u) {
    local_max = max(local_max, logits[base + c]);
  }
  sdata[tid] = local_max;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { sdata[tid] = max(sdata[tid], sdata[tid + stride]); }
    workgroupBarrier();
  }
  let row_max = sdata[0];
  workgroupBarrier();

  // Phase 2: exp(x - max) and partial sum
  var local_sum: f32 = 0.0;
  for (var c = tid; c < V; c += 256u) {
    let e = exp(logits[base + c] - row_max);
    probs[base + c] = e;
    local_sum += e;
  }
  sdata[tid] = local_sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { sdata[tid] += sdata[tid + stride]; }
    workgroupBarrier();
  }
  let row_sum = max(sdata[0], 1e-12);
  workgroupBarrier();

  // Phase 3: normalize probs
  let inv_sum = 1.0 / row_sum;
  for (var c = tid; c < V; c += 256u) {
    probs[base + c] *= inv_sum;
  }

  // Phase 4: compute loss = -log(prob[target])
  if (tid == 0u) {
    let tgt = min(targets[row], V - 1u);
    losses[row] = -log(max(probs[base + tgt], 1e-12));
  }
}
