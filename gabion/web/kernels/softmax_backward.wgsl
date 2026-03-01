// softmax_backward.wgsl — Backward pass for scaled causal softmax
// Given: attn[rows, cols] (forward softmax output), dAttn[rows, cols] (upstream grad)
// Compute: dScores[r,c] = attn[r,c] * (dAttn[r,c] - dot(dAttn[r,:], attn[r,:])) * scale
//          dScores[r,c] = 0  where c > r % cols  (causal mask zeroing)
//
// One workgroup per row, up to 256 columns per workgroup.

struct Params {
  rows: u32,
  cols: u32,
  scale: f32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> attn: array<f32>;
@group(0) @binding(2) var<storage, read> dAttn: array<f32>;
@group(0) @binding(3) var<storage, read_write> dScores: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let row = wid.x;
  if (row >= params.rows) { return; }
  let cols = params.cols;
  let tid = lid.x;
  let base = row * cols;
  let row_in_block = row % cols;

  // Phase 1: compute dot(dAttn[row,:], attn[row,:]) via tree reduce
  var local_dot: f32 = 0.0;
  for (var c = tid; c < cols; c += 256u) {
    local_dot += dAttn[base + c] * attn[base + c];
  }
  sdata[tid] = local_dot;
  workgroupBarrier();

  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    workgroupBarrier();
  }
  let dot_val = sdata[0];
  workgroupBarrier();

  // Phase 2: compute dScores = attn * (dAttn - dot) * scale, with causal mask zeroing
  let scale = params.scale;
  for (var c = tid; c < cols; c += 256u) {
    var ds: f32 = 0.0;
    if (c <= row_in_block) {
      let a = attn[base + c];
      ds = a * (dAttn[base + c] - dot_val) * scale;
    }
    dScores[base + c] = ds;
  }
}
