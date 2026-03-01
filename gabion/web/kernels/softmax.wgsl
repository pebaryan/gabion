// softmax.wgsl — Row-wise softmax: out[r,c] = exp(in[r,c] - max_r) / sum_r
// Also supports causal masking: if mask_flag=1, positions where col > row_within_block are set to -inf before softmax.
// Uniform: rows (u32), cols (u32), mask_flag (u32), _pad (u32)
// One workgroup per row, up to 256 columns per workgroup.

struct Params {
  rows: u32,
  cols: u32,
  mask_flag: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let row = wid.x;
  if (row >= params.rows) { return; }
  let cols = params.cols;
  let tid = lid.x;
  let base = row * cols;

  // For causal mask: which T-row is this within the T×T attention block?
  // rows = B*H*T, each T×T block has T rows. row_in_block = row % T
  // But we pass cols = T, so row_in_block = row % cols when mask_flag is set.
  let row_in_block = row % cols;

  // Phase 1: find row max (strided access + tree reduce)
  var local_max: f32 = -3.402823e+38; // -FLT_MAX
  for (var c = tid; c < cols; c += 256u) {
    var val = input[base + c];
    // Apply causal mask: if col > row_in_block, treat as -inf
    if (params.mask_flag == 1u && c > row_in_block) {
      val = -3.402823e+38;
    }
    local_max = max(local_max, val);
  }
  sdata[tid] = local_max;
  workgroupBarrier();

  // Tree reduce for max
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      sdata[tid] = max(sdata[tid], sdata[tid + stride]);
    }
    workgroupBarrier();
  }
  let row_max = sdata[0];
  workgroupBarrier();

  // Phase 2: compute exp(x - max) and partial sum
  var local_sum: f32 = 0.0;
  for (var c = tid; c < cols; c += 256u) {
    var val = input[base + c];
    if (params.mask_flag == 1u && c > row_in_block) {
      val = -3.402823e+38;
    }
    let e = exp(val - row_max);
    output[base + c] = e; // store exp temporarily
    local_sum += e;
  }
  sdata[tid] = local_sum;
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

  // Phase 3: normalize
  let inv_sum = 1.0 / row_sum;
  for (var c = tid; c < cols; c += 256u) {
    output[base + c] *= inv_sum;
  }
}
