// layernorm.wgsl — LayerNorm over the last dim.
// x: [rows, d], weight: [d] (optional, hasWeight), bias: [d] (optional, hasBias)
// out[r, j] = (x - mean_r) / sqrt(var_r + eps) * weight[j] + bias[j]
// One workgroup per row.

struct Params {
  rows: u32,
  d: u32,
  eps: f32,
  hasWeight: u32,
  hasBias: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let row = wid.x;
  if (row >= params.rows) { return; }
  let d = params.d;
  let tid = lid.x;
  let base = row * d;

  // Phase 1: row mean
  var local_sum: f32 = 0.0;
  for (var j = tid; j < d; j += 256u) {
    local_sum += x[base + j];
  }
  sdata[tid] = local_sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { sdata[tid] += sdata[tid + stride]; }
    workgroupBarrier();
  }
  let mean = sdata[0] / f32(d);
  workgroupBarrier();

  // Phase 2: row variance
  var local_var: f32 = 0.0;
  for (var j = tid; j < d; j += 256u) {
    let v = x[base + j] - mean;
    local_var += v * v;
  }
  sdata[tid] = local_var;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { sdata[tid] += sdata[tid + stride]; }
    workgroupBarrier();
  }
  let varr = sdata[0] / f32(d);
  let inv_std = 1.0 / sqrt(varr + params.eps);
  workgroupBarrier();

  // Phase 3: normalize + affine
  for (var j = tid; j < d; j += 256u) {
    var y: f32 = (x[base + j] - mean) * inv_std;
    if (params.hasWeight != 0u) { y *= weight[j]; }
    if (params.hasBias != 0u) { y += bias[j]; }
    out[base + j] = y;
  }
}
