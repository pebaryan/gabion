// layernorm_backward.wgsl — LayerNorm backward over the last dim, one workgroup per row.
// dx[r,j] = inv_std * (gout[r,j] - meanG - y[r,j] * meanGY)
//   y = (x - mean) * inv_std, meanG = mean(gout), meanGY = mean(gout * y).
// Recomputes mean/var from x (no forward intermediates stored on GPU).
// Pure per-row gather — no write races.

struct Params {
  rows: u32,
  d: u32,
  eps: f32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gout: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

var<workgroup> sdata: array<f32, 256>;
var<workgroup> gdata: array<vec2<f32>, 256>;

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

  // Phase 2: row variance -> inv_std
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
  let inv_std = 1.0 / sqrt(sdata[0] / f32(d) + params.eps);
  workgroupBarrier();

  // Phase 3: sumG = sum(gout), dot = sum(gout * y) in one vec2 reduce
  var local_g: vec2<f32> = vec2<f32>(0.0, 0.0);
  for (var j = tid; j < d; j += 256u) {
    let y = (x[base + j] - mean) * inv_std;
    let gv = gout[base + j];
    local_g += vec2<f32>(gv, gv * y);
  }
  gdata[tid] = local_g;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { gdata[tid] += gdata[tid + stride]; }
    workgroupBarrier();
  }
  let meanG = gdata[0].x / f32(d);
  let meanGY = gdata[0].y / f32(d);
  workgroupBarrier();

  // Phase 4: dx
  for (var j = tid; j < d; j += 256u) {
    let y = (x[base + j] - mean) * inv_std;
    dx[base + j] = inv_std * (gout[base + j] - meanG - y * meanGY);
  }
}
