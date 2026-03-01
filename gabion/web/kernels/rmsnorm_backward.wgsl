// rmsnorm_backward.wgsl — Backward pass for RMSNorm with learnable weight
// Input:  x[rows * d], gout[rows * d], weight[d]
// Output: dX[rows * d], dW[d] (atomically accumulated across rows)
//
// One workgroup per row, up to 256 cols per workgroup.
// Recomputes inv = 1/RMS(x) from x[] to avoid storing forward intermediates on GPU.
//
// dX[r,j] = (gout[r,j]*w[j]) * inv - x[r,j] * (inv^3 * dot / d)
//   where dot = sum_j( gout[r,j]*w[j] * x[r,j] )
// dW[j] += gout[r,j] * x[r,j] * inv   (accumulated across all rows)

struct Params {
  rows: u32,
  d: u32,
  eps: f32,
  hasWeight: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gout: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read_write> dX: array<f32>;
@group(0) @binding(5) var<storage, read_write> dW: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let row = wid.x;
  if (row >= params.rows) { return; }
  let d = params.d;
  let tid = lid.x;
  let base = row * d;

  // Phase 0: compute sum_sq = sum(x[j]^2) via tree reduce
  var local_ss: f32 = 0.0;
  for (var j = tid; j < d; j += 256u) {
    let v = x[base + j];
    local_ss += v * v;
  }
  sdata[tid] = local_ss;
  workgroupBarrier();

  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    workgroupBarrier();
  }
  let sum_sq = sdata[0];
  let inv = 1.0 / sqrt(sum_sq / f32(d) + params.eps);
  workgroupBarrier();

  // Phase 1: compute dot = sum(gout_adj[j] * x[j]) via tree reduce
  // where gout_adj[j] = gout[j] * weight[j] if hasWeight, else gout[j]
  var local_dot: f32 = 0.0;
  for (var j = tid; j < d; j += 256u) {
    var gj = gout[base + j];
    if (params.hasWeight != 0u) {
      gj *= weight[j];
    }
    local_dot += gj * x[base + j];
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

  // Phase 2: compute dX and accumulate dW
  let coeff = inv * inv * inv * dot_val / f32(d);
  for (var j = tid; j < d; j += 256u) {
    var gj = gout[base + j];
    let xj = x[base + j];

    // dW[j] += gout[j] * x[j] * inv  (no weight adjustment needed)
    if (params.hasWeight != 0u) {
      // Use atomicAdd-style manual accumulation — but WGSL storage doesn't support
      // atomics on f32 directly. Instead we'll let the host accumulate dW per-row
      // by writing to dW at row-offset and reducing on CPU.
      // Actually, for simplicity: write per-row dW contribution to dW[row*d + j].
      // Host will sum across rows.
      dW[base + j] = gj * xj * inv;

      gj *= weight[j];
    }

    // dX[r,j] = gj_adj * inv - x[j] * coeff
    dX[base + j] = gj * inv - xj * coeff;
  }
}
