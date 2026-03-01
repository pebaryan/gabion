// Row-wise reduction over a 2D buffer [rows, cols].
// op: 0=sum, 1=max, 2=sumSquares
// Output: [rows] — one value per row.

struct Params {
  rows: u32,
  cols: u32,
  op: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const WG_SIZE: u32 = 256u;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row = wid.x;
  if (row >= params.rows) { return; }

  let cols = params.cols;
  let tid = lid.x;
  let rowOff = row * cols;

  // Initialize accumulator based on op
  var acc: f32;
  switch params.op {
    case 1u: { acc = -3.402823e+38; }  // max: -FLT_MAX
    default: { acc = 0.0; }            // sum, sumSquares
  }

  // Each thread reduces a strided slice of the row
  var i = tid;
  loop {
    if (i >= cols) { break; }
    let val = input[rowOff + i];
    switch params.op {
      case 0u: { acc += val; }
      case 1u: { acc = max(acc, val); }
      case 2u: { acc += val * val; }
      default: { acc += val; }
    }
    i += WG_SIZE;
  }

  shared[tid] = acc;
  workgroupBarrier();

  // Tree reduction in shared memory
  var stride = WG_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      switch params.op {
        case 1u: { shared[tid] = max(shared[tid], shared[tid + stride]); }
        default: { shared[tid] = shared[tid] + shared[tid + stride]; }
      }
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (tid == 0u) {
    output[row] = shared[0];
  }
}
