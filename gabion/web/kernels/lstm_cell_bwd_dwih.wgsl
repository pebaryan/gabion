// lstm_cell_bwd_dwih.wgsl — LSTM input-projection weight gradient, one thread per weight element.
// dwih[(g*H+j)*I + k] = sum_n ds[n, g*H+j] * x[n, k]. Pure gather — no write races.

struct Params {
  B: u32, H: u32, inputSize: u32, _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> ds: array<f32>;
@group(0) @binding(3) var<storage, read_write> dwih: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = 4u * params.H * params.inputSize;
  if (idx >= total) { return; }
  let I = params.inputSize;
  let H = params.H;
  let k = idx % I;
  let j = (idx / I) % H;
  let g = (idx / (I * H)) % 4u;

  var acc: f32 = 0.0;
  for (var n = 0u; n < params.B; n++) {
    acc += ds[n * 4u * H + g * H + j] * x[n * I + k];
  }
  dwih[idx] = acc;
}
