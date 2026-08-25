// lstm_cell_bwd_dx.wgsl — LSTM cell input gradient, one thread per (batch, input dim).
// dx[n,k] = sum_{j,g} ds[n, g*H+j] * wih[(g*H+j)*I + k]. Pure gather — no write races.

struct Params {
  B: u32, H: u32, inputSize: u32, _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> ds: array<f32>;
@group(0) @binding(2) var<storage, read> wih: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.B * params.inputSize;
  if (idx >= total) { return; }
  let I = params.inputSize;
  let n = idx / I;
  let k = idx % I;
  let H = params.H;

  var acc: f32 = 0.0;
  let dsBase = n * 4u * H;
  for (var j = 0u; j < H; j++) {
    for (var g = 0u; g < 4u; g++) {
      acc += ds[dsBase + g * H + j] * wih[(g * H + j) * I + k];
    }
  }
  dx[idx] = acc;
}
