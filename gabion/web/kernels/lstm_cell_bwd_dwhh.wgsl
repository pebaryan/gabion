// lstm_cell_bwd_dwhh.wgsl — LSTM recurrent-projection weight gradient, one thread per weight element.
// dwhh[(g*H+j)*H + k] = sum_n ds[n, g*H+j] * hPrev[n, k]. Pure gather — no write races.

struct Params {
  B: u32, H: u32, inputSize: u32, _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> h: array<f32>;
@group(0) @binding(2) var<storage, read> ds: array<f32>;
@group(0) @binding(3) var<storage, read_write> dwhh: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = 4u * params.H * params.H;
  if (idx >= total) { return; }
  let H = params.H;
  let k = idx % H;
  let j = (idx / H) % H;
  let g = (idx / (H * H)) % 4u;

  var acc: f32 = 0.0;
  for (var n = 0u; n < params.B; n++) {
    acc += ds[n * 4u * H + g * H + j] * h[n * H + k];
  }
  dwhh[idx] = acc;
}
