// lstm_cell_bwd_dh.wgsl — LSTM cell hidden-state gradient, one thread per (batch, hidden dim).
// dhPrev[n,k] = sum_{j,g} ds[n, g*H+j] * whh[(g*H+j)*H + k]. Pure gather — no write races.

struct Params {
  B: u32, H: u32, inputSize: u32, _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> ds: array<f32>;
@group(0) @binding(2) var<storage, read> whh: array<f32>;
@group(0) @binding(3) var<storage, read_write> dhPrev: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.B * params.H;
  if (idx >= total) { return; }
  let H = params.H;
  let n = idx / H;
  let k = idx % H;

  var acc: f32 = 0.0;
  let dsBase = n * 4u * H;
  for (var j = 0u; j < H; j++) {
    for (var g = 0u; g < 4u; g++) {
      acc += ds[dsBase + g * H + j] * whh[(g * H + j) * H + k];
    }
  }
  dhPrev[idx] = acc;
}
