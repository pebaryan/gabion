// lstm_cell.wgsl — single-step LSTM cell, one thread per (batch, hidden unit).
// Gate order matches tinygrad LSTMCell: i, f, g, o (rows 0..H-1, H..2H-1, 2H..3H-1, 3H..4H-1).
//   s_g = bIh[gH+j] + bHh[gH+j] + sum_k x[n,k]*wih[(gH+j)*I+k] + sum_k h[n,k]*whh[(gH+j)*H+k]
//   i = sigmoid(s_i), f = sigmoid(s_f), g = tanh(s_g), o = sigmoid(s_o)
//   c' = f*c + i*g, h' = o*tanh(c')
// Each thread writes exactly one hOut and one cOut element — no write races.

struct Params {
  B: u32, H: u32, inputSize: u32, hasBias: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> h: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read> wih: array<f32>;
@group(0) @binding(5) var<storage, read> whh: array<f32>;
@group(0) @binding(6) var<storage, read> bIh: array<f32>;
@group(0) @binding(7) var<storage, read> bHh: array<f32>;
@group(0) @binding(8) var<storage, read_write> hOut: array<f32>;
@group(0) @binding(9) var<storage, read_write> cOut: array<f32>;

fn sigmoid(v: f32) -> f32 {
  return 1.0 / (1.0 + exp(-v));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.B * params.H;
  if (idx >= total) { return; }
  let H = params.H;
  let n = idx / H;
  let j = idx % H;
  let I = params.inputSize;

  var s0: f32 = params.hasBias != 0u ? bIh[j] + bHh[j] : 0.0;
  var s1: f32 = params.hasBias != 0u ? bIh[H + j] + bHh[H + j] : 0.0;
  var s2: f32 = params.hasBias != 0u ? bIh[2u * H + j] + bHh[2u * H + j] : 0.0;
  var s3: f32 = params.hasBias != 0u ? bIh[3u * H + j] + bHh[3u * H + j] : 0.0;

  let xBase = n * I;
  let hBase = n * H;
  for (var k = 0u; k < I; k++) {
    let xv = x[xBase + k];
    s0 += xv * wih[j * I + k];
    s1 += xv * wih[(H + j) * I + k];
    s2 += xv * wih[(2u * H + j) * I + k];
    s3 += xv * wih[(3u * H + j) * I + k];
  }
  for (var k = 0u; k < H; k++) {
    let hv = h[hBase + k];
    s0 += hv * whh[j * H + k];
    s1 += hv * whh[(H + j) * H + k];
    s2 += hv * whh[(2u * H + j) * H + k];
    s3 += hv * whh[(3u * H + j) * H + k];
  }

  let i = sigmoid(s0);
  let f = sigmoid(s1);
  let g = tanh(s2);
  let o = sigmoid(s3);
  let cnew = f * c[idx] + i * g;
  let hnew = o * tanh(cnew);
  cOut[idx] = cnew;
  hOut[idx] = hnew;
}
