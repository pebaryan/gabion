// lstm_cell_bwd.wgsl — LSTM cell backward (gate grads + cell grads), one thread per (batch, unit).
// Recomputes the forward activations from x/h/c/weights (no stored intermediates).
// Given upstream dh[n,j] (grad of h') and dc[n,j] (grad of c'):
//   c1 = tanh(c'); dcTot = dc + dh*o*(1-c1^2)
//   ds_i = dcTot*g*i*(1-i);  ds_f = dcTot*c*f*(1-f);  ds_g = dcTot*i*(1-g^2);  ds_o = dh*c1*o*(1-o)
//   dcPrev = dcTot*f
// Writes dsBuf [B*4H] (gate order i,f,g,o) and dcPrevBuf [B*H]. Pure gather.

struct Params {
  B: u32, H: u32, inputSize: u32, _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> h: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read> wih: array<f32>;
@group(0) @binding(5) var<storage, read> whh: array<f32>;
@group(0) @binding(6) var<storage, read> bIh: array<f32>;
@group(0) @binding(7) var<storage, read> bHh: array<f32>;
@group(0) @binding(8) var<storage, read> dh: array<f32>;
@group(0) @binding(9) var<storage, read> dc: array<f32>;
@group(0) @binding(10) var<storage, read_write> dsBuf: array<f32>;
@group(0) @binding(11) var<storage, read_write> dcPrevBuf: array<f32>;

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

  // Recompute forward gates (mirrors lstm_cell.wgsl)
  var s0: f32 = bIh[j] + bHh[j];
  var s1: f32 = bIh[H + j] + bHh[H + j];
  var s2: f32 = bIh[2u * H + j] + bHh[2u * H + j];
  var s3: f32 = bIh[3u * H + j] + bHh[3u * H + j];
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
  let cprev = c[idx];
  let cnew = f * cprev + i * g;
  let c1 = tanh(cnew);

  let dhu = dh[idx];
  let dcu = dc[idx];
  let dcTot = dcu + dhu * o * (1.0 - c1 * c1);

  let dsI = dcTot * g * i * (1.0 - i);
  let dsF = dcTot * cprev * f * (1.0 - f);
  let dsG = dcTot * i * (1.0 - g * g);
  let dsO = dhu * c1 * o * (1.0 - o);

  let base = n * 4u * H;
  dsBuf[base + j] = dsI;
  dsBuf[base + H + j] = dsF;
  dsBuf[base + 2u * H + j] = dsG;
  dsBuf[base + 3u * H + j] = dsO;
  dcPrevBuf[idx] = dcTot * f;
}
