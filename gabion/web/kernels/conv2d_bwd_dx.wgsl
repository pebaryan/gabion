// conv2d_bwd_dx.wgsl — conv2d input gradient, one thread per input element.
// dx[n,ic,ih,iw] = sum over contributing (oc,oh,ow,kh,kw) of gout[n,oc,oh,ow] * w[oc,icL,kh,kw]
// where ih = oh*sH - pH + kh*dH  <=>  oh = (ih + pH - kh*dH) / sH (exact division).
// Pure gather — each thread writes exactly one dx element.

struct Params {
  N: u32, Cin: u32, H: u32, W: u32,
  Cout: u32, Ho: u32, Wo: u32, kH: u32,
  kW: u32, groups: u32, cinPerG: u32, coutPerG: u32,
  sH: u32, sW: u32, dH: u32, dW: u32,
  pH: u32, pW: u32, hasBias: u32, _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> gout: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.N * params.Cin * params.H * params.W;
  if (idx >= total) { return; }
  let W = params.W, H = params.H;
  let iw = idx % W;
  let ih = (idx / W) % H;
  let ic = (idx / (W * H)) % params.Cin;
  let n = idx / (W * H * params.Cin);

  let g = ic / params.cinPerG;
  let oc0 = g * params.coutPerG;
  let icL = ic - g * params.cinPerG;
  let kH = params.kH, kW = params.kW;
  let sH = params.sH, sW = params.sW, dH = params.dH, dW = params.dW;
  let pH = params.pH, pW = params.pW;
  let Ho = params.Ho, Wo = params.Wo;
  let coutPerG = params.coutPerG;

  var acc: f32 = 0.0;
  for (var ocL = 0u; ocL < coutPerG; ocL++) {
    let oc = oc0 + ocL;
    let wOff = oc * params.cinPerG * kH * kW + icL * kH * kW;
    let gN = (n * params.Cout + oc) * Ho;
    for (var kh = 0u; kh < kH; kh++) {
      let num = i32(ih) + i32(pH) - i32(kh) * i32(dH);
      if (num < 0) { continue; }
      let oh = num / i32(sH);
      if (num % i32(sH) != 0 || oh >= i32(Ho)) { continue; }
      let gRow = gN + u32(oh);
      for (var kw = 0u; kw < kW; kw++) {
        let num2 = i32(iw) + i32(pW) - i32(kw) * i32(dW);
        if (num2 < 0) { continue; }
        let ow = num2 / i32(sW);
        if (num2 % i32(sW) != 0 || ow >= i32(Wo)) { continue; }
        acc += gout[gRow * Wo + u32(ow)] * w[wOff + kh * kW + kw];
      }
    }
  }
  dx[idx] = acc;
}
