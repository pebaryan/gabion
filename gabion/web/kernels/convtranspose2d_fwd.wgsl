// convtranspose2d_fwd.wgsl — NCHW conv_transpose2d forward, one thread per output element.
// x: [N*Cin*H*W], w: [Cin*coutPerG*kH*kW] (tinygrad ConvTranspose2d layout), b: [Cout] (optional)
// Scatter y[n,oc,oh,ow] += x[n,ic,ih,iw] * w[ic,ocL,kh,kw] with oh = ih*sH - pH + kh*dH
// is rewritten as a gather: for output (oc,oh,ow),
//   ih = (oh + pH - kh*dH) / sH   (exact division), iw = (ow + pW - kw*dW) / sW.
// g = oc / coutPerG, ocL = oc % coutPerG, ic = g*cinPerG + icL. Pure gather — no write races.

struct Params {
  N: u32, Cin: u32, H: u32, W: u32,
  Cout: u32, Ho: u32, Wo: u32, kH: u32,
  kW: u32, groups: u32, cinPerG: u32, coutPerG: u32,
  sH: u32, sW: u32, dH: u32, dW: u32,
  pH: u32, pW: u32, hasBias: u32, _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.N * params.Cout * params.Ho * params.Wo;
  if (idx >= total) { return; }
  let Wo = params.Wo, Ho = params.Ho;
  let ow = idx % Wo;
  let oh = (idx / Wo) % Ho;
  let oc = (idx / (Wo * Ho)) % params.Cout;
  let n = idx / (Wo * Ho * params.Cout);

  var acc: f32 = params.hasBias != 0u ? b[oc] : 0.0;
  let g = oc / params.coutPerG;
  let ocL = oc % params.coutPerG;
  let kH = params.kH, kW = params.kW;
  let sH = params.sH, sW = params.sW, dH = params.dH, dW = params.dW;
  let pH = params.pH, pW = params.pW;
  let H = params.H, W = params.W;
  let cinPerG = params.cinPerG;
  let xN = n * params.Cin * H * W;
  for (var icL = 0u; icL < cinPerG; icL++) {
    let ic = g * cinPerG + icL;
    let wOff = ic * params.coutPerG * kH * kW + ocL * kH * kW;
    let xC = xN + ic * H * W;
    for (var kh = 0u; kh < kH; kh++) {
      let num = i32(oh) + i32(pH) - i32(kh) * i32(dH);
      if (num < 0) { continue; }
      let ih = num / i32(sH);
      if (num % i32(sH) != 0 || ih >= i32(H)) { continue; }
      let xRow = xC + u32(ih) * W;
      for (var kw = 0u; kw < kW; kw++) {
        let num2 = i32(ow) + i32(pW) - i32(kw) * i32(dW);
        if (num2 < 0) { continue; }
        let iw = num2 / i32(sW);
        if (num2 % i32(sW) != 0 || iw >= i32(W)) { continue; }
        acc += x[xRow + u32(iw)] * w[wOff + kh * kW + kw];
      }
    }
  }
  out[idx] = acc;
}
