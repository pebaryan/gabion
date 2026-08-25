// conv2d_fwd.wgsl — NCHW conv2d forward, one thread per output element.
// x: [N*Cin*H*W], w: [Cout*cinPerG*kH*kW], b: [Cout] (optional, hasBias)
// out[n,oc,oh,ow] = b[oc] + sum_{icL,kh,kw} x[n, g*cinPerG+icL, oh*sH-pH+kh*dH, ow*sW-pW+kw*dW] * w[oc,icL,kh,kw]
// where g = oc / coutPerG. Pure gather — no write races.

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
  let ic0 = g * params.cinPerG;
  let kH = params.kH, kW = params.kW;
  let sH = params.sH, sW = params.sW, dH = params.dH, dW = params.dW;
  let pH = params.pH, pW = params.pW;
  let H = params.H, W = params.W;
  let cinPerG = params.cinPerG;
  let wBase = oc * cinPerG * kH * kW;
  let xN = n * params.Cin * H * W;
  for (var icL = 0u; icL < cinPerG; icL++) {
    let ic = ic0 + icL;
    let xNC = xN + ic * H * W;
    let wOff = wBase + icL * kH * kW;
    for (var kh = 0u; kh < kH; kh++) {
      let ih = i32(oh) * i32(sH) - i32(pH) + i32(kh) * i32(dH);
      if (ih < 0 || ih >= i32(H)) { continue; }
      let xRow = xNC + u32(ih) * W;
      for (var kw = 0u; kw < kW; kw++) {
        let iw = i32(ow) * i32(sW) - i32(pW) + i32(kw) * i32(dW);
        if (iw < 0 || iw >= i32(W)) { continue; }
        acc += x[xRow + u32(iw)] * w[wOff + kh * kW + kw];
      }
    }
  }
  out[idx] = acc;
}
