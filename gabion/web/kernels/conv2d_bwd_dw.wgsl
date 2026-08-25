// conv2d_bwd_dw.wgsl — conv2d weight gradient, one thread per weight element.
// dw[oc,icL,kh,kw] = sum_{n,oh,ow valid} gout[n,oc,oh,ow] * x[n, g*cinPerG+icL, oh*sH-pH+kh*dH, ow*sW-pW+kw*dW]
// where g = oc / coutPerG. Pure gather — each thread writes exactly one dw element.

struct Params {
  N: u32, Cin: u32, H: u32, W: u32,
  Cout: u32, Ho: u32, Wo: u32, kH: u32,
  kW: u32, groups: u32, cinPerG: u32, coutPerG: u32,
  sH: u32, sW: u32, dH: u32, dW: u32,
  pH: u32, pW: u32, hasBias: u32, _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gout: array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.Cout * params.cinPerG * params.kH * params.kW;
  if (idx >= total) { return; }
  let kW = params.kW, kH = params.kH;
  let kw = idx % kW;
  let kh = (idx / kW) % kH;
  let icL = (idx / (kW * kH)) % params.cinPerG;
  let oc = idx / (kW * kH * params.cinPerG);

  let g = oc / params.coutPerG;
  let ic0 = g * params.cinPerG;
  let H = params.H, W = params.W;
  let Ho = params.Ho, Wo = params.Wo;

  var acc: f32 = 0.0;
  for (var n = 0u; n < params.N; n++) {
    let xN = n * params.Cin * H * W;
    let gN = (n * params.Cout + oc) * Ho;
    let xC = xN + (ic0 + icL) * H * W;
    for (var oh = 0u; oh < Ho; oh++) {
      let ih = i32(oh) * i32(params.sH) - i32(params.pH) + i32(kh) * i32(params.dH);
      if (ih < 0 || ih >= i32(H)) { continue; }
      let xRow = xC + u32(ih) * W;
      let gRow = gN + oh;
      for (var ow = 0u; ow < Wo; ow++) {
        let iw = i32(ow) * i32(params.sW) - i32(params.pW) + i32(kw) * i32(params.dW);
        if (iw < 0 || iw >= i32(W)) { continue; }
        acc += gout[gRow * Wo + ow] * x[xRow + u32(iw)];
      }
    }
  }
  dw[idx] = acc;
}
