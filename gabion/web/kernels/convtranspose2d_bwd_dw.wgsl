// convtranspose2d_bwd_dw.wgsl — conv_transpose2d weight gradient, one thread per weight element.
// dw[ic,ocL,kh,kw] = sum_{n,ih,iw valid} x[n,ic,ih,iw] * gout[n,oc,oh,ow]
// with oh = ih*sH - pH + kh*dH, ow = iw*sW - pW + kw*dW (direct, mirrors the CPU closure).
// g = ic / cinPerG, oc = g*coutPerG + ocL. Pure gather — no write races.

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
  let total = params.Cin * params.coutPerG * params.kH * params.kW;
  if (idx >= total) { return; }
  let kW = params.kW, kH = params.kH;
  let kw = idx % kW;
  let kh = (idx / kW) % kH;
  let ocL = (idx / (kW * kH)) % params.coutPerG;
  let ic = idx / (kW * kH * params.coutPerG);

  let g = ic / params.cinPerG;
  let oc = g * params.coutPerG + ocL;
  let H = params.H, W = params.W;
  let Ho = params.Ho, Wo = params.Wo;
  let sH = params.sH, sW = params.sW, dH = params.dH, dW = params.dW;
  let pH = params.pH, pW = params.pW;

  var acc: f32 = 0.0;
  for (var n = 0u; n < params.N; n++) {
    let xN = n * params.Cin * H * W;
    let gN = (n * params.Cout + oc) * Ho;
    let xC = xN + ic * H * W;
    for (var ih = 0u; ih < H; ih++) {
      let oh = i32(ih) * i32(sH) - i32(pH) + i32(kh) * i32(dH);
      if (oh < 0 || oh >= i32(Ho)) { continue; }
      let xRow = xC + ih * W;
      let gRow = gN + u32(oh);
      for (var iw = 0u; iw < W; iw++) {
        let ow = i32(iw) * i32(sW) - i32(pW) + i32(kw) * i32(dW);
        if (ow < 0 || ow >= i32(Wo)) { continue; }
        acc += x[xRow + iw] * gout[gRow * Wo + u32(ow)];
      }
    }
  }
  dw[idx] = acc;
}
