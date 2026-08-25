// batchnorm_stats.wgsl — per-channel mean/var (training mode), one thread per channel.
// x: [N*C*rest]. Two-pass (mean then sum of squared deviations) to match the CPU path.
// meanBuf: [C], varBuf: [C] (biased variance, /(N*rest)).

struct Params {
  N: u32, C: u32, rest: u32, eps: f32, hasWeight: u32, hasBias: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> meanBuf: array<f32>;
@group(0) @binding(3) var<storage, read_write> varBuf: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= params.C) { return; }
  let cnt = params.N * params.rest;

  var sum: f32 = 0.0;
  for (var n = 0u; n < params.N; n++) {
    let off = (n * params.C + c) * params.rest;
    for (var i = 0u; i < params.rest; i++) { sum += x[off + i]; }
  }
  let mean = sum / f32(cnt);

  var varr: f32 = 0.0;
  for (var n = 0u; n < params.N; n++) {
    let off = (n * params.C + c) * params.rest;
    for (var i = 0u; i < params.rest; i++) {
      let d = x[off + i] - mean;
      varr += d * d;
    }
  }
  meanBuf[c] = mean;
  varBuf[c] = varr / f32(cnt);
}
