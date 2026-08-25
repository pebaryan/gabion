// dropout_fwd.wgsl — Inverted dropout with hash-based deterministic mask.
// out[i] = x[i] * (hash(i, seed) >= p ? 1/(1-p) : 0)
// hash is a fast integer hash; identical for fwd and bwd given same seed.
// When p == 0, copies input through (identity).

struct Params {
  len: u32,
  seed: u32,
  p: f32,
  _pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn hash2(i: u32, seed: u32) -> u32 {
  var h = i ^ seed;
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return h;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) { return; }
  if (params.p <= 0.0) { out[i] = input[i]; return; }
  let h = hash2(i, params.seed);
  let r = f32(h >> 8u) / 16777215.0;
  if (r < params.p) { out[i] = 0.0; }
  else { out[i] = input[i] / (1.0 - params.p); }
}
