// Fused SiLU-Mul for SwiGLU: out[i] = silu(gate[i]) * up[i]
// Input A is [N, 2*dFF] where first half is gate, second half is up.
// Output is [N, dFF].

struct Params {
  len: u32,     // N * dFF (total output elements)
  dFF: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) { return; }

  let row = i / params.dFF;
  let col = i % params.dFF;
  let stride = params.dFF * 2u;
  let gate = A[row * stride + col];
  let up = A[row * stride + params.dFF + col];
  let sig = 1.0 / (1.0 + exp(-gate));
  Out[i] = gate * sig * up;
}
