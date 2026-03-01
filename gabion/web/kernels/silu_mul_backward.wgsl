// Backward for fused SiLU-Mul (SwiGLU activation).
// Given dOut[N, dFF] and original A[N, 2*dFF]:
//   gate = A[row, col], up = A[row, dFF + col]
//   dGate = dOut * up * dsilu(gate) = dOut * up * sig(gate) * (1 + gate*(1-sig(gate)))
//   dUp   = dOut * silu(gate) = dOut * gate * sig(gate)
// Output dA[N, 2*dFF]: first half = dGate, second half = dUp

struct Params {
  len: u32,     // N * dFF
  dFF: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> dOut: array<f32>;
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<storage, read_write> dA: array<f32>;

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
  let silu_gate = gate * sig;
  let d = dOut[i];

  // dGate = d * up * sig * (1 + gate * (1 - sig))
  let dGate = d * up * sig * (1.0 + gate * (1.0 - sig));
  // dUp = d * silu(gate)
  let dUp = d * silu_gate;

  dA[row * stride + col] = dGate;
  dA[row * stride + params.dFF + col] = dUp;
}
