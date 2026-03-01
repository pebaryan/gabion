// Fused elementwise operations on 1D buffer views.
// op: 0=add(A,B), 1=mul(A,B), 2=silu(A), 3=scale(A, scalar), 4=sub(A,B), 5=addScalar(A, scalar)
// op: 6=silu_backward(A=x, B=gout) → gout * sig(x) * (1 + x*(1-sig(x)))

struct Params {
  len: u32,
  op: u32,
  scalar: f32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;  // unused for unary ops
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) { return; }

  let a = A[i];
  var result: f32 = 0.0;

  switch params.op {
    case 0u: {  // add
      result = a + B[i];
    }
    case 1u: {  // mul
      result = a * B[i];
    }
    case 2u: {  // silu: x * sigmoid(x)
      let sig = 1.0 / (1.0 + exp(-a));
      result = a * sig;
    }
    case 3u: {  // scale by scalar
      result = a * params.scalar;
    }
    case 4u: {  // sub
      result = a - B[i];
    }
    case 5u: {  // add scalar
      result = a + params.scalar;
    }
    case 6u: {  // silu_backward: A=x (original input), B=gout (upstream grad)
      let sig = 1.0 / (1.0 + exp(-a));
      result = B[i] * sig * (1.0 + a * (1.0 - sig));
    }
    default: {
      result = a;
    }
  }

  Out[i] = result;
}
