// batched_transpose.wgsl — Transpose last two dims: out[b, j, i] = in[b, i, j]
// Input:  in[B * M * N]  (shape [B, M, N])
// Output: out[B * N * M] (shape [B, N, M])
// Dispatched as (ceil(N/16), ceil(M/16), B) workgroups.

struct Params {
  M: u32,
  N: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let batch = wid.z;
  let row = gid.y;  // row in input = i
  let col = gid.x;  // col in input = j
  let M = params.M;
  let N = params.N;

  if (row >= M || col >= N) { return; }

  let inIdx = batch * M * N + row * N + col;
  let outIdx = batch * N * M + col * M + row;
  output[outIdx] = input[inIdx];
}
