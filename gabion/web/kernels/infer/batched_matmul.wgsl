// Batched tiled matrix multiply: C[b, M, N] = A[b, M, K] * B[b, K, N]
// A: [B*M*K], B: [B*K*N], C: [B*M*N] — contiguous batch slices.
// Workgroups dispatched as (ceil(N/16), ceil(M/16), B).

struct Params {
  M: u32,
  K: u32,
  N: u32,
  flags: u32,  // bit0 = transposeB
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const TILE: u32 = 16u;

var<workgroup> tileA: array<f32, 256>;  // TILE * TILE
var<workgroup> tileB: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let batch = wid.z;
  let row = wid.y * TILE + lid.y;
  let col = wid.x * TILE + lid.x;
  let M = params.M;
  let K = params.K;
  let N = params.N;
  let transposeB = (params.flags & 1u) != 0u;

  // Batch offsets into flat arrays
  let aBase = batch * M * K;
  let bBase = batch * K * N;
  let cBase = batch * M * N;

  var acc: f32 = 0.0;
  let numTiles = (K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
    // Load tile of A
    let aCol = t * TILE + lid.x;
    if (row < M && aCol < K) {
      tileA[lid.y * TILE + lid.x] = A[aBase + row * K + aCol];
    } else {
      tileA[lid.y * TILE + lid.x] = 0.0;
    }

    // Load tile of B (with optional transpose)
    let bRow = t * TILE + lid.y;
    if (transposeB) {
      if (bRow < K && col < N) {
        tileB[lid.y * TILE + lid.x] = B[bBase + col * K + bRow];
      } else {
        tileB[lid.y * TILE + lid.x] = 0.0;
      }
    } else {
      if (bRow < K && col < N) {
        tileB[lid.y * TILE + lid.x] = B[bBase + bRow * N + col];
      } else {
        tileB[lid.y * TILE + lid.x] = 0.0;
      }
    }

    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      acc += tileA[lid.y * TILE + k] * tileB[k * TILE + lid.x];
    }

    workgroupBarrier();
  }

  if (row < M && col < N) {
    C[cBase + row * N + col] = acc;
  }
}
