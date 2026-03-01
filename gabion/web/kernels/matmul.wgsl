// Tiled matrix multiply: C[M,N] = A[M,K] * B[K,N]
// 32x32 output tile, 16x16 workgroup, each thread computes a 2x2 output block.
// Shared memory: 2 * 32 * 32 * 4 = 8 KB (within 16 KB limit).

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

const TILE: u32 = 32u;
const WG: u32 = 16u;

var<workgroup> tileA: array<f32, 1024>;  // TILE * TILE = 32*32
var<workgroup> tileB: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let M = params.M;
  let K = params.K;
  let N = params.N;
  let transposeB = (params.flags & 1u) != 0u;

  // Each thread computes a 2x2 block in the output tile
  let row0 = wid.y * TILE + lid.y * 2u;
  let col0 = wid.x * TILE + lid.x * 2u;

  var acc00: f32 = 0.0;
  var acc01: f32 = 0.0;
  var acc10: f32 = 0.0;
  var acc11: f32 = 0.0;

  let numTiles = (K + TILE - 1u) / TILE;
  let flatId = lid.y * WG + lid.x;  // 0..255

  for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
    let tileOff = t * TILE;

    // Cooperative load: 256 threads load 32*32 = 1024 elements (4 per thread)
    for (var p: u32 = 0u; p < 4u; p = p + 1u) {
      let idx = flatId + p * 256u;
      let lr = idx / TILE;  // row within tile
      let lc = idx % TILE;  // col within tile

      // Load tileA: A[wid.y*TILE + lr, tileOff + lc]
      let aRow = wid.y * TILE + lr;
      let aCol = tileOff + lc;
      if (aRow < M && aCol < K) {
        tileA[idx] = A[aRow * K + aCol];
      } else {
        tileA[idx] = 0.0;
      }

      // Load tileB: B[tileOff + lr, wid.x*TILE + lc] (or transposed)
      let bRow = tileOff + lr;
      let bCol = wid.x * TILE + lc;
      if (transposeB) {
        if (bRow < K && bCol < N) {
          tileB[idx] = B[bCol * K + bRow];
        } else {
          tileB[idx] = 0.0;
        }
      } else {
        if (bRow < K && bCol < N) {
          tileB[idx] = B[bRow * N + bCol];
        } else {
          tileB[idx] = 0.0;
        }
      }
    }

    workgroupBarrier();

    // Accumulate 2x2 block
    let r0 = lid.y * 2u;
    let c0 = lid.x * 2u;
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      let a0 = tileA[r0 * TILE + k];
      let a1 = tileA[(r0 + 1u) * TILE + k];
      let b0 = tileB[k * TILE + c0];
      let b1 = tileB[k * TILE + c0 + 1u];
      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
    }

    workgroupBarrier();
  }

  // Write 2x2 output block
  if (row0 < M && col0 < N) { C[row0 * N + col0] = acc00; }
  if (row0 < M && col0 + 1u < N) { C[row0 * N + col0 + 1u] = acc01; }
  if (row0 + 1u < M && col0 < N) { C[(row0 + 1u) * N + col0] = acc10; }
  if (row0 + 1u < M && col0 + 1u < N) { C[(row0 + 1u) * N + col0 + 1u] = acc11; }
}
