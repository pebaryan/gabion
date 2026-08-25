// kv_group_sum.wgsl — GQA backward: sum attention grads across a query-head group.
// Input:  dKV [BH * T * headDim]  (per-query-head grads, BH = B*H)
// Output: out [B * kvH * T * headDim] (one row per KV head)
// Thread (b*kvH*T*headDim): out[bidx, t, d] = sum_g in[(b*H + g0 + g)*T*headDim + t*headDim + d]
// where g0 = (bidx % kvH) * group, group = H / kvH. Pure gather — no write races.
// NOTE: no grad flows through the KV-cache decode path, so this is forward-only
// for the training attention (k/v projection backward).

struct Params {
  B: u32, H: u32, kvH: u32, T: u32, headDim: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let group = params.H / params.kvH;
  let rowsOut = params.B * params.kvH * params.T * params.headDim;
  if (idx >= rowsOut) { return; }
  let TD = params.T * params.headDim;
  let bkv = idx / TD;
  let rest = idx % TD;
  let b = bkv / params.kvH;
  let g0 = (bkv % params.kvH) * group;
  let base = (b * params.H + g0) * TD + rest;
  var acc: f32 = 0.0;
  for (var g = 0u; g < group; g++) {
    acc += input[base + g * TD];
  }
  out[idx] = acc;
}
