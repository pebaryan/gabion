"""Warm-cache single-token benchmark for a Qwen3.8 GGUF shard.

Examples (PowerShell):
  $env:CUDA_PATH = 'C:\Windows\System32\nvcuda.dll'
  py -3.11 tools/bench_qwen35_cuda.py --shard 0 --visible-device 0
  py -3.11 tools/bench_qwen35_cuda.py --shard 1 --visible-device 1

The first measured call is intentionally excluded: tinygrad compiles CUDA
kernels lazily.  This measures the steady-state layer/dequant path, not mesh
websocket overhead or tokenizer time.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=r"D:\aimodels\Qwen3.8-27B-IQ4_NL.gguf")
    ap.add_argument("--shard", type=int, choices=(0, 1), required=True)
    ap.add_argument("--num-shards", type=int, default=2)
    ap.add_argument("--visible-device", default=None)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--prefill-tokens", type=int, default=8)
    args = ap.parse_args()

    # These must be set before importing tinygrad.
    os.environ.setdefault("DEV", "CUDA")
    if args.visible_device is not None:
        # tinygrad 0.14 ignores HCQ_VISIBLE_DEVICES and DEV=CUDA:<idx> (the
        # second colon-field is a renderer name); the qwen35 adapter reads
        # the explicit indexed device string from QWEN35_DEVICE.
        os.environ["QWEN35_DEVICE"] = f"CUDA:{args.visible_device}"

    import numpy as np

    from gabion.user_models.qwen35_text import Qwen35TextAdapter

    path = Path(args.gguf)
    t0 = time.perf_counter()
    model = Qwen35TextAdapter.from_gguf_shard(path, args.shard, args.num_shards)
    load_s = time.perf_counter() - t0
    state = model.new_state()
    if args.shard == 0:
        run = lambda: model.forward_shard_ids_to_hidden([123], state)
        prefill_state = model.new_state()
        prompt_ids = list(range(max(1, args.prefill_tokens)))
        prefill = lambda: model.forward_shard_ids_to_hidden(prompt_ids, prefill_state)
    else:
        hidden = np.zeros((1, model.d_model), dtype=np.float32)
        run = lambda: model.forward_shard_hidden_to_logits(hidden, state)
        prefill_state = model.new_state()
        prompt_hidden = np.zeros((max(1, args.prefill_tokens), model.d_model), dtype=np.float32)
        prefill = lambda: model.forward_shard_hidden_to_logits(prompt_hidden, prefill_state)

    # Compile and populate the modest dequant cache before timing.
    warm_t0 = time.perf_counter()
    out = run()
    warm_s = time.perf_counter() - warm_t0
    if not np.isfinite(np.asarray(out)).all():
        raise RuntimeError("warmup produced non-finite output")

    prefill_t0 = time.perf_counter()
    prefill_out = prefill()
    prefill_s = time.perf_counter() - prefill_t0
    if not np.isfinite(np.asarray(prefill_out)).all():
        raise RuntimeError("prefill produced non-finite output")

    times = []
    for _ in range(max(1, args.steps)):
        t0 = time.perf_counter()
        out = run()
        times.append(time.perf_counter() - t0)
        if not np.isfinite(np.asarray(out)).all():
            raise RuntimeError("benchmark produced non-finite output")
    print(
        f"gguf={path.name} shard={args.shard}/{args.num_shards} "
        f"layers={model.layer_start}-{model.layer_end - 1} "
        f"load_s={load_s:.3f} warmup_s={warm_s:.3f} "
        f"prefill_s={prefill_s:.3f} prefill_tok_s={max(1, args.prefill_tokens) / prefill_s:.4f} "
        f"steady_s={sum(times) / len(times):.3f} "
        f"min_s={min(times):.3f} steps={len(times)}"
    )


if __name__ == "__main__":
    main()
