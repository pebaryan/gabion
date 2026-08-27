"""Full-pipeline qwen35 benchmark — the authoritative measurement tool.

Implements the protocol in ``docs/qwen35-benchmark.md``.  ONE process holds
BOTH shards (device="CUDA:0" / device="CUDA:1") — the proven arrangement for
this box (two separate processes per GPU hit a WDDM per-context ~7.4GB wall).

Prints a single machine-readable report line; paste it into any PR/commit
that claims a speedup.

Usage:
  python tools/bench_qwen35_pipeline.py --gguf D:/aimodels/Qwen3.5-0.8B-IQ4_NL.gguf --verify
  python tools/bench_qwen35_pipeline.py --gguf D:/aimodels/Qwen3.8-27B-IQ4_NL.gguf --verify --decode-tokens 10 --prefill-lens 64 512
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

import numpy as np

# --- fixed correctness-gate prompt (tokenizer-stable, verified 2026-08-27) ---
# template: <|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n
# 0.8B has no bos (add_bos_token=false); 27B prefixes bos 248044.
FRANCE_IDS_08B = [248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198]
FRANCE_IDS_27B = [248044] + FRANCE_IDS_08B
EXPECTED_08B = [248068, 198, 90700, 8340, 25, 271]
EXPECTED_27B = [248068, 198, 760, 1156, 369, 9859]


def pick_gate(gguf: str) -> tuple[list[int], list[int]]:
    name = Path(gguf).name.lower()
    if "0.8b" in name:
        return FRANCE_IDS_08B, EXPECTED_08B
    return FRANCE_IDS_27B, EXPECTED_27B


def run_pipeline(left, right, ids, s0, s1) -> int:
    """One full forward over `ids`; returns the argmax of the final position."""
    hidden = left.forward_shard_ids_to_hidden(ids, s0)
    logits = right.forward_shard_hidden_to_logits(hidden, s1)
    return int(np.argmax(logits))


def mem_report() -> str:
    from tinygrad.helpers import GlobalCounters

    used = GlobalCounters.mem_used_per_device
    return ",".join(f"{k}:{v / 2**30:.2f}GiB" for k, v in sorted(used.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=r"D:\aimodels\Qwen3.8-27B-IQ4_NL.gguf")
    ap.add_argument("--verify", action="store_true", help="run the correctness gate (mandatory for claims)")
    ap.add_argument("--decode-tokens", type=int, default=10, help="decode steps measured after warmup")
    ap.add_argument("--prefill-lens", type=int, nargs="+", default=[64, 512], help="synthetic prefill lengths")
    ap.add_argument("--warmup-prefill", type=int, default=32, help="synthetic prefill tokens for warmup")
    ap.add_argument("--warmup-decode", type=int, default=2, help="decode steps discarded as warmup")
    args = ap.parse_args()

    os.environ.setdefault("DEV", "CUDA")
    from gabion.user_models.qwen35_text import Qwen35TextAdapter

    gate_ids, gate_expected = pick_gate(args.gguf)
    t0 = time.perf_counter()
    left = Qwen35TextAdapter.from_gguf_shard(args.gguf, 0, 2, device="CUDA:0")
    right = Qwen35TextAdapter.from_gguf_shard(args.gguf, 1, 2, device="CUDA:1")
    load_s = time.perf_counter() - t0
    print(f"loaded shards 0/1 in {load_s:.1f}s | devices {left.dev} / {right.dev} | {mem_report()}", flush=True)

    # ---- correctness gate ----
    gate = "SKIP"
    if args.verify:
        s0, s1 = left.new_state(), right.new_state()
        got = [run_pipeline(left, right, gate_ids, s0, s1)]
        for _ in range(1, len(gate_expected)):
            got.append(run_pipeline(left, right, [got[-1]], s0, s1))  # continue the same states
        gate = "PASS" if got[: len(gate_expected)] == gate_expected else f"FAIL {got[: len(gate_expected)]}"
        print(f"correctness gate: {gate} (want {gate_expected})", flush=True)
        if gate != "PASS":
            print("STOP: correctness gate failed — do not trust any timing below.", flush=True)

    # ---- warmup (jit capture + compile happen here; discarded) ----
    t_w = time.perf_counter()
    s0, s1 = left.new_state(), right.new_state()
    warm_ids = list(range(args.warmup_prefill))
    run_pipeline(left, right, warm_ids, s0, s1)
    for _ in range(args.warmup_decode):
        run_pipeline(left, right, [warm_ids[-1]], s0, s1)
    warm_s = time.perf_counter() - t_w
    print(f"warmup (capture+compile): {warm_s:.1f}s | {mem_report()}", flush=True)

    # ---- decode measurement: fresh state prefilled with the gate prompt ----
    s0, s1 = left.new_state(), right.new_state()
    nxt = run_pipeline(left, right, gate_ids, s0, s1)
    times = []
    for _ in range(args.decode_tokens):
        t1 = time.perf_counter()
        nxt = run_pipeline(left, right, [nxt], s0, s1)
        times.append(time.perf_counter() - t1)
    times = np.array(times)
    decode_min, decode_med, decode_mean = times.min(), np.median(times), times.mean()

    # ---- prefill measurement: synthetic lengths, fresh state each ----
    prefill = {}
    for L in args.prefill_lens:
        s0, s1 = left.new_state(), right.new_state()
        ids = [i % 248000 for i in range(L)]
        t1 = time.perf_counter()
        run_pipeline(left, right, ids, s0, s1)
        dt = time.perf_counter() - t1
        prefill[L] = L / dt

    name = Path(args.gguf).name
    pref = " ".join(f"prefill{L}_tok_s={prefill[L]:.3f}" for L in args.prefill_lens)
    print(
        "qwen35-bench v1 "
        f"gguf={name} "
        f"load_s={load_s:.1f} warmup_s={warm_s:.1f} "
        f"decode_s_tok_min={decode_min:.4f} decode_s_tok_median={decode_med:.4f} "
        f"decode_s_tok_mean={decode_mean:.4f} decode_tok_s={1 / decode_med:.2f} "
        f"{pref} "
        f"vram_load={mem_report()} "
        f"gate={gate}",
        flush=True,
    )


if __name__ == "__main__":
    main()
