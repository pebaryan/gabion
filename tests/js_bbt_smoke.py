"""CPU numeric smoke: Python BBT vs browser JS port (Node + IIFE stub)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "_js_bbt_fixture.json"


def _flatten(params) -> list[float]:
    from gabion.pebble.adapters import flatten_tensors

    return flatten_tensors(params)


def build_fixture() -> dict:
    from gabion.user_models.bbt_transformer import BBTTransformerAdapter

    cfg = {
        "vocab_size": 32,
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "seq_len": 8,
        "d_ff": 32,
        "tie_weights": True,
        "act_quant": True,
    }
    adapter = BBTTransformerAdapter(
        input_dim=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        seq_len=cfg["seq_len"],
        d_ff=cfg["d_ff"],
        act_quant=cfg["act_quant"],
        tie_weights=cfg["tie_weights"],
        use_wikitext=False,
    )
    params = adapter.init_params(seed=7)
    x, y = adapter.sample_batch(batch_size=2, seed=11)
    # Closest match: JS ternarize=true applies STE bitlinear; Python does that
    # on the default (ternarize=False) path.
    logits = adapter.forward(params, x, ternarize=False)
    loss = adapter.loss(logits, y)
    weights = _flatten(params)
    x_np = np.array(x.numpy(), dtype=np.int32)
    y_np = np.array(y.numpy(), dtype=np.int32)
    logits_np = np.array(logits.numpy(), dtype=np.float64)
    sequences = x_np.tolist()
    return {
        "config": cfg,
        "batch_size": int(x_np.shape[0]),
        "ternarize": True,
        "weights": [float(v) for v in weights],
        "x_flat": x_np.reshape(-1).astype(int).tolist(),
        "y_flat": y_np.reshape(-1).astype(int).tolist(),
        "sequences": sequences,
        "logits_flat": logits_np.reshape(-1).astype(float).tolist(),
        "loss": float(loss.item()),
    }


def main() -> int:
    fixture = build_fixture()
    FIXTURE.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"fixture weights={len(fixture['weights'])} logits={len(fixture['logits_flat'])} py_loss={fixture['loss']:.6f}")
    js = Path(__file__).resolve().parent / "js_bbt_smoke.mjs"
    proc = subprocess.run(
        ["node", str(js), str(FIXTURE)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        print(f"node failed exit={proc.returncode}")
        return proc.returncode

    report = json.loads((FIXTURE.with_name("_js_bbt_fixture_js.json")).read_text(encoding="utf-8"))
    # Forward match is the real check. Train step only needs to finish finite.
    logit_ok = report["logits_max_abs"] < 1e-3
    loss_ok = report["loss_abs"] < 1e-4
    train_ok = report["finite"] and report["train_updated_len"] == len(fixture["weights"])
    print(
        "verdict "
        f"logits={'PASS' if logit_ok else 'FAIL'} "
        f"loss={'PASS' if loss_ok else 'FAIL'} "
        f"train={'PASS' if train_ok else 'FAIL'}"
    )
    if not math.isfinite(report["js_loss"]):
        return 1
    if not train_ok:
        return 1
    # Numeric mismatch is reported but still a useful smoke if JS ran.
    return 0 if (logit_ok and loss_ok and train_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
