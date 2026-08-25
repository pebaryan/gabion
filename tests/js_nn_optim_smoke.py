"""CPU smoke: JS nn/optim vs tinygrad 0.14 LayerNorm, GELU, ReLU, AdamW."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "_js_nn_optim_fixture.json"


def build_fixture() -> dict:
    from tinygrad import Tensor, nn, Context
    from tinygrad.nn.optim import AdamW, LAMB

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(6,)).astype(np.float32)
    xt = Tensor(x.tolist())
    relu = xt.relu().numpy().astype(np.float64).reshape(-1)
    gelu = xt.gelu().numpy().astype(np.float64).reshape(-1)

    ln_x = rng.normal(0, 1, size=(2, 4)).astype(np.float32)
    ln = nn.LayerNorm(4, eps=1e-5)
    # pin affine to known values so JS can load them
    ln.weight.assign(Tensor([1.1, 0.9, 1.0, 1.2]))
    ln.bias.assign(Tensor([0.1, -0.1, 0.0, 0.05]))
    ln_y = ln(Tensor(ln_x.tolist())).numpy().astype(np.float64).reshape(-1)

    p0 = rng.normal(0, 1, size=(5,)).astype(np.float32)
    g0 = rng.normal(0, 1, size=(5,)).astype(np.float32)
    pt = Tensor(p0.tolist())
    pt.grad = Tensor(g0.tolist())
    opt = AdamW([pt], lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)
    with Context(TRAINING=1):
        opt.step()
    p_after = pt.numpy().astype(np.float64).reshape(-1)

    # Conv2d
    conv_x = rng.normal(0, 1, size=(1, 1, 4, 4)).astype(np.float32)
    conv = nn.Conv2d(1, 2, 3, padding=1, bias=True)
    conv_y = conv(Tensor(conv_x.tolist())).numpy().astype(np.float64).reshape(-1)

    # GroupNorm
    gn_x = rng.normal(0, 1, size=(2, 4, 3, 3)).astype(np.float32)
    gn = nn.GroupNorm(2, 4, eps=1e-5)
    gn.weight.assign(Tensor([1.1, 0.9, 1.0, 1.2]))
    gn.bias.assign(Tensor([0.05, -0.05, 0.0, 0.1]))
    gn_y = gn(Tensor(gn_x.tolist())).numpy().astype(np.float64).reshape(-1)

    # BatchNorm train
    bn_x = rng.normal(0, 1, size=(2, 3, 2, 2)).astype(np.float32)
    bn = nn.BatchNorm(3, eps=1e-5, momentum=0.1)
    bn.weight.assign(Tensor([1.0, 1.1, 0.9]))
    bn.bias.assign(Tensor([0.0, 0.1, -0.1]))
    with Context(TRAINING=1):
        bn_y = bn(Tensor(bn_x.tolist())).numpy().astype(np.float64).reshape(-1)

    # LSTMCell
    lstm = nn.LSTMCell(3, 4, bias=True)
    lx = rng.normal(0, 1, size=(2, 3)).astype(np.float32)
    with Context(TRAINING=0):
        h, c = lstm(Tensor(lx.tolist()))
    lstm_h = h.numpy().astype(np.float64).reshape(-1)
    lstm_c = c.numpy().astype(np.float64).reshape(-1)

    # LAMB
    lp = rng.normal(0, 1, size=(5,)).astype(np.float32)
    lg = rng.normal(0, 1, size=(5,)).astype(np.float32)
    lpt = Tensor(lp.tolist())
    lpt.grad = Tensor(lg.tolist())
    lamb = LAMB([lpt], lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01, adam=False)
    with Context(TRAINING=1):
        lamb.step()
    lamb_after = lpt.numpy().astype(np.float64).reshape(-1)

    return {
        "x": x.astype(float).tolist(),
        "x_shape": [6],
        "relu": relu.tolist(),
        "gelu": gelu.tolist(),
        "ln_dim": 4,
        "ln_x": ln_x.astype(float).reshape(-1).tolist(),
        "ln_x_shape": [2, 4],
        "ln_weight": ln.weight.numpy().astype(float).reshape(-1).tolist(),
        "ln_bias": ln.bias.numpy().astype(float).reshape(-1).tolist(),
        "ln_y": ln_y.tolist(),
        "p": p0.astype(float).tolist(),
        "g": g0.astype(float).tolist(),
        "p_after": p_after.tolist(),
        "lr": 1e-3,
        "wd": 0.01,
        "conv_x": conv_x.astype(float).reshape(-1).tolist(),
        "conv_x_shape": list(conv_x.shape),
        "conv_w": conv.weight.numpy().astype(float).reshape(-1).tolist(),
        "conv_w_shape": list(conv.weight.shape),
        "conv_b": conv.bias.numpy().astype(float).reshape(-1).tolist(),
        "conv_y": conv_y.tolist(),
        "gn_x": gn_x.astype(float).reshape(-1).tolist(),
        "gn_x_shape": list(gn_x.shape),
        "gn_w": gn.weight.numpy().astype(float).reshape(-1).tolist(),
        "gn_b": gn.bias.numpy().astype(float).reshape(-1).tolist(),
        "gn_y": gn_y.tolist(),
        "bn_x": bn_x.astype(float).reshape(-1).tolist(),
        "bn_x_shape": list(bn_x.shape),
        "bn_w": bn.weight.numpy().astype(float).reshape(-1).tolist(),
        "bn_b": bn.bias.numpy().astype(float).reshape(-1).tolist(),
        "bn_y": bn_y.tolist(),
        "lstm_x": lx.astype(float).reshape(-1).tolist(),
        "lstm_wih": lstm.weight_ih.numpy().astype(float).reshape(-1).tolist(),
        "lstm_whh": lstm.weight_hh.numpy().astype(float).reshape(-1).tolist(),
        "lstm_bih": lstm.bias_ih.numpy().astype(float).reshape(-1).tolist(),
        "lstm_bhh": lstm.bias_hh.numpy().astype(float).reshape(-1).tolist(),
        "lstm_h": lstm_h.tolist(),
        "lstm_c": lstm_c.tolist(),
        "lamb_p": lp.astype(float).tolist(),
        "lamb_g": lg.astype(float).tolist(),
        "lamb_after": lamb_after.tolist(),
    }


def main() -> int:
    fixture = build_fixture()
    FIXTURE.write_text(json.dumps(fixture), encoding="utf-8")
    js = Path(__file__).resolve().parent / "js_nn_optim_smoke.mjs"
    proc = subprocess.run(["node", str(js), str(FIXTURE)], cwd=str(ROOT), capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode
    report = json.loads(FIXTURE.with_name("_js_nn_optim_fixture_js.json").read_text(encoding="utf-8"))
    checks = {
        "relu": report["relu_max_abs"] < 1e-6,
        "gelu": report["gelu_max_abs"] < 1e-5,
        "layernorm": report["ln_max_abs"] < 1e-5,
        "dropout_id": report["dropout_identity_ok"],
        "dropout_p1": report["dropout_p1_zero"],
        "adamw": report["adamw_max_abs"] < 1e-5,
        "conv2d": report["conv_max_abs"] < 1e-5,
        "groupnorm": report["gn_max_abs"] < 1e-5,
        "batchnorm": report["bn_max_abs"] < 1e-5,
        "lstm": report["lstm_h_max_abs"] < 1e-5 and report["lstm_c_max_abs"] < 1e-5,
        "lamb": report["lamb_max_abs"] < 1e-5,
    }
    print("verdict", {k: ("PASS" if v else "FAIL") for k, v in checks.items()})
    print("nn", report["nn_modules"], "optim", report["optim_exports"])
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
