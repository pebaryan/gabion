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
    from tinygrad.nn.optim import AdamW, LAMB, Muon

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

    convg = nn.Conv2d(4, 4, 3, groups=2, padding=1, bias=True)
    convg_x = rng.normal(0, 1, size=(1, 4, 5, 5)).astype(np.float32)
    convg_y = convg(Tensor(convg_x.tolist())).numpy().astype(np.float64).reshape(-1)

    convd = nn.Conv2d(1, 1, 3, dilation=2, padding=2, bias=False)
    convd_x = rng.normal(0, 1, size=(1, 1, 6, 6)).astype(np.float32)
    convd_y = convd(Tensor(convd_x.tolist())).numpy().astype(np.float64).reshape(-1)

    ct = nn.ConvTranspose2d(1, 1, 3, bias=True)
    ct_x = rng.normal(0, 1, size=(1, 1, 3, 3)).astype(np.float32)
    ct_xt = Tensor(ct_x.tolist())
    ct_yt = ct(ct_xt)
    ct_y = ct_yt.numpy().astype(np.float64).reshape(-1)
    ct_g = rng.normal(0, 1, size=tuple(ct_yt.shape)).astype(np.float32)
    ct_w_grad, ct_b_grad, ct_x_grad = (ct_yt * Tensor(ct_g.tolist())).sum().gradient(
        ct.weight, ct.bias, ct_xt
    )

    # harder ConvTranspose2d: groups=2, stride=2, padding=1, output_padding=1
    ct2 = nn.ConvTranspose2d(2, 4, 3, stride=2, padding=1, output_padding=1, groups=2, bias=True)
    ct2_x = rng.normal(0, 1, size=(2, 2, 4, 4)).astype(np.float32)
    ct2_xt = Tensor(ct2_x.tolist())
    ct2_yt = ct2(ct2_xt)
    ct2_y = ct2_yt.numpy().astype(np.float64).reshape(-1)
    ct2_g = rng.normal(0, 1, size=tuple(ct2_yt.shape)).astype(np.float32)
    ct2_w_grad, ct2_b_grad, ct2_x_grad = (ct2_yt * Tensor(ct2_g.tolist())).sum().gradient(
        ct2.weight, ct2.bias, ct2_xt
    )

    # Tensor sum/mean reductions
    red_x = rng.normal(0, 1, size=(2, 3, 4, 5)).astype(np.float32)
    red_xt = Tensor(red_x.tolist())
    ysum = red_xt.sum(axis=[2, 3])
    ymean = red_xt.mean(axis=1)
    red_g1 = rng.normal(0, 1, size=tuple(ysum.shape)).astype(np.float32)
    red_g2 = rng.normal(0, 1, size=tuple(ymean.shape)).astype(np.float32)
    sum_x_grad, = (ysum * Tensor(red_g1.tolist())).sum().gradient(red_xt)
    mean_x_grad, = (ymean * Tensor(red_g2.tolist())).sum().gradient(red_xt)

    mp = rng.normal(0, 1, size=(4, 4)).astype(np.float32)
    mg = rng.normal(0, 1, size=(4, 4)).astype(np.float32)
    mpt = Tensor(mp.tolist())
    mpt.grad = Tensor(mg.tolist())
    muon = Muon([mpt], lr=1e-3, momentum=0.95, weight_decay=0.1, ns_steps=5)
    with Context(TRAINING=1):
        muon.step()
    muon_after = mpt.numpy().astype(np.float64).reshape(-1)

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
        "convg_x": convg_x.astype(float).reshape(-1).tolist(),
        "convg_x_shape": list(convg_x.shape),
        "convg_w": convg.weight.numpy().astype(float).reshape(-1).tolist(),
        "convg_w_shape": list(convg.weight.shape),
        "convg_b": convg.bias.numpy().astype(float).reshape(-1).tolist(),
        "convg_y": convg_y.tolist(),
        "convd_x": convd_x.astype(float).reshape(-1).tolist(),
        "convd_x_shape": list(convd_x.shape),
        "convd_w": convd.weight.numpy().astype(float).reshape(-1).tolist(),
        "convd_w_shape": list(convd.weight.shape),
        "convd_y": convd_y.tolist(),
        "ct_x": ct_x.astype(float).reshape(-1).tolist(),
        "ct_x_shape": list(ct_x.shape),
        "ct_w": ct.weight.numpy().astype(float).reshape(-1).tolist(),
        "ct_w_shape": list(ct.weight.shape),
        "ct_b": ct.bias.numpy().astype(float).reshape(-1).tolist(),
        "ct_y": ct_y.tolist(),
        "ct_g": ct_g.astype(float).reshape(-1).tolist(),
        "ct_x_grad": ct_x_grad.numpy().astype(np.float64).reshape(-1).tolist(),
        "ct_w_grad": ct_w_grad.numpy().astype(np.float64).reshape(-1).tolist(),
        "ct_b_grad": ct_b_grad.numpy().astype(np.float64).reshape(-1).tolist(),
        "ct2_x": ct2_x.astype(float).reshape(-1).tolist(),
        "ct2_x_shape": list(ct2_x.shape),
        "ct2_w": ct2.weight.numpy().astype(float).reshape(-1).tolist(),
        "ct2_w_shape": list(ct2.weight.shape),
        "ct2_b": ct2.bias.numpy().astype(float).reshape(-1).tolist(),
        "ct2_y": ct2_y.tolist(),
        "ct2_g": ct2_g.astype(float).reshape(-1).tolist(),
        "ct2_x_grad": ct2_x_grad.numpy().astype(np.float64).reshape(-1).tolist(),
        "ct2_w_grad": ct2_w_grad.numpy().astype(np.float64).reshape(-1).tolist(),
        "ct2_b_grad": ct2_b_grad.numpy().astype(np.float64).reshape(-1).tolist(),
        "red_x": red_x.astype(float).reshape(-1).tolist(),
        "red_x_shape": list(red_x.shape),
        "sum_y": ysum.numpy().astype(np.float64).reshape(-1).tolist(),
        "sum_y_shape": list(ysum.shape),
        "sum_g": red_g1.astype(float).reshape(-1).tolist(),
        "sum_x_grad": sum_x_grad.numpy().astype(np.float64).reshape(-1).tolist(),
        "mean_y": ymean.numpy().astype(np.float64).reshape(-1).tolist(),
        "mean_y_shape": list(ymean.shape),
        "mean_g": red_g2.astype(float).reshape(-1).tolist(),
        "mean_x_grad": mean_x_grad.numpy().astype(np.float64).reshape(-1).tolist(),
        "muon_p": mp.astype(float).reshape(-1).tolist(),
        "muon_g": mg.astype(float).reshape(-1).tolist(),
        "muon_after": muon_after.tolist(),
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
        "conv_groups": report["convg_max_abs"] < 1e-5,
        "conv_dilate": report["convd_max_abs"] < 1e-5,
        "conv_transpose": report["ct_max_abs"] < 1e-4,
        "ct_backward": report["ct_grad_x"] < 1e-4 and report["ct_grad_w"] < 1e-4 and report["ct_grad_b"] < 1e-4,
        "ct2_forward": report["ct2_max_abs"] < 1e-4,
        "ct2_backward": report["ct2_grad_x"] < 1e-4 and report["ct2_grad_w"] < 1e-4 and report["ct2_grad_b"] < 1e-4,
        "muon": report["muon_max_abs"] < 1e-4,
        "reduce_sum": report["sum_fwd"] < 1e-5 and report["sum_grad"] < 1e-5,
        "reduce_mean": report["mean_fwd"] < 1e-5 and report["mean_grad"] < 1e-5,
    }
    print("verdict", {k: ("PASS" if v else "FAIL") for k, v in checks.items()})
    print("nn", report["nn_modules"], "optim", report["optim_exports"])
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
