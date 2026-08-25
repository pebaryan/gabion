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

    # Tensor pad/concat
    pad_x = rng.normal(0, 1, size=(2, 3, 4, 4)).astype(np.float32)
    pad_xt = Tensor(pad_x.tolist())
    pad_y = pad_xt.pad([(0, 0), (0, 0), (1, 1), (1, 1)])
    pad_g = rng.normal(0, 1, size=tuple(pad_y.shape)).astype(np.float32)
    pad_x_grad, = (pad_y * Tensor(pad_g.tolist())).sum().gradient(pad_xt)
    # hard asymmetric
    pad2_x = rng.normal(0, 1, size=(2, 3, 4, 5)).astype(np.float32)
    pad2_xt = Tensor(pad2_x.tolist())
    pad2_y = pad2_xt.pad([(0, 0), (0, 0), (1, 2), (0, 3)])
    pad2_g = rng.normal(0, 1, size=tuple(pad2_y.shape)).astype(np.float32)
    pad2_x_grad, = (pad2_y * Tensor(pad2_g.tolist())).sum().gradient(pad2_xt)

    cat_a = rng.normal(0, 1, size=(2, 4, 3, 3)).astype(np.float32)
    cat_b = rng.normal(0, 1, size=(2, 6, 3, 3)).astype(np.float32)
    cat_at = Tensor(cat_a.tolist())
    cat_bt = Tensor(cat_b.tolist())
    cat_y = cat_at.cat(cat_bt, dim=1)
    cat_g = rng.normal(0, 1, size=tuple(cat_y.shape)).astype(np.float32)
    cat_a_grad, cat_b_grad = (cat_y * Tensor(cat_g.tolist())).sum().gradient(cat_at, cat_bt)
    # hard axis=3
    cat2_a = rng.normal(0, 1, size=(2, 3, 4, 3)).astype(np.float32)
    cat2_b = rng.normal(0, 1, size=(2, 3, 4, 5)).astype(np.float32)
    cat2_at = Tensor(cat2_a.tolist())
    cat2_bt = Tensor(cat2_b.tolist())
    cat2_y = cat2_at.cat(cat2_bt, dim=3)
    cat2_g = rng.normal(0, 1, size=tuple(cat2_y.shape)).astype(np.float32)
    cat2_a_grad, cat2_b_grad = (cat2_y * Tensor(cat2_g.tolist())).sum().gradient(cat2_at, cat2_bt)

    # SinusoidalTimestep
    ts = [0, 17, 333, 999]
    half = 8
    ste = np.zeros((4, 16), dtype=np.float64)
    for b, t in enumerate(ts):
        for i in range(half):
            freq = np.exp(-np.log(10000.0) * (i / max(1, half - 1)))
            ang = float(t) * freq
            ste[b, i] = np.float32(np.sin(ang))
            ste[b, half + i] = np.float32(np.cos(ang))

    # ResBlock reference (mirrors JS composition)
    from tinygrad.nn import GroupNorm as TGGroupNorm, Conv2d as TGConv2d
    rb_x = rng.normal(0, 1, size=(2, 8, 6, 6)).astype(np.float32)
    rb_t = rng.normal(0, 1, size=(2, 16)).astype(np.float32)
    rb_gn1 = TGGroupNorm(4, 8, eps=1e-5)
    rb_conv1 = TGConv2d(8, 12, 3, padding=1, bias=False)
    rb_gn2 = TGGroupNorm(4, 12, eps=1e-5)
    rb_conv2 = TGConv2d(12, 12, 3, padding=1, bias=False)
    rb_mlp_w = rng.normal(0, 0.5, size=(16, 12)).astype(np.float32)
    rb_mlp_wt = Tensor(rb_mlp_w.tolist())
    rb_skip = TGConv2d(8, 12, 1, bias=False)

    def rb_forward(xt, tt):
        h = rb_gn1(xt).silu()
        h = rb_conv1(h)
        h = rb_gn2(h).silu()
        h = rb_conv2(h)
        h = h + tt.matmul(rb_mlp_wt).reshape(2, 12, 1, 1)
        return h + rb_skip(xt)

    rb_xt = Tensor(rb_x.tolist())
    rb_tt = Tensor(rb_t.tolist())
    rb_yt = rb_forward(rb_xt, rb_tt)
    rb_y = rb_yt.numpy().astype(np.float64).reshape(-1)
    rb_gout = rng.normal(0, 1, size=tuple(rb_yt.shape)).astype(np.float32)
    grads = (rb_yt * Tensor(rb_gout.tolist())).sum().gradient(
        rb_xt, rb_gn1.weight, rb_gn1.bias, rb_conv1.weight,
        rb_gn2.weight, rb_gn2.bias, rb_conv2.weight, rb_mlp_wt, rb_skip.weight,
    )
    (rb_x_grad, rb_gn1w_grad, rb_gn1b_grad, rb_conv1w_grad,
     rb_gn2w_grad, rb_gn2b_grad, rb_conv2w_grad, rb_mlp_grad, rb_skipw_grad) = [g.numpy().astype(np.float64).reshape(-1) for g in grads]

    # SpatialAttention reference (mirrors JS nchwToBhwc + qkv + attn)
    sa_x = rng.normal(0, 1, size=(2, 4, 8, 8)).astype(np.float32)
    sa_gn = TGGroupNorm(2, 4, eps=1e-5)
    sa_qkv_w = rng.normal(0, 0.5, size=(4, 12)).astype(np.float32)
    sa_qkv_wt = Tensor(sa_qkv_w.tolist())
    sa_proj_w = rng.normal(0, 0.5, size=(4, 4)).astype(np.float32)
    sa_proj_wt = Tensor(sa_proj_w.tolist())

    def sa_forward(xt):
        h = sa_gn(xt)
        bhwc = h.permute(0, 2, 3, 1).reshape(2, 64, 4)
        flat = bhwc.reshape(128, 4)
        qkvFlat = flat.matmul(sa_qkv_wt)
        qkv3d = qkvFlat.reshape(2, 64, 12)
        q, k, v = qkv3d.chunk(3, dim=2)
        scale = 1 / np.sqrt(4)
        scores = (q @ k.transpose(1, 2)) * scale
        attn = scores.softmax(axis=-1)
        out = attn @ v
        outFlat = out.reshape(128, 4)
        projFlat = outFlat.matmul(sa_proj_wt)
        proj3d = projFlat.reshape(2, 64, 4)
        projNchw = proj3d.reshape(2, 8, 8, 4).permute(0, 3, 1, 2)
        return projNchw + xt

    sa_xt = Tensor(sa_x.tolist())
    sa_yt = sa_forward(sa_xt)
    sa_y = sa_yt.numpy().astype(np.float64).reshape(-1)
    sa_gout = rng.normal(0, 1, size=tuple(sa_yt.shape)).astype(np.float32)
    sa_grads = (sa_yt * Tensor(sa_gout.tolist())).sum().gradient(
        sa_xt, sa_gn.weight, sa_gn.bias, sa_qkv_wt, sa_proj_wt
    )
    sa_x_grad, sa_gnw_grad, sa_gnb_grad, sa_qkv_grad, sa_proj_grad = [g.numpy().astype(np.float64).reshape(-1) for g in sa_grads]

    # UNet tiny (8x8, base=4, chMults [1,2], in=2) forward reference
    from tinygrad.nn import ConvTranspose2d as TGConvTranspose2d
    unet_x = rng.normal(0, 1, size=(2, 2, 8, 8)).astype(np.float32)
    unet_t = rng.normal(0, 1, size=(2, 16)).astype(np.float32)
    # stem
    unet_stem = TGConv2d(2, 4, 3, padding=1, bias=False)
    # down0 rb 4->4
    unet_d0_gn1 = TGGroupNorm(2, 4, eps=1e-5)
    unet_d0_c1 = TGConv2d(4, 4, 3, padding=1, bias=False)
    unet_d0_gn2 = TGGroupNorm(2, 4, eps=1e-5)
    unet_d0_c2 = TGConv2d(4, 4, 3, padding=1, bias=False)
    unet_d0_mlp = rng.normal(0, 0.5, size=(16, 4)).astype(np.float32)
    unet_d0_mlp_t = Tensor(unet_d0_mlp.tolist())
    unet_down0 = TGConv2d(4, 8, 3, stride=2, padding=1, bias=False)
    # down1 rb 8->8
    unet_d1_gn1 = TGGroupNorm(2, 8, eps=1e-5)
    unet_d1_c1 = TGConv2d(8, 8, 3, padding=1, bias=False)
    unet_d1_gn2 = TGGroupNorm(2, 8, eps=1e-5)
    unet_d1_c2 = TGConv2d(8, 8, 3, padding=1, bias=False)
    unet_d1_mlp = rng.normal(0, 0.5, size=(16, 8)).astype(np.float32)
    unet_d1_mlp_t = Tensor(unet_d1_mlp.tolist())
    # mid
    unet_m1_gn1 = TGGroupNorm(2, 8, eps=1e-5)
    unet_m1_c1 = TGConv2d(8, 8, 3, padding=1, bias=False)
    unet_m1_gn2 = TGGroupNorm(2, 8, eps=1e-5)
    unet_m1_c2 = TGConv2d(8, 8, 3, padding=1, bias=False)
    unet_m1_mlp = rng.normal(0, 0.5, size=(16, 8)).astype(np.float32)
    unet_m1_mlp_t = Tensor(unet_m1_mlp.tolist())
    # mid attn
    unet_mid_gn = TGGroupNorm(2, 8, eps=1e-5)
    unet_mid_qkv = rng.normal(0, 0.5, size=(8, 24)).astype(np.float32)
    unet_mid_qkv_t = Tensor(unet_mid_qkv.tolist())
    unet_mid_proj = rng.normal(0, 0.5, size=(8, 8)).astype(np.float32)
    unet_mid_proj_t = Tensor(unet_mid_proj.tolist())
    unet_m2_gn1 = TGGroupNorm(2, 8, eps=1e-5)
    unet_m2_c1 = TGConv2d(8, 8, 3, padding=1, bias=False)
    unet_m2_gn2 = TGGroupNorm(2, 8, eps=1e-5)
    unet_m2_c2 = TGConv2d(8, 8, 3, padding=1, bias=False)
    unet_m2_mlp = rng.normal(0, 0.5, size=(16, 8)).astype(np.float32)
    unet_m2_mlp_t = Tensor(unet_m2_mlp.tolist())
    # up0 rb 16->8 (concat 8+8)
    unet_u0_gn1 = TGGroupNorm(2, 16, eps=1e-5)
    unet_u0_c1 = TGConv2d(16, 8, 3, padding=1, bias=False)
    unet_u0_gn2 = TGGroupNorm(2, 8, eps=1e-5)
    unet_u0_c2 = TGConv2d(8, 8, 3, padding=1, bias=False)
    unet_u0_mlp = rng.normal(0, 0.5, size=(16, 8)).astype(np.float32)
    unet_u0_mlp_t = Tensor(unet_u0_mlp.tolist())
    unet_u0_skip = TGConv2d(16, 8, 1, bias=False)
    unet_up0 = TGConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1, bias=False)
    # up1 rb 8->4
    unet_u1_gn1 = TGGroupNorm(2, 8, eps=1e-5)
    unet_u1_c1 = TGConv2d(8, 4, 3, padding=1, bias=False)
    unet_u1_gn2 = TGGroupNorm(2, 4, eps=1e-5)
    unet_u1_c2 = TGConv2d(4, 4, 3, padding=1, bias=False)
    unet_u1_mlp = rng.normal(0, 0.5, size=(16, 4)).astype(np.float32)
    unet_u1_mlp_t = Tensor(unet_u1_mlp.tolist())
    unet_u1_skip = TGConv2d(8, 4, 1, bias=False)
    unet_out_gn = TGGroupNorm(2, 4, eps=1e-5)
    unet_out_c = TGConv2d(4, 2, 3, padding=1, bias=True)

    def unet_rb(xt, tt, gn1, c1, gn2, c2, mlp_t, skip):
        h = gn1(xt).silu()
        h = c1(h)
        h = gn2(h).silu()
        h = c2(h)
        h = h + tt.matmul(mlp_t).reshape(xt.shape[0], h.shape[1], 1, 1)
        if skip is not None:
            return h + skip(xt)
        return h + xt

    def unet_sa(xt, gn, qkv_t, proj_t):
        h = gn(xt)
        B, C, H, W = xt.shape
        HW = H*W
        bhwc = h.permute(0,2,3,1).reshape(B, HW, C)
        flat = bhwc.reshape(B*HW, C)
        qkvFlat = flat.matmul(qkv_t)
        qkv3d = qkvFlat.reshape(B, HW, 3*C)
        q, k, v = qkv3d.chunk(3, dim=2)
        scale = 1 / np.sqrt(C)
        scores = (q @ k.transpose(1,2)) * scale
        attn = scores.softmax(axis=-1)
        out = attn @ v
        outFlat = out.reshape(B*HW, C)
        projFlat = outFlat.matmul(proj_t)
        proj3d = projFlat.reshape(B, HW, C)
        projNchw = proj3d.reshape(B, H, W, C).permute(0,3,1,2)
        return projNchw + xt

    def unet_forward(xt, tt):
        skips = []
        h = unet_stem(xt)
        h = unet_rb(h, tt, unet_d0_gn1, unet_d0_c1, unet_d0_gn2, unet_d0_c2, unet_d0_mlp_t, None)
        skips.append(h)
        h = unet_down0(h)
        h = unet_rb(h, tt, unet_d1_gn1, unet_d1_c1, unet_d1_gn2, unet_d1_c2, unet_d1_mlp_t, None)
        skips.append(h)
        h = unet_rb(h, tt, unet_m1_gn1, unet_m1_c1, unet_m1_gn2, unet_m1_c2, unet_m1_mlp_t, None)
        h = unet_sa(h, unet_mid_gn, unet_mid_qkv_t, unet_mid_proj_t)
        h = unet_rb(h, tt, unet_m2_gn1, unet_m2_c1, unet_m2_gn2, unet_m2_c2, unet_m2_mlp_t, None)
        skip = skips.pop()
        h = h.cat(skip, dim=1)
        h = unet_rb(h, tt, unet_u0_gn1, unet_u0_c1, unet_u0_gn2, unet_u0_c2, unet_u0_mlp_t, unet_u0_skip)
        h = unet_up0(h)
        skip = skips.pop()
        h = h.cat(skip, dim=1)
        h = unet_rb(h, tt, unet_u1_gn1, unet_u1_c1, unet_u1_gn2, unet_u1_c2, unet_u1_mlp_t, unet_u1_skip)
        h = unet_out_gn(h).silu()
        h = unet_out_c(h)
        return h

    unet_xt = Tensor(unet_x.tolist())
    unet_tt = Tensor(unet_t.tolist())
    unet_yt = unet_forward(unet_xt, unet_tt)
    unet_y = unet_yt.numpy().astype(np.float64).reshape(-1)
    # DDPM MSE loss + grad (e2e)
    unet_noise = rng.normal(0, 1, size=(2, 2, 8, 8)).astype(np.float32)
    unet_noise_t = Tensor(unet_noise.tolist())
    unet_loss = ((unet_yt - unet_noise_t) * (unet_yt - unet_noise_t)).mean()
    unet_loss_val = float(unet_loss.numpy())
    unet_loss_grads = unet_loss.gradient(unet_stem.weight, unet_out_c.weight)
    # DDPM sampler single step reference (t=1)
    samp_betas = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    samp_alphas = np.float32(1.0) - samp_betas
    samp_alphaBars = np.empty_like(samp_betas)
    prod = np.float32(1.0)
    for i, a in enumerate(samp_alphas):
        prod = np.float32(prod * np.float32(a))
        samp_alphaBars[i] = prod
    samp_x = rng.normal(0, 1, size=(2,2,2,2)).astype(np.float32)
    samp_eps = rng.normal(0, 1, size=(2,2,2,2)).astype(np.float32)
    samp_t = 1
    samp_beta = np.float32(samp_betas[samp_t])
    samp_alpha = np.float32(samp_alphas[samp_t])
    samp_alphaBar = np.float32(samp_alphaBars[samp_t])
    samp_coef1 = np.float32(1.0 / np.sqrt(float(samp_alpha)))
    samp_coef2 = np.float32(float(samp_beta) / np.sqrt(float(np.float32(1.0) - samp_alphaBar)))
    samp_scaled = (samp_eps.astype(np.float64) * float(samp_coef2)).astype(np.float32)
    samp_sub = (samp_x.astype(np.float64) - samp_scaled.astype(np.float64)).astype(np.float32)
    samp_out = (samp_sub.astype(np.float64) * float(samp_coef1)).astype(np.float32)

    # VAE tiny (8x8 -> 4x4 latent) forward reference
    vae_x = rng.normal(0, 1, size=(2, 2, 8, 8)).astype(np.float32)
    vae_eps = rng.normal(0, 1, size=(2, 4, 4, 4)).astype(np.float32)
    vae_stem = TGConv2d(2, 4, 3, padding=1, bias=False)
    vae_enc_gn1 = TGGroupNorm(2, 4, eps=1e-5)
    vae_enc_c1 = TGConv2d(4, 4, 3, padding=1, bias=False)
    vae_enc_gn2 = TGGroupNorm(2, 4, eps=1e-5)
    vae_enc_c2 = TGConv2d(4, 4, 3, padding=1, bias=False)
    vae_down = TGConv2d(4, 4, 3, stride=2, padding=1, bias=False)
    vae_enc2_gn1 = TGGroupNorm(2, 4, eps=1e-5)
    vae_enc2_c1 = TGConv2d(4, 4, 3, padding=1, bias=False)
    vae_enc2_gn2 = TGGroupNorm(2, 4, eps=1e-5)
    vae_enc2_c2 = TGConv2d(4, 4, 3, padding=1, bias=False)
    vae_mu_conv = TGConv2d(4, 4, 1, bias=True)
    vae_logvar_conv = TGConv2d(4, 4, 1, bias=True)
    vae_dec_gn1 = TGGroupNorm(2, 4, eps=1e-5)
    vae_dec_c1 = TGConv2d(4, 4, 3, padding=1, bias=False)
    vae_dec_gn2 = TGGroupNorm(2, 4, eps=1e-5)
    vae_dec_c2 = TGConv2d(4, 4, 3, padding=1, bias=False)
    vae_up = TGConvTranspose2d(4, 4, 3, stride=2, padding=1, output_padding=1, bias=False)
    vae_dec2_gn1 = TGGroupNorm(2, 4, eps=1e-5)
    vae_dec2_c1 = TGConv2d(4, 4, 3, padding=1, bias=False)
    vae_dec2_gn2 = TGGroupNorm(2, 4, eps=1e-5)
    vae_dec2_c2 = TGConv2d(4, 4, 3, padding=1, bias=False)
    vae_out_gn = TGGroupNorm(2, 4, eps=1e-5)
    vae_out_c = TGConv2d(4, 2, 3, padding=1, bias=True)

    def vae_rb(xt, gn1, c1, gn2, c2):
        h = gn1(xt).silu()
        h = c1(h)
        h = gn2(h).silu()
        h = c2(h)
        return h + xt

    def vae_encode(xt):
        h = vae_stem(xt)
        h = vae_rb(h, vae_enc_gn1, vae_enc_c1, vae_enc_gn2, vae_enc_c2)
        h = vae_down(h)
        h = vae_rb(h, vae_enc2_gn1, vae_enc2_c1, vae_enc2_gn2, vae_enc2_c2)
        mu = vae_mu_conv(h)
        logvar = vae_logvar_conv(h)
        return mu, logvar

    def vae_decode(zt):
        h = vae_rb(zt, vae_dec_gn1, vae_dec_c1, vae_dec_gn2, vae_dec_c2)
        h = vae_up(h)
        h = vae_rb(h, vae_dec2_gn1, vae_dec2_c1, vae_dec2_gn2, vae_dec2_c2)
        h = vae_out_gn(h).silu()
        h = vae_out_c(h)
        return h

    vae_xt = Tensor(vae_x.tolist())
    vae_eps_t = Tensor(vae_eps.tolist())
    vae_mu_t, vae_logvar_t = vae_encode(vae_xt)
    vae_std_t = (vae_logvar_t * 0.5).exp()
    vae_z_t = vae_mu_t + vae_std_t * vae_eps_t
    vae_recon_t = vae_decode(vae_z_t)
    vae_recon = vae_recon_t.numpy().astype(np.float64).reshape(-1)
    vae_mu = vae_mu_t.numpy().astype(np.float64).reshape(-1)
    vae_logvar = vae_logvar_t.numpy().astype(np.float64).reshape(-1)
    vae_z = vae_z_t.numpy().astype(np.float64).reshape(-1)
    # grad check for encoder stem
    vae_loss = ((vae_recon_t - vae_xt) * (vae_recon_t - vae_xt)).mean()
    vae_loss_val = float(vae_loss.numpy())
    vae_grads = vae_loss.gradient(vae_stem.weight, vae_out_c.weight)
    vae_g_stem, vae_g_out = [g.numpy().astype(np.float64).reshape(-1) for g in vae_grads]
    # alternative direct float32 via numpy: ensure same as JS scale/add
    # recompute via same float32 path as JS does (scale then sub then scale)
    # The above already mimics JS float32
    unet_loss_g_stem, unet_loss_g_out = [g.numpy().astype(np.float64).reshape(-1) for g in unet_loss_grads]

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
        "pad_x": pad_x.astype(float).reshape(-1).tolist(),
        "pad_x_shape": list(pad_x.shape),
        "pad_pads": [[0, 0], [0, 0], [1, 1], [1, 1]],
        "pad_y": pad_y.numpy().astype(np.float64).reshape(-1).tolist(),
        "pad_g": pad_g.astype(float).reshape(-1).tolist(),
        "pad_x_grad": pad_x_grad.numpy().astype(np.float64).reshape(-1).tolist(),
        "pad2_x": pad2_x.astype(float).reshape(-1).tolist(),
        "pad2_x_shape": list(pad2_x.shape),
        "pad2_pads": [[0, 0], [0, 0], [1, 2], [0, 3]],
        "pad2_y": pad2_y.numpy().astype(np.float64).reshape(-1).tolist(),
        "pad2_g": pad2_g.astype(float).reshape(-1).tolist(),
        "pad2_x_grad": pad2_x_grad.numpy().astype(np.float64).reshape(-1).tolist(),
        "cat_a": cat_a.astype(float).reshape(-1).tolist(),
        "cat_a_shape": list(cat_a.shape),
        "cat_b": cat_b.astype(float).reshape(-1).tolist(),
        "cat_b_shape": list(cat_b.shape),
        "cat_axis": 1,
        "cat_y": cat_y.numpy().astype(np.float64).reshape(-1).tolist(),
        "cat_g": cat_g.astype(float).reshape(-1).tolist(),
        "cat_a_grad": cat_a_grad.numpy().astype(np.float64).reshape(-1).tolist(),
        "cat_b_grad": cat_b_grad.numpy().astype(np.float64).reshape(-1).tolist(),
        "cat2_a": cat2_a.astype(float).reshape(-1).tolist(),
        "cat2_a_shape": list(cat2_a.shape),
        "cat2_b": cat2_b.astype(float).reshape(-1).tolist(),
        "cat2_b_shape": list(cat2_b.shape),
        "cat2_axis": 3,
        "cat2_y": cat2_y.numpy().astype(np.float64).reshape(-1).tolist(),
        "cat2_g": cat2_g.astype(float).reshape(-1).tolist(),
        "cat2_a_grad": cat2_a_grad.numpy().astype(np.float64).reshape(-1).tolist(),
        "cat2_b_grad": cat2_b_grad.numpy().astype(np.float64).reshape(-1).tolist(),
        "timesteps": ts,
        "ste_dim": 16,
        "ste_ref": ste.reshape(-1).tolist(),
        "rb_x": rb_x.astype(float).reshape(-1).tolist(),
        "rb_x_shape": list(rb_x.shape),
        "rb_t": rb_t.astype(float).reshape(-1).tolist(),
        "rb_gn1w": rb_gn1.weight.numpy().astype(float).reshape(-1).tolist(),
        "rb_gn1b": rb_gn1.bias.numpy().astype(float).reshape(-1).tolist(),
        "rb_conv1w": rb_conv1.weight.numpy().astype(float).reshape(-1).tolist(),
        "rb_gn2w": rb_gn2.weight.numpy().astype(float).reshape(-1).tolist(),
        "rb_gn2b": rb_gn2.bias.numpy().astype(float).reshape(-1).tolist(),
        "rb_conv2w": rb_conv2.weight.numpy().astype(float).reshape(-1).tolist(),
        "rb_mlpw": rb_mlp_w.astype(float).reshape(-1).tolist(),
        "rb_mlpw_shape": [16, 12],
        "rb_skipw": rb_skip.weight.numpy().astype(float).reshape(-1).tolist(),
        "rb_y": rb_y.tolist(),
        "rb_gout": rb_gout.astype(float).reshape(-1).tolist(),
        "rb_x_grad": rb_x_grad.tolist(),
        "rb_gn1w_grad": rb_gn1w_grad.tolist(),
        "rb_gn1b_grad": rb_gn1b_grad.tolist(),
        "rb_conv1w_grad": rb_conv1w_grad.tolist(),
        "rb_gn2w_grad": rb_gn2w_grad.tolist(),
        "rb_gn2b_grad": rb_gn2b_grad.tolist(),
        "rb_conv2w_grad": rb_conv2w_grad.tolist(),
        "rb_mlp_grad": rb_mlp_grad.tolist(),
        "rb_skipw_grad": rb_skipw_grad.tolist(),
        "sa_x": sa_x.astype(float).reshape(-1).tolist(),
        "sa_x_shape": list(sa_x.shape),
        "sa_gnw": sa_gn.weight.numpy().astype(float).reshape(-1).tolist(),
        "sa_gnb": sa_gn.bias.numpy().astype(float).reshape(-1).tolist(),
        "sa_qkvw": sa_qkv_w.astype(float).reshape(-1).tolist(),
        "sa_projw": sa_proj_w.astype(float).reshape(-1).tolist(),
        "sa_y": sa_y.tolist(),
        "sa_gout": sa_gout.astype(float).reshape(-1).tolist(),
        "sa_x_grad": sa_x_grad.tolist(),
        "sa_gnw_grad": sa_gnw_grad.tolist(),
        "sa_gnb_grad": sa_gnb_grad.tolist(),
        "sa_qkv_grad": sa_qkv_grad.tolist(),
        "sa_proj_grad": sa_proj_grad.tolist(),
        "unet_x": unet_x.astype(float).reshape(-1).tolist(),
        "unet_x_shape": list(unet_x.shape),
        "unet_t": unet_t.astype(float).reshape(-1).tolist(),
        "unet_stem_w": unet_stem.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_d0_gn1w": unet_d0_gn1.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_d0_gn1b": unet_d0_gn1.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_d0_c1w": unet_d0_c1.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_d0_gn2w": unet_d0_gn2.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_d0_gn2b": unet_d0_gn2.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_d0_c2w": unet_d0_c2.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_d0_mlp": unet_d0_mlp.astype(float).reshape(-1).tolist(),
        "unet_down0w": unet_down0.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_d1_gn1w": unet_d1_gn1.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_d1_gn1b": unet_d1_gn1.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_d1_c1w": unet_d1_c1.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_d1_gn2w": unet_d1_gn2.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_d1_gn2b": unet_d1_gn2.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_d1_c2w": unet_d1_c2.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_d1_mlp": unet_d1_mlp.astype(float).reshape(-1).tolist(),
        "unet_m1_gn1w": unet_m1_gn1.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_m1_gn1b": unet_m1_gn1.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_m1_c1w": unet_m1_c1.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_m1_gn2w": unet_m1_gn2.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_m1_gn2b": unet_m1_gn2.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_m1_c2w": unet_m1_c2.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_m1_mlp": unet_m1_mlp.astype(float).reshape(-1).tolist(),
        "unet_mid_gnw": unet_mid_gn.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_mid_gnb": unet_mid_gn.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_mid_qkv": unet_mid_qkv.astype(float).reshape(-1).tolist(),
        "unet_mid_proj": unet_mid_proj.astype(float).reshape(-1).tolist(),
        "unet_m2_gn1w": unet_m2_gn1.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_m2_gn1b": unet_m2_gn1.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_m2_c1w": unet_m2_c1.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_m2_gn2w": unet_m2_gn2.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_m2_gn2b": unet_m2_gn2.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_m2_c2w": unet_m2_c2.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_m2_mlp": unet_m2_mlp.astype(float).reshape(-1).tolist(),
        "unet_u0_gn1w": unet_u0_gn1.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_u0_gn1b": unet_u0_gn1.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_u0_c1w": unet_u0_c1.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_u0_gn2w": unet_u0_gn2.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_u0_gn2b": unet_u0_gn2.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_u0_c2w": unet_u0_c2.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_u0_mlp": unet_u0_mlp.astype(float).reshape(-1).tolist(),
        "unet_u0_skipw": unet_u0_skip.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_up0w": unet_up0.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_u1_gn1w": unet_u1_gn1.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_u1_gn1b": unet_u1_gn1.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_u1_c1w": unet_u1_c1.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_u1_gn2w": unet_u1_gn2.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_u1_gn2b": unet_u1_gn2.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_u1_c2w": unet_u1_c2.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_u1_mlp": unet_u1_mlp.astype(float).reshape(-1).tolist(),
        "unet_u1_skipw": unet_u1_skip.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_out_gnw": unet_out_gn.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_out_gnb": unet_out_gn.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_out_cw": unet_out_c.weight.numpy().astype(float).reshape(-1).tolist(),
        "unet_out_cb": unet_out_c.bias.numpy().astype(float).reshape(-1).tolist(),
        "unet_y": unet_y.tolist(),
        "unet_noise": unet_noise.astype(float).reshape(-1).tolist(),
        "unet_loss": float(unet_loss_val),
        "unet_loss_g_stem": unet_loss_g_stem.tolist(),
        "unet_loss_g_out": unet_loss_g_out.tolist(),
        "samp_betas": samp_betas.astype(float).tolist(),
        "samp_x": samp_x.astype(float).reshape(-1).tolist(),
        "samp_eps": samp_eps.astype(float).reshape(-1).tolist(),
        "samp_t": int(samp_t),
        "samp_out": samp_out.astype(float).reshape(-1).tolist(),
        "vae_x": vae_x.astype(float).reshape(-1).tolist(),
        "vae_eps": vae_eps.astype(float).reshape(-1).tolist(),
        "vae_stem_w": vae_stem.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_enc_gn1w": vae_enc_gn1.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_enc_gn1b": vae_enc_gn1.bias.numpy().astype(float).reshape(-1).tolist(),
        "vae_enc_c1w": vae_enc_c1.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_enc_gn2w": vae_enc_gn2.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_enc_gn2b": vae_enc_gn2.bias.numpy().astype(float).reshape(-1).tolist(),
        "vae_enc_c2w": vae_enc_c2.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_down_w": vae_down.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_enc2_gn1w": vae_enc2_gn1.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_enc2_gn1b": vae_enc2_gn1.bias.numpy().astype(float).reshape(-1).tolist(),
        "vae_enc2_c1w": vae_enc2_c1.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_enc2_gn2w": vae_enc2_gn2.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_enc2_gn2b": vae_enc2_gn2.bias.numpy().astype(float).reshape(-1).tolist(),
        "vae_enc2_c2w": vae_enc2_c2.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_mu_w": vae_mu_conv.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_mu_b": vae_mu_conv.bias.numpy().astype(float).reshape(-1).tolist(),
        "vae_logvar_w": vae_logvar_conv.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_logvar_b": vae_logvar_conv.bias.numpy().astype(float).reshape(-1).tolist(),
        "vae_dec_gn1w": vae_dec_gn1.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_dec_gn1b": vae_dec_gn1.bias.numpy().astype(float).reshape(-1).tolist(),
        "vae_dec_c1w": vae_dec_c1.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_dec_gn2w": vae_dec_gn2.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_dec_gn2b": vae_dec_gn2.bias.numpy().astype(float).reshape(-1).tolist(),
        "vae_dec_c2w": vae_dec_c2.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_up_w": vae_up.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_dec2_gn1w": vae_dec2_gn1.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_dec2_gn1b": vae_dec2_gn1.bias.numpy().astype(float).reshape(-1).tolist(),
        "vae_dec2_c1w": vae_dec2_c1.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_dec2_gn2w": vae_dec2_gn2.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_dec2_gn2b": vae_dec2_gn2.bias.numpy().astype(float).reshape(-1).tolist(),
        "vae_dec2_c2w": vae_dec2_c2.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_out_gnw": vae_out_gn.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_out_gnb": vae_out_gn.bias.numpy().astype(float).reshape(-1).tolist(),
        "vae_out_cw": vae_out_c.weight.numpy().astype(float).reshape(-1).tolist(),
        "vae_out_cb": vae_out_c.bias.numpy().astype(float).reshape(-1).tolist(),
        "vae_recon": vae_recon.tolist(),
        "vae_mu": vae_mu.tolist(),
        "vae_logvar": vae_logvar.tolist(),
        "vae_z": vae_z.tolist(),
        "vae_loss": float(vae_loss_val),
        "vae_g_stem": vae_g_stem.tolist(),
        "vae_g_out": vae_g_out.tolist(),
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
        "pad": report["pad_fwd"] < 1e-5 and report["pad_grad"] < 1e-5,
        "pad_asym": report["pad2_fwd"] < 1e-5 and report["pad2_grad"] < 1e-5,
        "concat": report["cat_fwd"] < 1e-5 and report["cat_a_grad"] < 1e-5 and report["cat_b_grad"] < 1e-5,
        "concat_last_axis": report["cat2_fwd"] < 1e-5 and report["cat2_a_grad"] < 1e-5 and report["cat2_b_grad"] < 1e-5,
        "sinusoidal_timestep": report["ste"] < 1e-6,
        "resblock": report["rb_fwd"] < 1e-4 and max(
            report["rb_grad_x"], report["rb_grad_gn1w"], report["rb_grad_gn1b"],
            report["rb_grad_conv1w"], report["rb_grad_gn2w"], report["rb_grad_gn2b"],
            report["rb_grad_conv2w"], report["rb_grad_mlpw"], report["rb_grad_skipw"]) < 1e-4,
        "spatial_attention": report["sa_fwd"] < 1e-4 and max(
            report["sa_grad_x"], report["sa_gnw"], report["sa_gnb"], report["sa_qkv"], report["sa_proj"]) < 1e-4,
        "unet": report["unet_fwd"] < 1e-4,
        "unet_loss": report["unet_loss_err"] < 1e-4 and report["unet_loss_g_stem"] < 1e-4 and report["unet_loss_g_out"] < 1e-4,
        "sampler": report["samp_fwd"] < 1e-5,
        "vae": report["vae_recon"] < 1e-4 and report["vae_mu"] < 1e-4 and report["vae_logvar"] < 1e-4 and report["vae_z"] < 1e-4 and report["vae_loss_err"] < 1e-4 and report["vae_g_stem"] < 1e-4 and report["vae_g_out"] < 1e-4,
    }
    print("verdict", {k: ("PASS" if v else "FAIL") for k, v in checks.items()})
    print("nn", report["nn_modules"], "optim", report["optim_exports"])
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
