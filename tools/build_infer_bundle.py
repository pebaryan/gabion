#!/usr/bin/env python3
"""Build an inference-only bundle: copy only kernels needed for LLM decode."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "gabion/web/kernels"
DST = ROOT / "gabion/web/kernels/infer"

# Kernels actually dispatched during LLM inference (decode + prefill)
# - training kernels (backward, adam, sgd, conv_bwd, batchnorm_bwd, dropout_bwd, lstm_bwd, etc) are dead
KEEP = {
    "matmul.wgsl",
    "batched_matmul.wgsl",
    "batched_transpose.wgsl",
    "embedding_forward.wgsl",
    "elementwise.wgsl",          # silu etc
    "silu_mul.wgsl",
    "reduce.wgsl",
    "softmax.wgsl",
    "rope.wgsl",
    "fused_attention.wgsl",
    "kv_attention_scores.wgsl",
    "kv_attention_apply.wgsl",
    "kv_group_sum.wgsl",
    # keep layernorm for diffusion inference; rmsnorm is CPU today but keep wgsl if present
    "layernorm.wgsl",
    "rmsnorm_forward.wgsl",
    "pad.wgsl",                  # VAE padding if latent diffusion demo
    "concat.wgsl",
}

DROP_HINT = {
    "adam_update.wgsl","adamw_update.wgsl","sgd_update.wgsl",
    "softmax_backward.wgsl","rope_backward.wgsl","layernorm_backward.wgsl","rmsnorm_backward.wgsl",
    "silu_mul_backward.wgsl","dropout_fwd.wgsl","dropout_bwd.wgsl",
    "cross_entropy_forward.wgsl","cross_entropy_backward.wgsl",
    "conv2d_fwd.wgsl","conv2d_bwd_dx.wgsl","conv2d_bwd_dw.wgsl","conv_bwd_db.wgsl",
    "convtranspose2d_fwd.wgsl","convtranspose2d_bwd_dx.wgsl","convtranspose2d_bwd_dw.wgsl",
    "batchnorm_fwd.wgsl","batchnorm_stats.wgsl","batchnorm_bwd_stats.wgsl","batchnorm_bwd_dx.wgsl",
    "affine_channel.wgsl","affine_channel_bwd_dw.wgsl","affine_last.wgsl","affine_last_bwd_dw.wgsl",
    "lstm_cell.wgsl","lstm_cell_bwd.wgsl","lstm_cell_bwd_dx.wgsl","lstm_cell_bwd_dh.wgsl","lstm_cell_bwd_dwih.wgsl","lstm_cell_bwd_dwhh.wgsl",
    "concat_bwd.wgsl","pad_bwd.wgsl",
}

def main():
    DST.mkdir(parents=True, exist_ok=True)
    # clean old
    for p in DST.glob("*.wgsl"):
        p.unlink()
    kept = 0
    for f in SRC.glob("*.wgsl"):
        if f.name in KEEP:
            shutil.copy2(f, DST / f.name)
            kept += 1
    total = len(list(SRC.glob("*.wgsl")))
    raw_keep = sum((DST / f).stat().st_size for f in KEEP if (DST / f).exists())
    raw_total = sum(p.stat().st_size for p in SRC.glob("*.wgsl"))
    import gzip
    gz_keep = len(gzip.compress(b"".join((DST / f).read_bytes() for f in sorted(DST.glob("*.wgsl")))))
    gz_total = len(gzip.compress(b"".join(p.read_bytes() for p in sorted(SRC.glob("*.wgsl")))))
    print(f"infer kernels: {kept}/{total} files kept")
    print(f"raw: {raw_keep} / {raw_total} bytes ({raw_keep/raw_total*100:.1f}%)")
    print(f"gz:  {gz_keep} / {gz_total} bytes ({gz_keep/gz_total*100:.1f}%)")
    print(f"saving: {raw_total-raw_keep} raw, {gz_total-gz_keep} gz")
    # also report JS inference-only savings: tinygrad_v0.js optimizer stripped not yet done
    js = ROOT / "gabion/web/tinygrad_v0.js"
    print(f"tinygrad_v0.js raw {js.stat().st_size} gz {len(gzip.compress(js.read_bytes()))}")
    # write manifest
    (DST / "MANIFEST.txt").write_text("\n".join(sorted(KEEP)) + "\n")

if __name__ == "__main__":
    main()
