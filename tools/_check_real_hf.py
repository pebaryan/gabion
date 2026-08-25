"""Validate export_hf on a REAL HuggingFace checkpoint (Qwen3.6-27B-DFlash-FP16).

The wire's f16 flat is compared element-wise against an independent re-read of
the safetensors source (this script reimplements the tensor order/transforms
from the HF spec, so it is not circular). Runs at ~20GB peak.

Usage: python tools/_check_real_hf.py <model_dir>
"""
import base64
import json
import mmap
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_model import export_hf  # noqa: E402

# tensor order + transforms, mirroring export_hf (llama-family)
LAYER_PARTS = [
    ("self_attn.q_proj.weight", "T"),
    ("self_attn.k_proj.weight", "T"),
    ("self_attn.v_proj.weight", "T"),
    ("self_attn.o_proj.weight", "T"),
    ("input_layernorm.weight", "flat"),
    ("mlp.gate_up", "gate_up"),
    ("post_attention_layernorm.weight", "flat"),
    ("mlp.down_proj.weight", "T"),
]


def read_st(path):
    with open(path, "rb") as f:
        (hdr_len,) = struct.unpack_from("<Q", f.read(8))
        hdr = json.loads(f.read(hdr_len))
        f.seek(0)
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return mm, hdr, hdr_len


def get_tensor(mm, hdr_len, hdr, name):
    info = hdr[name]
    start, end = info["data_offsets"]
    if info["dtype"] == "F16":
        arr = np.frombuffer(mm, dtype="<f2", count=(end - start) // 2, offset=8 + hdr_len + start)
    elif info["dtype"] == "F32":
        arr = np.frombuffer(mm, dtype="<f4", count=(end - start) // 4, offset=8 + hdr_len + start)
    elif info["dtype"] == "BF16":
        u = np.frombuffer(mm, dtype="<u2", count=(end - start) // 2, offset=8 + hdr_len + start)
        arr = (u.astype(np.int32) << 16).view(np.float32)
    else:
        raise NotImplementedError(info["dtype"])
    return arr.reshape(info["shape"])


def main():
    d = Path(sys.argv[1])
    cfg = json.loads((d / "config.json").read_text())
    st_path = d / "model.safetensors"
    idx_path = d / "model.safetensors.index.json"
    if idx_path.is_file():
        wmap = json.loads(idx_path.read_text())["weight_map"]
    else:
        wmap = None
    shards = {}

    def src(name):
        spath = st_path if wmap is None else d / wmap[name]
        if spath not in shards:
            shards[spath] = read_st(spath)
        mm, hdr, hdr_len = shards[spath]
        return get_tensor(mm, hdr_len, hdr, name)

    # ---- run the real export first (its flat is freed on return) ----
    print("[1/3] export_hf ...")
    out = export_hf(d)
    print(f"      config: d={out['config']['d_model']} L={out['config']['n_layers']} "
          f"heads={out['config']['n_heads']}/{out['config']['n_kv_heads']} "
          f"vocab={out['config']['vocab_size']} tie={out['config']['tie_weights']} "
          f"rope={out['config']['rope_base']} seq={out['config']['seq_len']}")

    # ---- independent reference flat ----
    print("[2/3] independent re-read ...")
    L = int(cfg.get("num_hidden_layers", cfg["text_config"]["num_hidden_layers"]) if "text_config" in cfg else cfg["num_hidden_layers"])
    refs = [src("model.embed_tokens.weight")]
    for i in range(L):
        for part, mode in LAYER_PARTS:
            if mode == "gate_up":
                gate = src(f"model.layers.{i}.mlp.gate_proj.weight").T
                up = src(f"model.layers.{i}.mlp.up_proj.weight").T
                refs.append(np.concatenate([gate, up], axis=1))
            else:
                t = src(f"model.layers.{i}.{part}")
                refs.append(t.T if mode == "T" else t.reshape(-1))
    refs.append(src("model.norm.weight").reshape(-1))
    if not cfg.get("tie_word_embeddings", True):
        refs.append(src("lm_head.weight").T)
    ref16 = np.concatenate([np.asarray(t).astype(np.float16).reshape(-1) for t in refs])
    del refs

    # ---- compare in chunks (b64 -> f16 slices); chunks are 6-byte aligned so
    # both the base64 quads and the f16 element grid stay aligned ----
    print("[3/3] comparing wire vs source (f16) ...")
    b64 = out["weights_b64"]
    n = ref16.size
    assert len(b64) == ((n * 2 + 2) // 3) * 4, "wire length mismatch"
    ok = True
    CH = 3_000_000  # elements = 6MB, multiple of 3 -> quad-aligned
    for off in range(0, n, CH):
        end = min(off + CH, n)
        c0 = (off * 2) // 3 * 4
        c1 = (end * 2 + 2) // 3 * 4
        raw = base64.b64decode(b64[c0:c1] + "=" * ((4 - (c1 - c0) % 4) % 4))
        got = np.frombuffer(raw, dtype="<f2")[:end - off]
        ref = ref16[off:end]
        if not np.array_equal(got, ref):
            i = int(np.argwhere(got != ref)[0][0])
            print(f"  MISMATCH at {off + i}: wire={got[i]} src={ref[i]}")
            ok = False
            break
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} ({n:,} f16 weights, "
          f"{len(out.get('vocab', {}))} vocab tokens)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
