#!/usr/bin/env python3
"""Minify web JS with terser (compress + mangle), keep .min.js + .map."""
from pathlib import Path
import subprocess, gzip, shutil

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "gabion/web"
FILES = ["tinygrad_v0.js", "bbt_forward.js", "webgpu_backend.js", "model_loader.js", "tokenizer.js"]

def has_terser():
    return shutil.which("terser") or shutil.which("npx") is not None

def minify(src: Path, dst: Path):
    # use npx terser to avoid global install (Windows needs shell for .cmd)
    cmd = f'npx terser "{src}" --compress --mangle --output "{dst}" --source-map "includeSources,url={dst.name}.map"'
    subprocess.run(cmd, shell=True, check=True)

def main():
    if not has_terser():
        print("terser not found (need node + npx), skipping")
        return
    total_raw = total_min = total_gz = total_gz_min = 0
    for name in FILES:
        src = WEB / name
        dst = WEB / name.replace(".js", ".min.js")
        minify(src, dst)
        raw = src.stat().st_size
        mn = dst.stat().st_size
        gz = len(gzip.compress(src.read_bytes()))
        gz_mn = len(gzip.compress(dst.read_bytes()))
        print(f"{name}: {raw} -> {mn} ({mn/raw*100:.1f}% raw)  gz {gz} -> {gz_mn} ({gz_mn/gz*100:.1f}% gz)")
        total_raw += raw; total_min += mn; total_gz += gz; total_gz_min += gz_mn
    print(f"total JS: {total_raw} -> {total_min} raw, {total_gz} -> {total_gz_min} gz (saved {total_raw-total_min} raw, {total_gz-total_gz_min} gz)")

if __name__ == "__main__":
    main()
