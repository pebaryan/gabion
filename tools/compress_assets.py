#!/usr/bin/env python3
"""Compress static web assets with gzip + brotli if available."""
from pathlib import Path
import gzip, subprocess, shutil

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "gabion/web"

PATTERNS = ["**/*.js", "**/*.wgsl", "**/*.json", "**/*.html"]

def has_brotli():
    return shutil.which("brotli") is not None

def compress_file(p: Path):
    data = p.read_bytes()
    # gzip
    gz_path = p.with_suffix(p.suffix + ".gz")
    gz_path.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    # brotli if available
    if has_brotli():
        br_path = p.with_suffix(p.suffix + ".br")
        # quality 11, deterministic
        subprocess.run(["brotli", "-f", "-q", "11", "-o", str(br_path), str(p)], check=False)

def main():
    files = []
    for pat in PATTERNS:
        files.extend(WEB.glob(pat))
    # also compress f16 weights if present (optional, skip if >100M to avoid slow CI)
    for p in files:
        if p.suffix in (".gz", ".br"):
            continue
        if p.stat().st_size > 50_000_000:
            print(f"skip large {p} ({p.stat().st_size} bytes)")
            continue
        size = p.stat().st_size
        compress_file(p)
        gz = p.with_suffix(p.suffix + ".gz").stat().st_size
        br = p.with_suffix(p.suffix + ".br")
        br_size = br.stat().st_size if br.exists() else 0
        print(f"{p.relative_to(ROOT)}: {size} -> gz {gz} ({gz/size*100:.1f}%)" + (f" br {br_size} ({br_size/size*100:.1f}%)" if br_size else ""))

if __name__ == "__main__":
    main()
