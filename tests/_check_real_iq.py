"""Cross-check the exporter's dequants against gguf-py on REAL files.

Streams via mmap (no full-file reads) so multi-GB files are fine. Checks every
tensor whose type is in the NEW set. Usage:
    python tests/_check_real_iq.py <file.gguf> [--types IQ4_NL Q5_K ...]
"""
import mmap
import struct
import sys

import numpy as np

sys.path.insert(0, "tools")
from export_model import GGML_TYPE, _tensor_data  # noqa: E402

import gguf  # noqa: E402
from gguf.constants import GGMLQuantizationType as QT  # noqa: E402


def read_str(buf, off):
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    return buf[off:off + n].decode("utf-8", "replace"), off + n


def header_scan(path):
    """Parse metadata + tensor infos from the first 64MB (no data section)."""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        (version, n_tensors, n_kv) = struct.unpack_from("<IQQ", f.read(20), 0)
        f.seek(0)
        buf = f.read(64_000_000)
    off = 24
    for _ in range(n_kv):
        _, off = read_str(buf, off)
        (vtype,) = struct.unpack_from("<I", buf, off)
        off += 4
        if vtype == 8:
            _, off = read_str(buf, off)
        elif vtype == 9:
            (etype, n) = struct.unpack_from("<IQ", buf, off)
            off += 12
            for _ in range(n):
                if etype == 8:
                    _, off = read_str(buf, off)
                else:
                    off += {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[etype]
        else:
            off += {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[vtype]
    infos = []
    for _ in range(n_tensors):
        name, off = read_str(buf, off)
        (n_dims,) = struct.unpack_from("<I", buf, off)
        off += 4
        dims = struct.unpack_from(f"<{'Q' * n_dims}", buf, off)
        off += 8 * n_dims
        (gtype, toff) = struct.unpack_from("<IQ", buf, off)
        off += 12
        infos.append((name, tuple(dims), GGML_TYPE.get(gtype, f"T{gtype}"), toff))
    align = 32  # default; real files almost always 32
    # find general.alignment from raw buffer scan would be nicer, but 32 is fine
    # for these files (all written by llama.cpp defaults).
    return infos, (off + align - 1) & ~(align - 1)


def main():
    path = sys.argv[1]
    only = set(sys.argv[sys.argv.index("--types") + 1:]) if "--types" in sys.argv else None
    infos, base = header_scan(path)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)
        by_type = {}
        bad = 0
        total = 0
        for (name, dims, gtype, toff) in infos:
            if only and gtype not in only:
                continue
            if gtype not in by_type:
                by_type[gtype] = [0, 0]
            by_type[gtype][0] += 1
            try:
                got = _tensor_data(mv, name, dims, gtype, toff, base).reshape(-1)
            except Exception as e:
                print(f"  EXPORT-ERROR {name} ({gtype}): {e}")
                bad += 1
                continue
            n = int(np.prod(dims))
            qname = {"TQ2_0": "TQ2_0", "TQ1_0": "TQ1_0"}.get(gtype, gtype)
            qt = QT[qname]
            raw = mv[base + toff: base + toff + _tensor_size(gtype, n)]
            ref = gguf.quants.dequantize(np.frombuffer(raw, np.uint8), qt).astype(np.float32).reshape(-1)
            total += 1
            if not np.array_equal(got, ref):
                bad += 1
                i = int(np.argwhere(got != ref)[0][0])
                print(f"  MISMATCH {name} ({gtype}) idx {i}: mine={got[i]} ref={ref[i]}")
        for t, (cnt, _) in sorted(by_type.items()):
            print(f"  {t:10s} x{cnt}")
        print(f"TOTAL checked: {total}, bad: {bad}")
        sys.exit(1 if bad else 0)


def _tensor_size(gtype, n):
    sizes = {"F32": 4 * n, "F16": 2 * n, "BF16": 2 * n, "Q4_0": 18 * n // 32, "Q4_1": 20 * n // 32,
             "Q5_0": 22 * n // 32, "Q5_1": 24 * n // 32, "Q8_0": 34 * n // 32,
             "Q2_K": 84 * n // 256, "Q3_K": 110 * n // 256, "Q4_K": 144 * n // 256,
             "Q5_K": 176 * n // 256, "Q6_K": 210 * n // 256, "TQ1_0": 54 * n // 256,
             "TQ2_0": 66 * n // 256, "IQ2_XXS": 66 * n // 256, "IQ2_XS": 74 * n // 256,
             "IQ2_S": 82 * n // 256, "IQ3_XXS": 98 * n // 256, "IQ3_S": 110 * n // 256,
             "IQ1_S": 50 * n // 256, "IQ1_M": 56 * n // 256, "IQ4_NL": 18 * n // 32,
             "IQ4_XS": 136 * n // 256, "MXFP4": 17 * n // 32, "NVFP4": 36 * n // 64}
    return sizes[gtype]


if __name__ == "__main__":
    main()
