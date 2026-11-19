"""Print every column of a retargeted-motion parquet with its shape and a sample value.

Usage: python tools/learn/inspect_motion.py <file.parquet | partition_dir>
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

path = Path(sys.argv[1])
files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
for f in files:
    t = pq.read_table(f)
    print(f"== {f}  rows={t.num_rows} cols={t.num_columns}")
    for name in t.column_names:
        v = t.column(name)[0].as_py()
        try:
            arr = np.asarray(v)
            shape = arr.shape
        except Exception:
            arr, shape = None, "ragged"
        sample = ""
        if arr is not None and arr.dtype.kind in "U" and arr.size <= 64:
            sample = ", ".join(map(str, arr.ravel()[:8])) + (" ..." if arr.size > 8 else "")
        elif arr is not None and arr.ndim == 0:
            sample = str(v)[:70]
        elif arr is not None and arr.dtype.kind == "f" and arr.size:
            sample = f"min={np.nanmin(arr):.3f} max={np.nanmax(arr):.3f}"
        print(f"  {name:40s} {str(shape):18s} {sample}")
