"""Compare one module call's outputs (reference tree vs candidate tree), file by file.

Stdlib + numpy (+ PIL when present). Runs inside the module image, so it must not import
anything else. Returns a flat dict of worst-case metrics so a smoke result is one row.

Metrics
  files / missing / identical      counts over the reference call's output files
  json_max_abs, json_p99_abs        numeric leaves of JSON files (abs difference)
  json_struct_diff                  JSON leaves that differ in type/shape/non-numeric value
  img_mask_err_max                  1 - IoU, for binary masks (0 = identical)
  img_mean_rel_max                  mean |a-b| / dtype range, other images (depth, RGB)
  arr_max_abs                       .npy/.npz arrays
  obj_max_vertex                    max vertex displacement (same topology) else chamfer
  bytes_diff                        other files that differ byte-for-byte
"""

from __future__ import annotations

import fnmatch
import json
import math
import os

import numpy as np

SKIP = ("*.mp4", "*/refined_simple_progress/*", "*_progress/*")  # diagnostics, never compared


def _skip(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch("/" + rel, pattern) for pattern in SKIP)


def _json_leaves(value, path=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _json_leaves(value[key], f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _json_leaves(item, f"{path}[{index}]")
    else:
        yield path, value


def _cmp_json(a: str, b: str, acc: dict) -> bool:
    with open(a) as fa, open(b) as fb:
        ja, jb = json.load(fa), json.load(fb)
    la, lb = dict(_json_leaves(ja)), dict(_json_leaves(jb))
    diffs = []
    struct = len(set(la) ^ set(lb))
    for key in set(la) & set(lb):
        va, vb = la[key], lb[key]
        if isinstance(va, bool) or isinstance(vb, bool) or not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            struct += int(va != vb)
            continue
        if math.isnan(va) and math.isnan(vb):
            continue
        diffs.append(abs(float(va) - float(vb)))
    if diffs:
        d = np.asarray(diffs)
        acc["json_max_abs"] = max(acc.get("json_max_abs", 0.0), float(d.max()))
        acc["json_p99_abs"] = max(acc.get("json_p99_abs", 0.0), float(np.percentile(d, 99)))
    acc["json_struct_diff"] = acc.get("json_struct_diff", 0) + struct
    return struct == 0 and (not diffs or max(diffs) == 0.0)


def _load_img(path: str) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415

    return np.asarray(Image.open(path))


def _cmp_img(a: str, b: str, acc: dict) -> bool:
    ia, ib = _load_img(a), _load_img(b)
    if ia.shape != ib.shape:
        acc["img_shape_diff"] = acc.get("img_shape_diff", 0) + 1
        return False
    if np.array_equal(ia, ib):
        return True
    uniq = np.unique(ia)
    if uniq.size <= 2:
        ma, mb = ia > 0, ib > 0
        union = np.logical_or(ma, mb).sum()
        err = 0.0 if union == 0 else 1.0 - np.logical_and(ma, mb).sum() / union
        acc["img_mask_err_max"] = max(acc.get("img_mask_err_max", 0.0), float(err))
    else:
        span = float(np.iinfo(ia.dtype).max) if np.issubdtype(ia.dtype, np.integer) else 1.0
        rel = float(np.abs(ia.astype(np.float64) - ib.astype(np.float64)).mean() / span)
        acc["img_mean_rel_max"] = max(acc.get("img_mean_rel_max", 0.0), rel)
    return False


def _cmp_arr(a: str, b: str, acc: dict) -> bool:
    def load(path):
        data = np.load(path, allow_pickle=False)
        return {k: data[k] for k in data.files} if hasattr(data, "files") else {"": data}

    da, db = load(a), load(b)
    same = True
    for key in set(da) | set(db):
        if key not in da or key not in db or da[key].shape != db[key].shape:
            acc["arr_struct_diff"] = acc.get("arr_struct_diff", 0) + 1
            same = False
            continue
        if da[key].dtype.kind in "fc" or db[key].dtype.kind in "fc":
            d = float(np.nanmax(np.abs(da[key].astype(np.float64) - db[key].astype(np.float64)))) if da[key].size else 0.0
            acc["arr_max_abs"] = max(acc.get("arr_max_abs", 0.0), d)
            same &= d == 0.0
        else:
            eq = bool(np.array_equal(da[key], db[key]))
            acc["arr_struct_diff"] = acc.get("arr_struct_diff", 0) + int(not eq)
            same &= eq
    return same


def _obj_vertices(path: str) -> np.ndarray:
    verts = []
    with open(path, errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3)


def _cmp_obj(a: str, b: str, acc: dict) -> bool:
    va, vb = _obj_vertices(a), _obj_vertices(b)
    if va.shape == vb.shape:
        d = float(np.abs(va - vb).max()) if va.size else 0.0
    else:  # different topology: symmetric chamfer on subsamples (brute force, small)
        rng = np.random.default_rng(0)
        sa = va[rng.choice(len(va), min(len(va), 3000), replace=False)]
        sb = vb[rng.choice(len(vb), min(len(vb), 3000), replace=False)]
        dab = np.sqrt(((sa[:, None] - sb[None]) ** 2).sum(-1)).min(1).mean()
        dba = np.sqrt(((sb[:, None] - sa[None]) ** 2).sum(-1)).min(1).mean()
        d = float((dab + dba) / 2)
        acc["obj_topology_diff"] = acc.get("obj_topology_diff", 0) + 1
    acc["obj_max_vertex"] = max(acc.get("obj_max_vertex", 0.0), d)
    return d == 0.0


def compare_trees(ref_root: str, cand_root: str, rel_paths: list[str]) -> dict:
    acc: dict = {"files": 0, "missing": 0, "identical": 0, "errors": 0}
    for rel in rel_paths:
        if _skip(rel):
            continue
        a, b = os.path.join(ref_root, rel), os.path.join(cand_root, rel)
        if not os.path.isfile(a):
            continue
        acc["files"] += 1
        if not os.path.isfile(b):
            acc["missing"] += 1
            continue
        with open(a, "rb") as fa, open(b, "rb") as fb:
            if fa.read() == fb.read():
                acc["identical"] += 1
                continue
        ext = os.path.splitext(rel)[1].lower()
        try:
            if ext == ".json":
                same = _cmp_json(a, b, acc)
            elif ext in (".png", ".jpg", ".jpeg"):
                same = _cmp_img(a, b, acc)
            elif ext in (".npy", ".npz"):
                same = _cmp_arr(a, b, acc)
            elif ext == ".obj":
                same = _cmp_obj(a, b, acc)
            else:
                acc["bytes_diff"] = acc.get("bytes_diff", 0) + 1
                same = False
            acc["identical"] += int(same)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not lose the row
            acc["errors"] += 1
            acc.setdefault("first_error", f"{rel}: {type(exc).__name__}: {exc}"[:200])
    return acc
