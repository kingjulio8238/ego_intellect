"""Reconstruction QA gate (stage 5): pass/fail limits on a reconstruction bundle.

Runs on CPU right after `reconstruct`, before any robot-phase GPU time. Two groups of checks:

- bundle-only (any bundle, including imported ones): tracking coverage, camera jumps,
  object/wrist jitter, object size, hands reaching the object;
- against the clip's own segmentation (when the run root still has `reconstruction/masks`):
  the object mesh, posed by the reconstruction, must cover the object's SAM2 mask
  (hand-occlusion aware), and each wrist must project onto its hand's mask.

    python3 -m tools.gates.recon_qa --run-root <run_root> [--out report.json]
    python3 -m tools.gates.recon_qa --bundle <bundle_dir> [--masks <dir>] --fps 30

Exit 0: pass; 3: rejected (report written); 2: could not evaluate. Limits: tools/gates/limits.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import REJECTED_EXIT, evaluate, write_report

SECTION = "recon_qa/v1"
BUNDLE = Path("reconstruction/result_slam_gravity_aligned")
MASK_SCALE = 0.5  # silhouettes and masks compared at half resolution
MASK_SAMPLES = 80  # frames compared against the masks (evenly spaced)
INSIDE_M = 0.01  # a wrist deeper than this inside the object's convex hull is a pose error


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle (rad) between rotation matrices a[i] and b[i]."""
    cos = (np.einsum("nij,nij->n", a, b) - 1.0) / 2.0
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _consecutive(valid: np.ndarray, width: int) -> np.ndarray:
    """Start indices i where frames i..i+width-1 are all valid."""
    ok = np.ones(len(valid) - width + 1, bool)
    for k in range(width):
        ok &= valid[k: len(valid) - width + 1 + k]
    return np.flatnonzero(ok)


def _accel_p95(positions: np.ndarray, valid: np.ndarray, fps: float) -> float | None:
    starts = _consecutive(valid, 3)
    if len(starts) == 0:
        return None
    second = positions[starts + 2] - 2 * positions[starts + 1] + positions[starts]
    return float(np.percentile(np.linalg.norm(second, axis=1), 95) * fps * fps)


def _surface_points(vertices: np.ndarray, faces: np.ndarray, count: int = 3000) -> np.ndarray:
    """Points spread over the mesh surface (area-weighted, fixed seed), so distances to the
    object do not depend on how finely it is tessellated."""
    corners = vertices[faces]
    area = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    if area.sum() <= 0:
        return vertices
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(faces), size=count, p=area / area.sum())
    a, b = rng.random((2, count, 1))
    flip = (a + b) > 1
    a, b = np.where(flip, 1 - a, a), np.where(flip, 1 - b, b)
    tri = corners[chosen]
    return tri[:, 0] + a * (tri[:, 1] - tri[:, 0]) + b * (tri[:, 2] - tri[:, 0])


def _obj_mesh(bundle: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    for line in (bundle / "mesh.obj").read_text(errors="replace").splitlines():
        if line.startswith("v "):
            vertices.append(line.split()[1:4])
        elif line.startswith("f "):
            faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    return np.asarray(vertices, float), np.asarray(faces, int).reshape(-1, 3)


def _silhouette(vertices: np.ndarray, faces: np.ndarray, pose: np.ndarray, intr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    cam = vertices @ pose[:3, :3].T + pose[:3, 3]
    if np.any(cam[:, 2] <= 1e-3):  # mesh behind or at the camera: no silhouette
        return np.zeros(shape, bool)
    fx, fy, cx, cy = intr
    uv = np.stack([fx * cam[:, 0] / cam[:, 2] + cx, fy * cam[:, 1] / cam[:, 2] + cy], axis=1) * MASK_SCALE
    canvas = np.zeros(shape, np.uint8)
    triangles = np.round(uv[faces] * 4).astype(np.int32)  # 2 fractional bits (shift=2)
    for triangle in triangles:  # one call per triangle: fillPoly on many polygons uses even-odd fill
        cv2.fillConvexPoly(canvas, triangle, 1, lineType=cv2.LINE_8, shift=2)
    return canvas > 0


def _mask(path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA) > 127


def mask_metrics(npz: Any, bundle_manifest: dict[str, Any], masks: Path, vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Object silhouette agreement and wrist-on-hand agreement against the SAM2 masks."""
    import cv2

    sources = bundle_manifest.get("sources", {})
    object_id = str(sources.get("object_track_id", 1))
    hand_ids = {side: Path(str(sources.get(f"{side}_hand_dir") or "")).name for side in ("left", "right")}
    probe = next((masks / object_id).glob("*.png"), None)
    if probe is None:
        raise FileNotFoundError(f"no object masks in {masks / object_id}")
    full = cv2.imread(str(probe), cv2.IMREAD_GRAYSCALE).shape
    shape = (round(full[0] * MASK_SCALE), round(full[1] * MASK_SCALE))
    n = len(npz["object_is_valid"])
    frames = np.unique(np.linspace(0, n - 1, min(MASK_SAMPLES, n)).round().astype(int))
    intr = np.asarray(npz["camera_intrinsics"], float)
    ious, present, wrist_hits, wrist_total = [], 0, 0, 0
    for i in frames:
        obj = _mask(masks / object_id / f"{i:06d}.png", shape)
        hands = {side: _mask(masks / hid / f"{i:06d}.png", shape) if hid else None for side, hid in hand_ids.items()}
        occluder = np.zeros(shape, bool)
        for hand in hands.values():
            if hand is not None:
                occluder |= hand
        if obj is not None and obj.any():
            present += 1
            if npz["object_is_valid"][i]:
                sil = _silhouette(vertices, faces, np.asarray(npz["object_to_camera_transform"][i], float), intr, shape)
                a, b = sil & ~occluder, obj & ~occluder
                union = (a | b).sum()
                ious.append(float((a & b).sum() / union) if union else 0.0)
        for side, hand in hands.items():
            if hand is None or not hand.any() or not npz[f"hand_{side}_is_valid"][i]:
                continue
            p = np.asarray(npz[f"hand_{side}_wrist_to_camera_transform"][i], float)[:3, 3]
            if p[2] <= 1e-3:
                wrist_total += 1
                continue
            u = (intr[0] * p[0] / p[2] + intr[2]) * MASK_SCALE
            v = (intr[1] * p[1] / p[2] + intr[3]) * MASK_SCALE
            ys, xs = np.nonzero(hand)
            pad_x, pad_y = 0.3 * (xs.max() - xs.min() + 1), 0.3 * (ys.max() - ys.min() + 1)
            wrist_total += 1
            wrist_hits += int(xs.min() - pad_x <= u <= xs.max() + pad_x and ys.min() - pad_y <= v <= ys.max() + pad_y)
    return {
        "object_mask_fraction": present / len(frames),
        "object_iou_median": float(np.median(ious)) if ious else 0.0,
        "object_iou_p10": float(np.percentile(ious, 10)) if ious else 0.0,
        "wrist_on_hand_mask_fraction": wrist_hits / wrist_total if wrist_total else 0.0,
    }


def analyze_bundle(bundle: Path, *, fps: float, masks: Path | None = None) -> dict[str, Any]:
    """All recon_qa/v1 metrics for ``bundle`` (mask metrics are None without ``masks``)."""
    npz = np.load(bundle / "result.npz")
    manifest = json.loads((bundle / "manifest.json").read_text())
    vertices, faces = _obj_mesh(bundle)
    cam = np.asarray(npz["camera_to_world_transform"], float)
    obj = np.asarray(npz["object_to_world_transform"], float)
    cam_valid = np.asarray(npz["camera_is_valid"], bool)
    obj_valid = np.asarray(npz["object_is_valid"], bool)
    hand_valid = {s: np.asarray(npz[f"hand_{s}_is_valid"], bool) for s in ("left", "right")}
    wrists = {s: np.asarray(npz[f"hand_{s}_wrist_to_world_transform"], float)[:, :3, 3] for s in ("left", "right")}

    pairs = _consecutive(cam_valid, 2)
    cam_speed = np.linalg.norm(cam[pairs + 1, :3, 3] - cam[pairs, :3, 3], axis=1) * fps if len(pairs) else np.zeros(1)
    cam_turn = _rotation_angle(cam[pairs + 1, :3, :3], cam[pairs, :3, :3]) * fps if len(pairs) else np.zeros(1)

    wrist_accel = [_accel_p95(wrists[s], hand_valid[s], fps) for s in ("left", "right")]
    wrist_accel = [a for a in wrist_accel if a is not None]

    # Hands at the object: per frame, the closer wrist's distance to the object surface
    # (subsampled mesh), and whether any wrist sits inside the object's convex hull.
    from scipy.spatial import ConvexHull

    planes = ConvexHull(vertices).equations  # n.x + d > 0 outside
    local = _surface_points(vertices, faces)
    inverse = np.linalg.inv(obj)
    gaps, inside, checked = [], 0, 0
    for i in np.flatnonzero(obj_valid):
        hands = [s for s in ("left", "right") if hand_valid[s][i]]
        if not hands:
            continue
        surface = local @ obj[i, :3, :3].T + obj[i, :3, 3]
        gaps.append(min(np.linalg.norm(surface - wrists[s][i], axis=1).min() for s in hands))
        for s in hands:
            point = inverse[i, :3, :3] @ wrists[s][i] + inverse[i, :3, 3]
            checked += 1
            inside += int((planes[:, :3] @ point + planes[:, 3]).max() < -INSIDE_M)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    metrics: dict[str, Any] = {
        "frames": int(len(cam_valid)),
        "fps": fps,
        "camera_valid_fraction": float(cam_valid.mean()),
        "object_valid_fraction": float(obj_valid.mean()),
        "hand_valid_fraction": float(max(hand_valid["left"].mean(), hand_valid["right"].mean())),
        "camera_speed_max_mps": float(cam_speed.max()),
        "camera_turn_max_radps": float(cam_turn.max()),
        "object_accel_p95_mps2": _accel_p95(obj[:, :3, 3], obj_valid, fps),
        "wrist_accel_p95_mps2": float(max(wrist_accel)) if wrist_accel else None,
        "object_extent_max_m": float(extent.max()),
        "object_extent_min_m": float(extent.min()),
        "hand_object_gap_p20_m": float(np.percentile(gaps, 20)) if gaps else None,
        "wrist_inside_object_fraction": inside / checked if checked else None,
        "object_mask_fraction": None,
        "object_iou_median": None,
        "object_iou_p10": None,
        "wrist_on_hand_mask_fraction": None,
    }
    if masks is not None:
        metrics.update(mask_metrics(npz, manifest, masks, vertices, faces))
    return metrics


def _video_fps(run_root: Path) -> float | None:
    """Frame rate of the clip's source video (from the run configuration)."""
    import cv2

    try:
        video = json.loads((run_root / "e2e_config.json").read_text())["inputs"]["video"]
    except (OSError, KeyError, ValueError):
        return None
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) if capture.isOpened() else 0.0
    capture.release()
    return float(fps) if fps and fps > 0 else None


def run(bundle: Path, *, fps: float, masks: Path | None, out: Path | None) -> dict[str, Any]:
    start = time.time()
    metrics = analyze_bundle(bundle, fps=fps, masks=masks)
    checks, reasons = evaluate(metrics, SECTION)
    skipped = [name for name, check in checks.items() if check["ok"] is None]
    report = {
        "schema": SECTION,
        "bundle": str(bundle),
        "masks": str(masks) if masks else None,
        "verdict": "reject" if reasons else "pass",
        "reasons": reasons,
        "skipped": skipped,
        "metrics": metrics,
        "checks": checks,
        "seconds": round(time.time() - start, 2),
    }
    if out is not None:
        write_report(out, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, help="e2e run root (bundle, masks and fps found from it)")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--masks", type=Path, help="SAM2 masks dir (reconstruction/masks)")
    parser.add_argument("--fps", type=float, help="source video frame rate")
    parser.add_argument("--exit-zero", action="store_true",
                        help="exit 0 after writing a rejection (pipeline stages read the verdict from the report)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.run_root is not None:
            bundle = args.bundle or args.run_root / BUNDLE
            masks = args.masks or (args.run_root / "reconstruction/masks")
            masks = masks if masks.is_dir() else None
            fps = args.fps or _video_fps(args.run_root)
            out = args.out or args.run_root / "qa/reconstruction.json"
        else:
            if args.bundle is None:
                parser.error("--run-root or --bundle is required")
            bundle, masks, fps, out = args.bundle, args.masks, args.fps, args.out
        if not fps:
            raise ValueError("frame rate unknown: pass --fps (the source video's frame rate)")
        report = run(bundle, fps=fps, masks=masks, out=out)
    except (OSError, ValueError, KeyError) as exc:
        print(f"recon_qa: cannot evaluate: {exc}", file=sys.stderr)
        return 2
    print(f"reconstruction QA: {report['verdict'].upper()}" + (f" -> {out}" if out else ""))
    for reason in report["reasons"]:
        print(f"  - {reason}")
    if report["skipped"]:
        print(f"  (not measurable here: {', '.join(report['skipped'])})")
    return REJECTED_EXIT if report["reasons"] and not args.exit_zero else 0


if __name__ == "__main__":
    raise SystemExit(main())
