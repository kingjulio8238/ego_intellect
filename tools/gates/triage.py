"""Triage gate (stage 3): reject a clip on CPU before any GPU time is spent on it.

Checks the video itself: format envelope (docs/pipeline/INTAKE.md), sharpness, exposure,
frozen frames, scene cuts, and whether close-up hands are in view (MediaPipe hand landmarker on
sampled frames). It does not judge the object; reconstruction QA does.

    python3 -m tools.gates.triage --video clip.mp4 [--out triage.json] [--hand-model PATH]

Exit 0: accept; 3: reject (report written); 2: could not evaluate. Limits: tools/gates/limits.json.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import REJECTED_EXIT, evaluate, write_report

SECTION = "triage/v1"
DEFAULT_HAND_MODEL = "/opt/mediapipe_tasks/hand_landmarker.task"
SAMPLE_FPS = 2.0  # frames per second analysed in detail
MAX_SAMPLES = 60
DETAIL_WIDTH = 640  # sharpness, exposure, texture and hands are measured at this width
DIFF_WIDTH = 160  # every frame, for frozen frames and cuts
SHARP_LAPVAR = 60.0  # Laplacian variance (gray, 640 px wide) below which a frame is blurred
FROZEN_DIFF = 0.5  # mean absolute gray difference (0-255, 160 px) below which a frame repeats the last
CUT_DISTANCE = 0.55  # Bhattacharyya distance between consecutive colour histograms that marks a cut


def _hand_detector(model: Path):
    try:
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    except ImportError as exc:
        raise RuntimeError("mediapipe is required for the triage hand check") from exc
    if not model.is_file():
        raise FileNotFoundError(f"hand landmarker model not found: {model}")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def _hands(detector: Any, rgb: np.ndarray) -> list[float]:
    """Sizes of the detected hands: bounding-box long side / image short side."""
    import mediapipe as mp

    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    h, w = rgb.shape[:2]
    sizes = []
    for landmarks in result.hand_landmarks:
        xs = np.array([p.x for p in landmarks]) * w
        ys = np.array([p.y for p in landmarks]) * h
        sizes.append(float(max(xs.max() - xs.min(), ys.max() - ys.min()) / min(h, w)))
    return sizes


def analyze_video(video: Path, *, hand_model: Path | None = None) -> dict[str, Any]:
    """All triage/v1 metrics for ``video``."""
    import cv2

    from tools.envpkg.intake import _video_errors, probe_video

    info = probe_video(video)
    format_errors = _video_errors(info)
    detector = _hand_detector(Path(hand_model or os.environ.get("V2D_HAND_LANDMARKER", DEFAULT_HAND_MODEL)))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or info["fps"] or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or info["frames"]
    stride = max(1, round(fps / SAMPLE_FPS), int(np.ceil(total / MAX_SAMPLES)) if total else 1)

    prev_small, prev_hist = None, None
    frozen, cuts, frames = 0, 0, 0
    lapvars, lumas, clipped, texture, hand_counts, hand_sizes = [], [], [], [], [], []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        small = cv2.resize(frame, (DIFF_WIDTH, max(1, round(frame.shape[0] * DIFF_WIDTH / frame.shape[1]))),
                           interpolation=cv2.INTER_AREA)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev_small is not None:
            frozen += int(np.abs(gray_small - prev_small).mean() < FROZEN_DIFF)
            cuts += int(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > CUT_DISTANCE)
        prev_small, prev_hist = gray_small, hist
        if index % stride == 0:
            detail = cv2.resize(frame, (DETAIL_WIDTH, max(1, round(frame.shape[0] * DETAIL_WIDTH / frame.shape[1]))),
                                interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(detail, cv2.COLOR_BGR2GRAY)
            lapvars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            lumas.append(float(gray.mean()))
            clipped.append(float(((gray < 10) | (gray > 245)).mean()))
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=8)
            texture.append(0 if corners is None else len(corners))
            sizes = _hands(detector, cv2.cvtColor(detail, cv2.COLOR_BGR2RGB))
            hand_counts.append(len(sizes))
            if sizes:
                hand_sizes.append(max(sizes))
        index += 1
    capture.release()
    detector.close()
    if frames < 2 or not lapvars:
        raise RuntimeError(f"{video}: decoded {frames} frames")
    lumas_arr, clipped_arr = np.array(lumas), np.array(clipped)
    exposure_ok = (lumas_arr >= 40) & (lumas_arr <= 215) & (clipped_arr <= 0.25)
    return {
        "format_errors": len(format_errors),
        "_format_reasons": format_errors,
        "_video": info,
        "frames_decoded": frames,
        "samples": len(lapvars),
        "hand_frame_fraction": float(np.mean(np.array(hand_counts) > 0)),
        "both_hands_fraction": float(np.mean(np.array(hand_counts) > 1)),
        "hand_size_median": float(np.median(hand_sizes)) if hand_sizes else 0.0,
        "sharp_frame_fraction": float(np.mean(np.array(lapvars) >= SHARP_LAPVAR)),
        "exposure_ok_fraction": float(exposure_ok.mean()),
        "frozen_fraction": frozen / (frames - 1),
        "cut_count": cuts,
        "texture_corners_median": float(np.median(texture)),
    }


def run(video: Path, *, hand_model: Path | None, out: Path | None) -> dict[str, Any]:
    start = time.time()
    metrics = analyze_video(video, hand_model=hand_model)
    checks, reasons = evaluate(metrics, SECTION)
    reasons = [f"format: {r}" for r in metrics["_format_reasons"]] + [r for r in reasons if not r.startswith("format_errors")]
    report = {
        "schema": SECTION,
        "video": str(video),
        "verdict": "reject" if reasons else "accept",
        "reasons": reasons,
        "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
        "video_info": metrics["_video"],
        "checks": checks,
        "seconds": round(time.time() - start, 2),
    }
    if out is not None:
        write_report(out, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--exit-zero", action="store_true",
                        help="exit 0 after writing a rejection (pipeline stages read the verdict from the report)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--hand-model", type=Path, help=f"MediaPipe hand_landmarker.task (default {DEFAULT_HAND_MODEL}, or $V2D_HAND_LANDMARKER)")
    args = parser.parse_args(argv)
    try:
        report = run(args.video, hand_model=args.hand_model, out=args.out)
    except Exception as exc:  # noqa: BLE001 - any failure to evaluate is exit 2, never a silent accept
        print(f"triage: cannot evaluate {args.video}: {exc}", file=sys.stderr)
        return 2
    print(f"triage: {report['verdict'].upper()} ({report['seconds']} s)" + (f" -> {args.out}" if args.out else ""))
    for reason in report["reasons"]:
        print(f"  - {reason}")
    return REJECTED_EXIT if report["reasons"] and not args.exit_zero else 0


if __name__ == "__main__":
    raise SystemExit(main())
