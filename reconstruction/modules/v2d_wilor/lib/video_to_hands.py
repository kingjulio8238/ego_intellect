# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WiLoR over a video: per-frame JSON list in <output_dir>/<frame:06d>.json.

The video is decoded with ffmpeg (cheaper + more robust than imageio across
container codecs). If ``--bboxes_dir`` is supplied, each frame's external
bboxes are read from ``<bboxes_dir>/<frame:06d>.json``.

Output schema per file: see ``image_to_hands.py``.

Usage:
    python -m v2d.wilor.lib.video_to_hands \\
        --video_path  /data/clip.mp4 \\
        --output_dir  /data/wilor \\
        --weights_dir /data/weights/wilor
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from v2d.wilor.lib._wilor import (
    get_pipeline, run_wilor_detect, run_wilor_on_bboxes,
)
from v2d.wilor.lib.image_to_hands import _load_external_bboxes


def _decode_video_to_frames(video_path: str, frames_dir: str) -> int:
    """Decode all frames of ``video_path`` to ``frames_dir/<06d>.png``. Return frame count."""
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        os.path.join(frames_dir, "%06d.png"),
    ], check=True)
    files = sorted(os.listdir(frames_dir))
    return len(files)


def _decode_selected_frames(video_path: str, frames_dir: str, frame_indices: list[int]) -> None:
    """Decode only ``frame_indices`` (0-based) to ``frames_dir/<idx+1:06d>.png``.

    Same ffmpeg decode as ``_decode_video_to_frames``; the select filter only drops the
    other frames, so the kept frames' pixels are identical.
    """
    os.makedirs(frames_dir, exist_ok=True)
    for idx in frame_indices:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"select=eq(n\\,{int(idx)})", "-vsync", "0", "-frames:v", "1",
            os.path.join(frames_dir, f"{int(idx) + 1:06d}.png"),
        ], check=True)


def video_to_hands(
    video_path: str,
    output_dir: str,
    weights_dir: str,
    bboxes_dir: Optional[str] = None,
    frame_indices: Optional[list[int]] = None,
) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    get_pipeline(weights_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        if frame_indices:
            _decode_selected_frames(video_path, tmpdir, frame_indices)
            one_based = [i + 1 for i in frame_indices]
        else:
            n = _decode_video_to_frames(video_path, tmpdir)
            if n == 0:
                raise RuntimeError(f"ffmpeg decoded zero frames from {video_path}")
            one_based = list(range(1, n + 1))

        # ffmpeg writes 1-indexed; remap to 0-indexed JSON names to match
        # the rest of the pipeline's frame_idx convention.
        for one_idx in tqdm(one_based, desc="wilor", ncols=80, unit="frame"):
            frame_idx = one_idx - 1
            out_path = os.path.join(output_dir, f"{frame_idx:06d}.json")
            if os.path.exists(out_path):
                continue
            frame_path = os.path.join(tmpdir, f"{one_idx:06d}.png")
            image = np.asarray(Image.open(frame_path).convert("RGB"))

            if bboxes_dir is None:
                records = run_wilor_detect(image, weights_dir=weights_dir)
            else:
                bb_path = os.path.join(bboxes_dir, f"{frame_idx:06d}.json")
                if not os.path.exists(bb_path):
                    continue
                bboxes, is_right = _load_external_bboxes(bb_path)
                records = (run_wilor_on_bboxes(image, bboxes, is_right,
                                               weights_dir=weights_dir)
                           if bboxes else [])

            records.sort(key=lambda d: d["score"], reverse=True)
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path",  required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--bboxes_dir", default=None,
                        help="Optional dir of <frame:06d>.json bboxes; skip WiLoR's detector when set.")
    parser.add_argument("--frame_indices", default=None,
                        help="Comma-separated 0-based frames to process (default: all).")
    args = parser.parse_args()
    video_to_hands(
        video_path  = args.video_path,
        output_dir  = args.output_dir,
        weights_dir = args.weights_dir,
        bboxes_dir  = args.bboxes_dir,
        frame_indices = ([int(x) for x in args.frame_indices.split(",")]
                         if args.frame_indices else None),
    )


if __name__ == "__main__":
    main()
