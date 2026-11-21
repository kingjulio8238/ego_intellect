"""Validate a vendor delivery (intake/v1) before anything is ingested.

Schema, then per clip: the files exist and match their hashes, the video is decodable and
inside the envelope the pipeline is run on (ffprobe), the optional intrinsics describe that
video, and the optional IMU stream covers it. Contract: docs/pipeline/INTAKE.md.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from .common import INTAKE_SCHEMA, ContractError, schema_errors, sha256_file

# Envelope (docs/pipeline/INTAKE.md). Tested end to end: 1280x720 and 1280x800 at 30 fps,
# 13-26 s. The limits below are what the pipeline accepts; triage (stage 3) may be stricter.
CONTAINERS = {"mov,mp4,m4a,3gp,3g2,mj2"}
CODECS = {"h264", "hevc"}
MIN_SHORT_SIDE = 480
MAX_LONG_SIDE = 3840
FPS_RANGE = (24.0, 60.0)
DURATION_RANGE_S = (2.0, 60.0)
IMU_HEADER = ["t_s", "ax", "ay", "az", "gx", "gy", "gz"]
IMU_MIN_RATE_HZ = 100.0


@dataclass
class IntakeReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    clips: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def probe_video(path: Path) -> dict[str, Any]:
    """Container, codec, size, rate and duration of ``path`` via ffprobe."""
    if shutil.which("ffprobe") is None:
        raise ContractError("ffprobe is required to check videos (or pass --no-probe to skip, not for acceptance)")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_packets:stream_side_data=rotation:format=format_name,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise ContractError(f"{path}: ffprobe failed: {out.stderr.strip()[:300]}")
    data = json.loads(out.stdout)
    if not data.get("streams"):
        raise ContractError(f"{path}: no video stream")
    stream = data["streams"][0]
    rotation = 0
    for side in stream.get("side_data_list", []) or []:
        rotation = int(side.get("rotation", 0) or 0)
    return {
        "format": data.get("format", {}).get("format_name"),
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(Fraction(stream.get("r_frame_rate", "0/1"))) if stream.get("r_frame_rate") != "0/0" else 0.0,
        "avg_fps": float(Fraction(stream.get("avg_frame_rate", "0/1"))) if stream.get("avg_frame_rate") != "0/0" else 0.0,
        "frames": int(stream.get("nb_read_packets", 0) or 0),
        "duration_s": float(data.get("format", {}).get("duration", 0) or 0),
        "rotation": rotation,
    }


def _video_errors(info: dict[str, Any]) -> list[str]:
    errors = []
    if info["format"] not in CONTAINERS:
        errors.append(f"container {info['format']!r}; expected MP4/MOV")
    if info["codec"] not in CODECS:
        errors.append(f"codec {info['codec']!r}; expected one of {sorted(CODECS)}")
    width, height = info["width"], info["height"]
    if info["rotation"] in (90, -90, 270, -270):
        width, height = height, width
    if min(width, height) < MIN_SHORT_SIDE or max(width, height) > MAX_LONG_SIDE:
        errors.append(f"resolution {width}x{height}; short side >= {MIN_SHORT_SIDE}, long side <= {MAX_LONG_SIDE}")
    if not FPS_RANGE[0] <= info["fps"] <= FPS_RANGE[1]:
        errors.append(f"frame rate {info['fps']:.2f}; expected {FPS_RANGE[0]:g}-{FPS_RANGE[1]:g} fps")
    if info["avg_fps"] and abs(info["avg_fps"] - info["fps"]) > 0.01 * info["fps"]:
        errors.append(f"variable frame rate (nominal {info['fps']:.2f}, average {info['avg_fps']:.2f}); constant rate required")
    if not DURATION_RANGE_S[0] <= info["duration_s"] <= DURATION_RANGE_S[1]:
        errors.append(f"duration {info['duration_s']:.1f} s; expected {DURATION_RANGE_S[0]:g}-{DURATION_RANGE_S[1]:g} s")
    return errors


def _imu_errors(path: Path, offset_s: float, duration_s: float | None) -> list[str]:
    with open(path, newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != IMU_HEADER:
            return [f"IMU header {header}; expected {IMU_HEADER}"]
        times = []
        for line_no, row in enumerate(reader, start=2):
            try:
                values = [float(v) for v in row]
            except ValueError:
                return [f"IMU line {line_no}: non-numeric value"]
            if len(values) != len(IMU_HEADER):
                return [f"IMU line {line_no}: {len(values)} columns"]
            times.append(values[0])
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        return ["IMU timestamps must be strictly increasing (at least 2 samples)"]
    errors = []
    rate = (len(times) - 1) / (times[-1] - times[0])
    if rate < IMU_MIN_RATE_HZ:
        errors.append(f"IMU rate {rate:.0f} Hz; expected >= {IMU_MIN_RATE_HZ:g} Hz")
    if duration_s is not None and (times[0] > offset_s or times[-1] < offset_s + duration_s):
        errors.append(f"IMU covers {times[0]:.2f}-{times[-1]:.2f} s; video spans {offset_s:.2f}-{offset_s + duration_s:.2f} s")
    return errors


def _file_errors(root: Path, entry: dict[str, Any], label: str) -> list[str]:
    path = root / entry["path"]
    if not path.is_file() or path.is_symlink():
        return [f"{label}: {entry['path']} missing (or a symlink)"]
    if not path.resolve().is_relative_to(root.resolve()):
        return [f"{label}: {entry['path']} resolves outside the delivery"]
    if path.stat().st_size != entry["bytes"]:
        return [f"{label}: {entry['path']} is {path.stat().st_size} bytes, manifest says {entry['bytes']}"]
    if sha256_file(path) != entry["sha256"]:
        return [f"{label}: {entry['path']} sha256 mismatch"]
    return []


def validate_intake(manifest_path: Path, *, probe: bool = True) -> IntakeReport:
    """Validate the delivery whose intake.json is ``manifest_path`` (files relative to its folder)."""
    report = IntakeReport()
    root = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        report.errors.append(f"{manifest_path}: unreadable ({exc})")
        return report
    report.errors += schema_errors(INTAKE_SCHEMA, manifest)
    if report.errors:
        return report
    ids = [clip["clip_id"] for clip in manifest["clips"]]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        report.errors.append(f"clip_id {dup} appears more than once")
    if not probe:
        report.warnings.append("videos not probed (--no-probe): format, rate and duration unchecked")
    for clip in manifest["clips"]:
        cid = clip["clip_id"]
        errors = _file_errors(root, clip["video"], f"{cid} video")
        info = None
        if not errors and probe:
            try:
                info = probe_video(root / clip["video"]["path"])
                errors += [f"{cid} video: {e}" for e in _video_errors(info)]
            except ContractError as exc:
                errors.append(f"{cid} video: {exc}")
        intrinsics = clip["capture"].get("intrinsics")
        if intrinsics:
            if not (0 < intrinsics["cx"] < intrinsics["width"] and 0 < intrinsics["cy"] < intrinsics["height"]):
                errors.append(f"{cid} intrinsics: principal point outside the image")
            if info and (intrinsics["width"], intrinsics["height"]) != (info["width"], info["height"]):
                errors.append(f"{cid} intrinsics: {intrinsics['width']}x{intrinsics['height']} vs video {info['width']}x{info['height']}")
            if intrinsics["model"] == "pinhole" and any(intrinsics["distortion"]):
                errors.append(f"{cid} intrinsics: pinhole model with non-zero distortion")
        imu = clip["capture"].get("imu")
        if imu:
            imu_file_errors = _file_errors(root, imu["file"], f"{cid} imu")
            errors += imu_file_errors or [f"{cid} {e}" for e in _imu_errors(
                root / imu["file"]["path"], imu["time_offset_s"], info["duration_s"] if info else None)]
        report.errors += errors
        report.clips[cid] = {"ok": not errors, "video": info}
    return report
