"""Resident preload for v2d_sam2 (see v2d_common/resident.py): builds the video
predictor with the call's --weights_dir; video_to_masks._get_predictor takes it."""

from __future__ import annotations


def preload(module: str, argv: list[str]) -> None:
    if module != "v2d.sam2.lib.video_to_masks" or "--weights_dir" not in argv:
        return
    weights_dir = argv[argv.index("--weights_dir") + 1]
    from v2d.common import resident
    from v2d.sam2.lib import video_to_masks

    resident.put(video_to_masks._predictor_key(weights_dir), video_to_masks._get_predictor(weights_dir))
