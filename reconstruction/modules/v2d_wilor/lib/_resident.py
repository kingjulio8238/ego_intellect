"""Resident preload for v2d_wilor (see v2d_common/resident.py): builds the WiLoR
pipeline (detector + reconstructor) for the call's --weights_dir; _wilor.get_pipeline takes it."""

from __future__ import annotations


def preload(module: str, argv: list[str]) -> None:
    if module != "v2d.wilor.lib.video_to_hands" or "--weights_dir" not in argv:
        return
    weights_dir = argv[argv.index("--weights_dir") + 1]
    from v2d.common import resident
    from v2d.wilor.lib import _wilor

    pipe = _wilor.get_pipeline(weights_dir)
    _wilor._PIPE = None  # handed over through the resident cache, so a key mismatch rebuilds
    resident.put(_wilor.pipeline_key(weights_dir), pipe)
