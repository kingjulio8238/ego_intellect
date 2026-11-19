"""Resident preload for v2d_sam3d (see v2d_common/resident.py, tools/modal_v2d/zygote.py).

Builds the SAM3D pipeline (weights on the GPU) with the same code and arguments the
call will use; image_to_mesh._get_pipeline then takes it instead of building its own.
"""

from __future__ import annotations


def _arg(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def preload(module: str, argv: list[str]) -> None:
    if module != "v2d.sam3d.lib.image_to_mesh":
        return
    weights_dir = _arg(argv, "--weights_dir")
    if not weights_dir:
        return
    from v2d.common import resident
    from v2d.sam3d.lib import image_to_mesh

    resident.put(image_to_mesh._pipeline_key(weights_dir), image_to_mesh._get_pipeline(weights_dir))
