"""Copy a motion_v1 partition, rewriting container paths (/workspace/video_to_data/...) to
repo-relative ones so host-side tools (replay_motion_viser.py) can find the object mesh.

The checked-in tissue_box_simple parquet stores absolute container paths; viser_playback
only resolves repo-relative ones. The output keeps the hive layout
(sequence_id=<id>/robot_name=<robot>/) because the reader derives those two fields from it.

Usage: python tools/learn/localize_motion.py <partition_dir> <out_root> [OLD_PREFIX=NEW_PREFIX ...]

Extra OLD=NEW pairs rewrite other prefixes too, e.g. a Modal run's
/e2e/runs/<run>/human_motion_data/...=<local download dir>/.
"""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PREFIXES = [("/workspace/video_to_data/robotic_grounding/", "")]
PREFIXES += [tuple(pair.split("=", 1)) for pair in sys.argv[3:]]
src, out_root = Path(sys.argv[1]), Path(sys.argv[2])


def localize(path: str) -> str:
    for old, new in PREFIXES:
        if path.startswith(old):
            return new + path[len(old):]
    return path


dst = out_root / src.parent.name / src.name
dst.mkdir(parents=True, exist_ok=True)
for f in src.glob("*.parquet"):
    t = pq.read_table(f)
    for col in ("object_mesh_paths", "object_urdf_paths"):
        if col in t.column_names:
            i = t.schema.get_field_index(col)
            vals = [[localize(p) for p in row] for row in t.column(col).to_pylist()]
            t = t.set_column(i, col, pa.array(vals, type=t.schema.field(col).type))
    pq.write_table(t, dst / f.name)
print(dst)
