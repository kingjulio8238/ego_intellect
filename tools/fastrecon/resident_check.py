"""Isolated check of resident execution on Modal (no e2e run).

    MODAL_PROFILE=<p> modal run -m tools.fastrecon.resident_check --journal <run journal.jsonl>

For each deterministic critical-path call (WiLoR reference frame, FoundationPose scale and
tracking), re-issues the call's recorded spec through the production path: spawn a
resident container from the recorded hint (worker.serve_resident), hand the spec over on
the resident queue, and wait for the result. Passes when every call ran on its preloaded
container with the preload hook (model built before the call) and exited 0. The calls
rewrite their own outputs in the recorded run's directory; they are deterministic, so
the content is unchanged (verify with the stage diff afterwards).
"""

from __future__ import annotations

import json
import time
import uuid

import modal

from tools.modal_v2d import config as C
from tools.modal_v2d.app import app, img_v2d_foundation_pose, img_v2d_wilor

CALLS = {
    "v2d.wilor.lib.video_to_hands": img_v2d_wilor,
    "v2d.foundation_pose.lib.run_estimate_mesh_scale": img_v2d_foundation_pose,
    "v2d.foundation_pose.lib.run_video_to_poses": img_v2d_foundation_pose,
}


@app.local_entrypoint()
def resident_check(journal: str) -> None:
    specs = {}
    for line in open(journal):
        row = json.loads(line)
        if row.get("kind") == "call_spec":
            spec = row["spec"]
            if len(spec["argv"]) > 2 and spec["argv"][2] in CALLS:
                specs.setdefault(spec["argv"][2], spec)
    hints = modal.Dict.from_name(C.RESIDENT_HINTS_DICT)
    queue = modal.Queue.from_name(C.RESIDENT_QUEUE, create_if_missing=True)
    failures = []
    for module, function in CALLS.items():
        spec = {**specs[module], "call_id": "rcheck-" + uuid.uuid4().hex[:8]}
        hint = (hints.get(spec["image"]) or {}).get(module)
        if not hint or not hint.get("mounts"):
            failures.append(f"{module}: no hint with mounts")
            continue
        key = "r" + uuid.uuid4().hex[:16]
        t0 = time.time()
        call = function.spawn({"image": spec["image"], "resident": {**hint, "key": key}, "call_id": "rcheck-res"})
        time.sleep(45)  # let it preload, as in a run (the call arrives later)
        queue.put(spec, partition=key)
        result = call.get()
        timing = result.get("timing") or {}
        ok = (result.get("returncode") == 0 and timing.get("resident") is True)
        print(f"{module}: rc={result.get('returncode')} resident={timing.get('resident')} "
              f"preload_s={timing.get('preload_s')} exec_s={timing.get('exec_s')} wall={time.time() - t0:.0f}s "
              f"{'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            failures.append(module)
    print("RESIDENT_CHECK " + ("PASS" if not failures else "FAIL " + ", ".join(failures)), flush=True)
