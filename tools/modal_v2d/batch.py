"""Batch mode: many clips through the per-clip pipeline concurrently, in one Modal app.

    MODAL_PROFILE=<profile> tools/modal_v2d/run_e2e_modal.sh --batch <manifest.json> [--max-parallel N]

One app for the whole batch means one container pool per module image: a container warmed
or used by one clip serves the next clip's call of the same image, instead of every clip
paying its own cold starts. Each clip runs in its own control container (init -> setup ->
reconstruct in one container start), under its own run-root lock and index partition, so
clips never touch each other's state.

Manifest (JSON):

    {
      "clips": [
        {"name": "tissue_box_b1", "video": "/workspace/video_to_data/reconstruction/assets/tissue_box.mp4",
         "task_profile": "/workspace/video_to_data/robotic_grounding/groot_finetune/task_profiles/tissue_box_lift_hold.json",
         "sequence_id": "tissue_box"}
      ],
      "embodiment_contract": "vega_sharpa_joint",
      "setup": true,
      "reconstruct": true,
      "reconstruct_args": ["--hand-tracking", "hamer", "--run-gsplat-refinement", "--dev", "--fast"],
      "options": {"gpu_slots": 8, "exit_after_call": true, "warm": false, "residents": false, "hedge_after_s": 60}
    }

`mode`: `shared` (default) runs every clip in one control container, so one GPU ledger
(tools/modal_v2d/ledger.py) sees all containers the batch holds and reuses idle ones across
clips; `per_clip` gives each clip its own control container, coordinated by GPU slots.

`options` (all optional): `gpu_slots` caps GPU module calls in flight across the batch (keep it
below the workspace's GPU concurrency: 10 on Modal Starter); `warm`/`residents` false turn off
the speculative containers that `--fast` starts for single-clip latency (each holds a GPU slot);
`exit_after_call` releases each GPU container right after its call (no idle scaledown
containers counting against the cap; costs a ~6-17 s cold start per call);
`hedge_after_s` (0 disables) re-sends a call that has no container yet to a second pool.

`render_every` (default 20; 0: none): which clips get the optional environment video
(`render_queue`). The batch records the queue; it is drained after delivery, never inline.

Run roots are /e2e/runs/<name>. Paths are as seen inside the control container.
"""

from __future__ import annotations

import hashlib
import json
import re

RUNS = "/e2e/runs"
CONFIG = "e2e_config.json"
DEFAULT_RECONSTRUCT_ARGS = ["--hand-tracking", "hamer", "--run-gsplat-refinement", "--dev", "--fast"]
SETUP_ARGS = [
    "--skip-reconstruction",  # weights and images are per workspace, prepared once (`setup`)
    "--mano-dir", "/weights/mano",
    "--isaac-groot-dir", "/opt/Isaac-GR00T",
    "--accept-nvidia-model-eula",
]
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def load_manifest(text: str) -> dict:
    manifest = json.loads(text)
    clips = manifest.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("manifest needs a non-empty 'clips' list")
    names = [clip.get("name", "") for clip in clips]
    for clip in clips:
        missing = [key for key in ("name", "video", "task_profile", "sequence_id") if not clip.get(key)]
        if missing:
            raise ValueError(f"clip {clip.get('name', '?')}: missing {', '.join(missing)}")
        if not _NAME.match(clip["name"]):
            raise ValueError(f"clip name {clip['name']!r}: use letters, digits, '_', '-', '.'")
    if len(set(names)) != len(names):
        raise ValueError("clip names must be unique (each is a run root)")
    return manifest


def clip_steps(clip: dict, manifest: dict) -> list[list[str]]:
    """run_e2e.sh argvs for one clip, in order; all run in the control role."""
    root = f"{RUNS}/{clip['name']}"
    config = f"{root}/{CONFIG}"
    steps = [[
        "init", "--run-root", root,
        "--sequence-id", clip["sequence_id"],
        "--embodiment-contract", clip.get("embodiment_contract") or manifest.get("embodiment_contract", "vega_sharpa_joint"),
        "--task-profile", clip["task_profile"],
    ]]
    if manifest.get("setup", True):
        steps.append(["--config", config, "setup", *SETUP_ARGS])
    if manifest.get("reconstruct", True):  # false: plumbing smoke (CPU only)
        reconstruct_args = clip.get("reconstruct_args") or manifest.get("reconstruct_args") or DEFAULT_RECONSTRUCT_ARGS
        steps.append(["--config", config, "reconstruct", "--video", clip["video"], *reconstruct_args])
    return steps


RENDER_EVERY = 20  # 1 clip in N gets the optional `render` visual check, plus every flagged clip


def render_queue(names: list[str], every: int = RENDER_EVERY, flagged: tuple[str, ...] | list[str] = ()) -> list[str]:
    """Clips that get the optional environment video (`run_e2e.sh render`): every flagged clip
    (failed a numeric check) plus a stable 1-in-`every` sample. The sample hashes the clip
    name, so it does not depend on manifest order, batch size or reruns. every <= 0: flagged only."""
    flagged_set = set(flagged)
    chosen = []
    for name in names:
        sampled = every > 0 and int(hashlib.sha1(name.encode()).hexdigest(), 16) % every == 0
        if name in flagged_set or sampled:
            chosen.append(name)
    return chosen


def render_commands(names: list[str]) -> list[list[str]]:
    """run_e2e.sh argvs that drain a render queue: one GPU, after the clip is delivered
    (needs `retarget` output); never on a clip's critical path."""
    return [["--config", f"{RUNS}/{name}/{CONFIG}", "render"] for name in names]


def summarize(results: list[dict]) -> str:
    """One line per clip plus the batch makespan (results from app.batch_driver)."""
    if not results:
        return "no clips"
    t0 = min(r["start"] for r in results)
    t1 = max(r["end"] for r in results)
    lines = [f"{'clip':24s} {'result':10s} {'start+':>8s} {'wall':>8s}"]
    for r in sorted(results, key=lambda r: r["start"]):
        lines.append(
            f"{r['name']:24s} {('ok' if r['code'] == 0 else 'FAIL ' + str(r['code']))[:10]:10s} "
            f"{_mmss(r['start'] - t0):>8s} {_mmss(r['end'] - r['start']):>8s}"
        )
    ok = sum(r["code"] == 0 for r in results)
    lines.append(f"makespan {_mmss(t1 - t0)} | {ok}/{len(results)} ok | "
                 f"{ok / max(t1 - t0, 1e-9) * 3600:.1f} clips/h")
    return "\n".join(lines)


def _mmss(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"
