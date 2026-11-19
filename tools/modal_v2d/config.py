"""Static layout shared by the local launcher, the Modal app, the docker shim, and workers.

Stdlib only: this module is imported inside every container image, including the
reconstruction module images whose Python environments we do not control.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = os.environ.get("V2D_MODAL_APP", "v2d-e2e")

# --- Volumes -------------------------------------------------------------------------
# State volume: canonical copy of every container's working tree (run roots, mirrors of
# bind-mounted host directories, per-call logs). Weights volume: model weights, MANO,
# Hugging Face cache. Both are mounted at the same paths in every container.
STATE_VOLUME = os.environ.get("V2D_MODAL_STATE_VOLUME", "v2d-e2e")
WEIGHTS_VOLUME = os.environ.get("V2D_MODAL_WEIGHTS_VOLUME", "v2d-weights")
LOCK_DICT = os.environ.get("V2D_MODAL_LOCK_DICT", "v2d-e2e-locks")
PROGRESS_DICT = os.environ.get("V2D_MODAL_PROGRESS_DICT", "v2d-e2e-progress")  # live run status
STATE_MOUNT = "/vol"
WEIGHTS_MOUNT = "/weights"
MIRROR_ROOT = "/vol/fs"  # local path P is mirrored at MIRROR_ROOT + P
INDEX_PATH = "/vol/.v2d_sync_index.json"
INDEX_PARTITION_DIR = "/vol/.v2d_sync_index"  # one index file per synced root
CALL_LOG_ROOT = "/vol/calls"
PROGRESS_ROOT = "/vol/progress"  # one JSONL journal per invocation

# --- Paths inside the control and robotic-grounding containers ------------------------
REPO_ROOT = "/workspace/video_to_data"  # same path the robotic-grounding image uses
REPO_UPLOAD = "/opt/v2d_repo_src"  # read-only upload of the local checkout
LOCAL_STATE_ROOT = "/e2e"  # run roots live at /e2e/runs/<name>; checked in/out per call
LOCAL_POINTER_DIR = "/e2e/.e2e"  # REPO_ROOT/.e2e is symlinked here (active-run pointer)
RECON_WEIGHTS = "/weights/recon"  # REPO_ROOT/reconstruction/data/weights -> here
MANO_DIR = "/weights/mano"  # must contain models/MANO_LEFT.pkl and models/MANO_RIGHT.pkl
HF_HOME = "/weights/hf"
# JIT-compiled torch extensions (gsplat, ...) persist here per image; workers copy them to
# local disk for the call (a volume is no place for a build tree and its lock files).
EXT_CACHE_ROOT = "/weights/cache/torch_extensions"
LOCAL_EXT_DIR = "/tmp/v2d_torch_extensions"
# torch.compile (inductor) artifacts, same scheme: gsplat --fast compiles ~9 s per call otherwise.
INDUCTOR_CACHE_ROOT = "/weights/cache/torchinductor"
LOCAL_INDUCTOR_DIR = "/tmp/v2d_torchinductor"
GROOT_DIR = "/opt/Isaac-GR00T"
ISAAC_BIN = "/opt/isaac/bin"  # Isaac Lab `python` wrapper (moved off /usr/local/bin)
SHIM_BIN = "/opt/v2d/bin"  # fake `docker` lives here
SHIM_STATE = "/tmp/v2d_docker"
RPC_SOCKET = "/tmp/v2d_modal_rpc.sock"

# Written at image build time so commands see the image's own environment, not the
# Modal runtime's (which prepends its client packages to PYTHONPATH).
IMAGE_PYTHONPATH_FILE = "/etc/v2d_image_pythonpath"
IMAGE_WORKDIR_FILE = "/etc/v2d_image_workdir"
# Variables the role process sets for itself that must not leak into shimmed containers.
ROLE_ONLY_ENV_PREFIXES = ("MODAL_", "V2D_")
ROLE_ONLY_ENV_KEYS = ("PYTHON",)

# --- Images --------------------------------------------------------------------------
# Docker image name -> module directory under reconstruction/modules. These are exactly
# the images `run_e2e.sh setup` builds for the HaMeR route plus the retargeting loader.
MODULE_IMAGES: dict[str, str] = {
    "v2d_anycalib": "v2d_anycalib",
    "v2d_moge": "v2d_moge",
    "v2d_grounding_dino": "v2d_grounding_dino",
    "v2d_sam2": "v2d_sam2",
    "v2d_sam3d": "v2d_sam3d",
    "v2d_foundation_pose": "v2d_foundation_pose",
    "v2d_hamer": "v2d_hamer",
    "v2d_wilor": "v2d_wilor",
    "v2d_geocalib": "v2d_geocalib",
    "v2d_droid_slam": "v2d_droid_slam",
    "v2d_gsplat_refinement": "v2d_gsplat_refinement",
    "v2d_task_library_loader": "v2d_task_library_loader",
}
RG_IMAGE = "robotic-grounding"
ALL_IMAGES = (*MODULE_IMAGES, RG_IMAGE)

# Images that hold a full repo copy at REPO_ROOT (bind mounts of repo paths map onto it).
IMAGES_WITH_REPO = {RG_IMAGE}
# `--dev` mounts reconstruction/modules over /workspace. Modal builds every module image
# from the current checkout at launch, so the image's /workspace already is that code (and
# keeps its compiled artifacts, which docker preserved with anonymous volumes).
DEV_MOUNT = ("reconstruction/modules", "/workspace")
# Extra PATH entries prepended for commands run in an image.
IMAGE_PATH_PREFIX = {RG_IMAGE: ISAAC_BIN}

# --- GPUs, timeouts ------------------------------------------------------------------
# Isaac Sim renders with RTX: use an RT-core GPU (L40S is the repo's verified 48 GB part).
# FoundationPose TensorRT engines are arch-specific, so keep module GPUs on one type too.
RG_GPU = os.environ.get("V2D_MODAL_RG_GPU", "L40S")
MODULE_GPU = os.environ.get("V2D_MODAL_MODULE_GPU", "L40S")
# A comma-separated value is a Modal GPU fallback list, e.g. "L40S,H100,A100-80GB": Modal takes
# the first type with capacity (measured: falls back within ~3 s when the preferred type has
# none, 2026-11-19). Every listed type must pass the numerics gate (docs/pipeline/OPTIMIZATION.md);
# A100-40GB is excluded: SAM3D peaks at 35+ GiB.


# Per-image override, JSON {"v2d_hamer": "L40S,A10G", ...} (memory and the numerics gate decide
# which types an image may use; peak GPU memory: SAM3D 35 GiB, gsplat 20, FoundationPose 19,
# MoGe 13, the rest <= 6).
GPU_PER_IMAGE: dict[str, str] = {
    str(k): str(v) for k, v in __import__("json").loads(os.environ.get("V2D_MODAL_GPU_PER_IMAGE") or "{}").items()
}


def gpu_spec(value: str):
    types = [part.strip() for part in value.split(",") if part.strip()]
    return types if len(types) > 1 else types[0]


def gpu_label(value: str) -> str:
    return value.split(",")[0].strip()  # the preferred type (cost estimates)
# Idle seconds before a module container is released. 120 (was 60) so a container warmed
# by reconstruct --fast (`docker pull`) is still up when its call arrives (~$0.03/idle min, L40S).
MODULE_SCALEDOWN_S = int(os.environ.get("V2D_MODAL_SCALEDOWN_S", "120"))
# Per-image override, JSON {"v2d_sam3d": 300, ...} (batch manifests set it via options.scaledown).
SCALEDOWN_PER_IMAGE: dict[str, int] = {
    str(k): int(v) for k, v in __import__("json").loads(os.environ.get("V2D_MODAL_SCALEDOWN_PER_IMAGE") or "{}").items()
}
# Resident calls (reconstruct --fast): a container preloads its upcoming call's imports and
# model, then waits up to RESIDENT_WAIT_S for the call's spec on this queue.
# Hedged dispatch: a GPU call whose container has not started within HEDGE_AFTER_S is
# also sent to a second function (same image, same GPU type, its own container pool); the
# first container to claim the call (CLAIM_DICT, atomic put) runs it, the other exits
# without running. Cold starts are normally 6-18 s, but some take 30-90 s even far below
# the GPU cap (newacc probe: 5 containers up, GeoCalib 87 s) and one waited 15.4 min
# (final2 MoGe). A hedge started at 30 s typically runs within ~5 s. 0 disables.
CLAIM_DICT = "v2d-claims"
# GPU type(s) of the hedge twins. Default L40S, like the primaries: outputs on another GPU
# architecture differ beyond the same-host stock noise floor (gate 2026-11-19: all-H100 and
# all-A100 runs fail hand metrics; with the hand chain pinned to L40S, H100 elsewhere still
# shifts FoundationPose's discrete scale search by 2% via 0.02%-level MoGe depth differences).
# Opt in with e.g. "L40S,H100,A100-80GB" once that variance is accepted (OPTIMIZATION.md).
HEDGE_GPU = os.environ.get("V2D_MODAL_HEDGE_GPU", "L40S")  # comma list allowed, as for MODULE_GPU
# Batch mode GPU slots: a token queue (one partition per batch) that every GPU module call
# takes a token from before it starts and returns after. Modal's Starter plan allows 10
# concurrent GPUs per workspace, idle (scaledown) containers included; past that, new
# containers silently wait. Slots keep a batch below the cap, so waits happen in our queue
# where they are measured, and a stalled call means Modal, not our own load.
GPU_SLOTS_QUEUE = "v2d-gpu-slots"
# Single-clip mode (no GPU slots): budget for the containers one invocation holds, including
# speculative warm-ups/resident preloads and idle scaledown containers (tools/modal_v2d/ledger.py).
# Speculation never takes the last GPU_BUDGET_RESERVE entries. 0 disables.
GPU_BUDGET = int(os.environ.get("V2D_MODAL_GPU_BUDGET", "9"))
GPU_BUDGET_RESERVE = int(os.environ.get("V2D_MODAL_GPU_BUDGET_RESERVE", "3"))
HEDGE_AFTER_S = float(os.environ.get("V2D_MODAL_HEDGE_AFTER_S", "30"))
RESIDENT_QUEUE = "v2d-resident"
RESIDENT_HINTS_DICT = "v2d-resident-hints"
RESIDENT_WAIT_S = int(os.environ.get("V2D_MODAL_RESIDENT_WAIT_S", "900"))
# How long a handed-over spec may sit unclaimed (resident container not running) before
# the call runs normally instead.
RESIDENT_CLAIM_S = int(os.environ.get("V2D_MODAL_RESIDENT_CLAIM_S", "45"))
# Only critical-path calls that measurably gain get a resident container (each one holds a
# GPU while it waits). fastrecon R3 smokes (preload untimed): WiLoR ref 17.0->2.4 s, DINO
# 10.8->4.5 s, SAM3D 92.9->62.3 s, FoundationPose scale 75.9->59.8 s, tracking 44.5->31.8 s.
# SAM2 (model loads in ~3 s) and gsplat (nothing to preload beyond imports) showed no gain.
RESIDENT_MODULES = frozenset({
    "v2d.wilor.lib.video_to_hands",
    "v2d.grounding_dino.lib.image_to_object_bboxes",
    "v2d.sam3d.lib.image_to_mesh",
    "v2d.foundation_pose.lib.run_estimate_mesh_scale",
    "v2d.foundation_pose.lib.run_video_to_poses",
})


def _hours(name: str, default: float) -> int:
    hours = float(os.environ.get(name, default))
    if not 0 < hours <= 24:
        raise ValueError(f"{name} must be in (0, 24] hours (Modal's per-call ceiling); got {hours}")
    return int(hours * 3600)


CONTROL_TIMEOUT = _hours("V2D_MODAL_CONTROL_TIMEOUT_H", 4)
RG_TIMEOUT = _hours("V2D_MODAL_RG_TIMEOUT_H", 6)
MODULE_TIMEOUT = _hours("V2D_MODAL_MODULE_TIMEOUT_H", 2)
DOCTOR_TIMEOUT = 45 * 60
SYNC_INTERVAL_S = int(os.environ.get("V2D_MODAL_SYNC_INTERVAL_S", "600"))
HEARTBEAT_S = int(os.environ.get("V2D_MODAL_HEARTBEAT_S", "60"))

# --- Container sizes and list prices (for spend *estimates* in progress output) ---------
CONTROL_CPU, CONTROL_MEM_GIB = 4, 16
RG_CPU, RG_MEM_GIB = 8, 64
MODULE_CPU, MODULE_MEM_GIB = 8, 32
DOWNLOAD_CPU, DOWNLOAD_MEM_GIB = 4, 16
# Module commands that run on the CPU twin of their image even though they ask for a GPU.
# Setup's weight downloads run concurrently before setup (runtime.prefetch_weights).
PREFETCH_WEIGHTS = os.environ.get("V2D_MODAL_PREFETCH", "1") != "0"
PREFETCH_PARALLEL = int(os.environ.get("V2D_MODAL_PREFETCH_PARALLEL", "12"))
# V2D_MODAL_DOWNLOADS_ON_GPU=1 restores docker's behaviour (everything on the GPU function).
CPU_ONLY_MARKERS = () if os.environ.get("V2D_MODAL_DOWNLOADS_ON_GPU") == "1" else ("download_weights",)
# USD per hour; Modal list prices. Billing (`modal billing report`) is the source of truth.
RATES = {
    "gpu_h": {"L40S": 1.95, "A10G": 1.10, "L4": 0.80, "A100": 2.10, "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95, "T4": 0.59},
    "gpu_h_unknown": 4.0,
    "cpu_core_h": 0.0473,
    "mem_gib_h": 0.0080,
}

# --- Roles ---------------------------------------------------------------------------
# Every run_e2e.sh subcommand runs inside one "role" container:
#   control: CPU container; reconstruction modules are dispatched to GPU module functions.
#   rg:      GPU container built from robotic_grounding/workflow/Dockerfile + Isaac-GR00T;
#            the `robotic-grounding` image runs locally, the loader image remotely.
CONTROL_COMMANDS = {"init", "use", "status", "setup", "reconstruct", "import-reconstruction", "inspect"}
RG_COMMANDS = {"doctor", "retarget", "simulate", "train-expert", "collect", "finetune", "evaluate"}
GLOBAL_FLAGS_WITH_VALUE = {"--config"}


def subcommand(argv: list[str]) -> str:
    """Return the run_e2e.sh subcommand in argv (global flags may precede it)."""
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in GLOBAL_FLAGS_WITH_VALUE:
            skip = True
            continue
        if token.startswith("-"):
            continue
        return token
    raise ValueError(f"no run_e2e.sh subcommand in {argv!r}")


def role_for(argv: list[str]) -> str:
    command = subcommand(argv)
    if "--dry-run" in argv:
        return "control"  # only prints commands; never pay for a GPU container to do that
    if command in CONTROL_COMMANDS:
        return "control"
    if command in RG_COMMANDS:
        return "rg"
    raise ValueError(f"unknown run_e2e.sh subcommand {command!r}")


def under(path: str | Path, root: str | Path) -> bool:
    path, root = Path(path), Path(root)
    return path == root or root in path.parents
