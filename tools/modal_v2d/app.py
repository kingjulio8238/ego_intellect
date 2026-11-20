"""Modal app: run `run_e2e.sh` stages on Modal GPUs.

Launch through the wrapper, which pins the Modal profile and picks timeouts:

    MODAL_PROFILE=<profile> tools/modal_v2d/run_e2e_modal.sh <run_e2e.sh args...>

Or directly from the repository root:

    modal run [--detach] -m tools.modal_v2d.app::main --cmd "status"
    modal run -m tools.modal_v2d.app::doctor

Functions:
  control  CPU role container: init/use/status/setup/reconstruct/import-reconstruction/inspect
  rg       GPU role container (Isaac Lab + Isaac-GR00T): doctor/retarget/simulate/
           train-expert/collect/finetune/evaluate
  img_*    one GPU function per docker image; shimmed `docker run` calls land here
  doctor_modal  platform preflight (GPU, Vulkan ICD, Isaac Kit headless boot, GR00T env)
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import time
import uuid
from pathlib import Path

import modal

from tools.modal_v2d import config as C
from tools.modal_v2d.images import build_images

if __package__ != "tools.modal_v2d":
    # Workers import tools.modal_v2d.* in every image; only the package entrypoint form
    # makes Modal mount the `tools` package into containers.
    raise RuntimeError("launch with `modal run -m tools.modal_v2d.app::<entrypoint>` from the repo root")

app = modal.App(C.APP_NAME)
state_volume = modal.Volume.from_name(C.STATE_VOLUME, create_if_missing=True)
weights_volume = modal.Volume.from_name(C.WEIGHTS_VOLUME, create_if_missing=True)
VOLUMES = {C.STATE_MOUNT: state_volume, C.WEIGHTS_MOUNT: weights_volume}
progress_store = modal.Dict.from_name(C.PROGRESS_DICT, create_if_missing=True)
SECRETS = [
    modal.Secret.from_name(name.strip())
    for name in os.environ.get("V2D_MODAL_SECRETS", "v2d-hf").split(",")
    if name.strip()
]

IMAGES = build_images()  # lazy: Dockerfiles and uploads are only read locally at launch

MODULE_KW = dict(
    gpu=C.gpu_spec(C.MODULE_GPU),
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=C.MODULE_TIMEOUT,
    retries=0,
    # Release the GPU quickly between pipeline steps. reconstruct --fast warms each image
    # ahead of its call (`docker pull`), so give warm containers a little longer.
    scaledown_window=C.MODULE_SCALEDOWN_S,
    cpu=C.MODULE_CPU,
    memory=C.MODULE_MEM_GIB * 1024,
)


def _module_kw(image: str, hedge: bool = False) -> dict:
    """MODULE_KW with this image's scaledown window (batch mode: heavy models stay warm
    longer so the next clip's call lands on a container with the model preloaded). Hedge
    twins run on C.HEDGE_GPU."""
    kw = {**MODULE_KW, "scaledown_window": C.SCALEDOWN_PER_IMAGE.get(image, C.MODULE_SCALEDOWN_S)}
    if image in C.GPU_PER_IMAGE:  # an image pinned to certain GPU types keeps them for hedges too
        kw["gpu"] = C.gpu_spec(C.GPU_PER_IMAGE[image])
    elif hedge:
        kw["gpu"] = C.gpu_spec(C.HEDGE_GPU)
    return kw


def _execute(spec: dict) -> dict:
    from tools.modal_v2d import worker

    if spec.get("claim") and not worker.claim(spec):  # hedged call already running elsewhere
        return {"lost_claim": True}
    if spec.get("resident"):  # preload, then wait for the real spec (reconstruct --fast)
        return worker.serve_resident(spec, state_volume=state_volume, weights_volume=weights_volume,
                                     progress_store=progress_store)
    return worker.execute(spec, state_volume=state_volume, weights_volume=weights_volume, progress_store=progress_store)


# One function per docker image. Explicit (not generated) so Modal can locate each by name.
@app.function(image=IMAGES["v2d_anycalib"], **_module_kw("v2d_anycalib"))
def img_v2d_anycalib(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_moge"], **_module_kw("v2d_moge"))
def img_v2d_moge(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_grounding_dino"], **_module_kw("v2d_grounding_dino"))
def img_v2d_grounding_dino(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_sam2"], **_module_kw("v2d_sam2"))
def img_v2d_sam2(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_sam3d"], **_module_kw("v2d_sam3d"))
def img_v2d_sam3d(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_foundation_pose"], **_module_kw("v2d_foundation_pose"))
def img_v2d_foundation_pose(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_hamer"], **_module_kw("v2d_hamer"))
def img_v2d_hamer(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_wilor"], **_module_kw("v2d_wilor"))
def img_v2d_wilor(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_geocalib"], **_module_kw("v2d_geocalib"))
def img_v2d_geocalib(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_droid_slam"], **_module_kw("v2d_droid_slam"))
def img_v2d_droid_slam(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_gsplat_refinement"], **_module_kw("v2d_gsplat_refinement"))
def img_v2d_gsplat_refinement(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_task_library_loader"], **_module_kw("v2d_task_library_loader"))
def img_v2d_task_library_loader(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES[C.RG_IMAGE], **{**MODULE_KW, "gpu": C.RG_GPU, "cpu": C.RG_CPU, "memory": C.RG_MEM_GIB * 1024})
def img_robotic_grounding(spec: dict) -> dict:
    """`docker exec` into the robotic-grounding image from the control role (e.g. setup)."""
    from tools.modal_v2d import runtime

    runtime.ensure_repo()  # the shimmed repo bind mount maps onto this copy (baked into the image)
    return _execute(spec)


IMAGE_FUNCTIONS = {
    "v2d_anycalib": img_v2d_anycalib,
    "v2d_moge": img_v2d_moge,
    "v2d_grounding_dino": img_v2d_grounding_dino,
    "v2d_sam2": img_v2d_sam2,
    "v2d_sam3d": img_v2d_sam3d,
    "v2d_foundation_pose": img_v2d_foundation_pose,
    "v2d_hamer": img_v2d_hamer,
    "v2d_wilor": img_v2d_wilor,
    "v2d_geocalib": img_v2d_geocalib,
    "v2d_droid_slam": img_v2d_droid_slam,
    "v2d_gsplat_refinement": img_v2d_gsplat_refinement,
    "v2d_task_library_loader": img_v2d_task_library_loader,
    C.RG_IMAGE: img_robotic_grounding,
}
assert set(IMAGE_FUNCTIONS) == set(C.ALL_IMAGES), "every image needs a Modal function"

# CPU twins of the module functions. Weight downloads pass `--gpus all` like every other
# module command, but need no GPU: on a CPU container they start faster and cost a
# fraction of an L40S (runtime.StateSync.run_remote routes them; see C.CPU_ONLY_MARKERS).
CPU_KW = {**MODULE_KW, "cpu": C.DOWNLOAD_CPU, "memory": C.DOWNLOAD_MEM_GIB * 1024}
CPU_KW.pop("gpu")
# Explicit (not generated): serialized functions would need the local Python version to
# match each image's, which differs across module images.


@app.function(image=IMAGES["v2d_anycalib"], **CPU_KW)
def cpu_v2d_anycalib(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_moge"], **CPU_KW)
def cpu_v2d_moge(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_grounding_dino"], **CPU_KW)
def cpu_v2d_grounding_dino(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_sam2"], **CPU_KW)
def cpu_v2d_sam2(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_sam3d"], **CPU_KW)
def cpu_v2d_sam3d(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_foundation_pose"], **CPU_KW)
def cpu_v2d_foundation_pose(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_hamer"], **CPU_KW)
def cpu_v2d_hamer(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_wilor"], **CPU_KW)
def cpu_v2d_wilor(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_geocalib"], **CPU_KW)
def cpu_v2d_geocalib(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_droid_slam"], **CPU_KW)
def cpu_v2d_droid_slam(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_gsplat_refinement"], **CPU_KW)
def cpu_v2d_gsplat_refinement(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_task_library_loader"], **CPU_KW)
def cpu_v2d_task_library_loader(spec: dict) -> dict:
    return _execute(spec)


CPU_FUNCTIONS = {
    "v2d_anycalib": cpu_v2d_anycalib,
    "v2d_moge": cpu_v2d_moge,
    "v2d_grounding_dino": cpu_v2d_grounding_dino,
    "v2d_sam2": cpu_v2d_sam2,
    "v2d_sam3d": cpu_v2d_sam3d,
    "v2d_foundation_pose": cpu_v2d_foundation_pose,
    "v2d_hamer": cpu_v2d_hamer,
    "v2d_wilor": cpu_v2d_wilor,
    "v2d_geocalib": cpu_v2d_geocalib,
    "v2d_droid_slam": cpu_v2d_droid_slam,
    "v2d_gsplat_refinement": cpu_v2d_gsplat_refinement,
    "v2d_task_library_loader": cpu_v2d_task_library_loader,
}
assert set(CPU_FUNCTIONS) == set(C.MODULE_IMAGES), "every module image needs a CPU twin"


# Hedge twins: same image and GPU, separate container pool (runtime.StateSync._call_hedged).
# Explicit for the same reason as the CPU twins.


@app.function(image=IMAGES["v2d_anycalib"], **_module_kw("v2d_anycalib", hedge=True))
def hedge_v2d_anycalib(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_moge"], **_module_kw("v2d_moge", hedge=True))
def hedge_v2d_moge(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_grounding_dino"], **_module_kw("v2d_grounding_dino", hedge=True))
def hedge_v2d_grounding_dino(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_sam2"], **_module_kw("v2d_sam2", hedge=True))
def hedge_v2d_sam2(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_sam3d"], **_module_kw("v2d_sam3d", hedge=True))
def hedge_v2d_sam3d(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_foundation_pose"], **_module_kw("v2d_foundation_pose", hedge=True))
def hedge_v2d_foundation_pose(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_hamer"], **_module_kw("v2d_hamer", hedge=True))
def hedge_v2d_hamer(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_wilor"], **_module_kw("v2d_wilor", hedge=True))
def hedge_v2d_wilor(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_geocalib"], **_module_kw("v2d_geocalib", hedge=True))
def hedge_v2d_geocalib(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_droid_slam"], **_module_kw("v2d_droid_slam", hedge=True))
def hedge_v2d_droid_slam(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_gsplat_refinement"], **_module_kw("v2d_gsplat_refinement", hedge=True))
def hedge_v2d_gsplat_refinement(spec: dict) -> dict:
    return _execute(spec)


@app.function(image=IMAGES["v2d_task_library_loader"], **_module_kw("v2d_task_library_loader", hedge=True))
def hedge_v2d_task_library_loader(spec: dict) -> dict:
    return _execute(spec)


HEDGE_FUNCTIONS = {
    "v2d_anycalib": hedge_v2d_anycalib,
    "v2d_moge": hedge_v2d_moge,
    "v2d_grounding_dino": hedge_v2d_grounding_dino,
    "v2d_sam2": hedge_v2d_sam2,
    "v2d_sam3d": hedge_v2d_sam3d,
    "v2d_foundation_pose": hedge_v2d_foundation_pose,
    "v2d_hamer": hedge_v2d_hamer,
    "v2d_wilor": hedge_v2d_wilor,
    "v2d_geocalib": hedge_v2d_geocalib,
    "v2d_droid_slam": hedge_v2d_droid_slam,
    "v2d_gsplat_refinement": hedge_v2d_gsplat_refinement,
    "v2d_task_library_loader": hedge_v2d_task_library_loader,
}
assert set(HEDGE_FUNCTIONS) == set(C.MODULE_IMAGES), "every module image needs a hedge twin"


# --------------------------------------------------------------------------- run locks
def _lock_keys(argv: list[str]) -> list[str]:
    """One lock per run root this invocation syncs, so different clips run concurrently
    (batch mode) while two invocations on the same run root still exclude each other."""
    from tools.modal_v2d import runtime

    roots = runtime.runs_touched(argv)
    if not roots:
        active = runtime._active_run_from_volume()
        roots = {active} if active else set()
    return sorted(f"run:{root}" for root in roots) or ["state"]


def _acquire_lock(role: str, argv: list[str], break_lock: bool, owner_id: str = "") -> tuple[str, list[str]]:
    """Take this invocation's run locks; return (ownership token, keys)."""
    locks = modal.Dict.from_name(C.LOCK_DICT, create_if_missing=True)
    token = uuid.uuid4().hex
    keys = _lock_keys(argv)
    owner = {
        "token": token,
        "role": role,
        "argv": argv,
        "host": socket.gethostname(),
        "since": time.time(),
        "owner_id": owner_id,  # same id = Modal restarting this very input after a preemption
    }
    taken: list[str] = []
    for key in keys:
        if break_lock:
            previous = locks.pop(key, None)
            if previous:
                print(f"[v2d-modal] broke lock {key} held by {previous}")
        if not locks.put(key, owner, skip_if_exists=True):
            holder = locks.get(key)
            if owner_id and holder and holder.get("owner_id") == owner_id:
                print(f"[v2d-modal] taking over lock {key} from this input's preempted attempt")
                locks.put(key, owner)
                taken.append(key)
                continue
            for done in taken:
                locks.pop(done, None)
            raise RuntimeError(
                f"another run_e2e.sh invocation holds the lock {key}: {holder}. Wait for it, or if it "
                "is dead (check `modal app list`), rerun with V2D_MODAL_BREAK_LOCK=1."
            )
        taken.append(key)
    return token, keys


def _release_lock(token: str, keys: list[str]) -> None:
    """Release only our own locks (a --break-lock may have handed one to another run)."""
    locks = modal.Dict.from_name(C.LOCK_DICT, create_if_missing=True)
    for key in keys:
        holder = locks.get(key, None)
        if holder and holder.get("token") == token:
            locks.pop(key, None)
        else:
            print(f"[v2d-modal] lock {key} now held by {holder}; not releasing it")


def _run(role: str, argv: list[str], source_revision: str, break_lock: bool, invocation: str, app_id: str,
         options: dict | None = None) -> int:
    from tools.modal_v2d import runtime

    token, keys = _acquire_lock(role, argv, break_lock, owner_id=invocation)
    try:
        return runtime.run_role(
            role,
            argv,
            source_revision=source_revision,
            state_volume=state_volume,
            weights_volume=weights_volume,
            functions=IMAGE_FUNCTIONS,
            cpu_functions=CPU_FUNCTIONS,
            hedge_functions=HEDGE_FUNCTIONS,
            options=options,
            progress_store=progress_store,
            invocation=invocation,
            app_id=app_id,
        )
    finally:
        _release_lock(token, keys)


def _run_steps(role: str, argv, source_revision: str, break_lock: bool, invocation: str, app_id: str,
               options: dict | None = None) -> int:
    """argv is one run_e2e.sh argv, or a list of them run in order in this container
    (batch mode: init -> setup -> reconstruct for one clip, one container start)."""
    from tools.modal_v2d import runtime

    steps = argv if argv and isinstance(argv[0], list) else [argv]
    for index, step in enumerate(steps):
        suffix = f"-s{index + 1}" if len(steps) > 1 else ""
        if len(steps) > 1 and runtime.init_done(step, mirrored=True):
            print(f"[v2d-modal] {step[2] if len(step) > 2 else step}: already initialized; skipping init", flush=True)
            continue
        code = _run(role, step, source_revision, break_lock, invocation + suffix, app_id, options)
        if code != 0:
            return code
    return 0


@app.function(
    image=IMAGES["control"],
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=C.CONTROL_TIMEOUT,
    retries=0,
    cpu=C.CONTROL_CPU,
    memory=C.CONTROL_MEM_GIB * 1024,
)
def control(argv: list, source_revision: str, break_lock: bool = False, invocation: str = "", app_id: str = "",
            options: dict | None = None) -> int:
    return _run_steps("control", argv, source_revision, break_lock, invocation, app_id, options)


@app.function(
    image=IMAGES[C.RG_IMAGE],
    gpu=C.RG_GPU,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=C.RG_TIMEOUT,
    retries=0,
    cpu=C.RG_CPU,
    memory=C.RG_MEM_GIB * 1024,
)
def rg(argv: list, source_revision: str, break_lock: bool = False, invocation: str = "", app_id: str = "",
       options: dict | None = None) -> int:
    return _run_steps("rg", argv, source_revision, break_lock, invocation, app_id, options)


@app.function(
    image=IMAGES["control"],
    volumes=VOLUMES,
    timeout=C.CONTROL_TIMEOUT,
    retries=0,
    cpu=1,
    memory=1024,
)
def batch_driver(clips: list[dict], max_parallel: int, source_revision: str, batch_id: str, app_id: str = "",
                 options: dict | None = None) -> list[dict]:
    """Run each clip's steps in its own control container, at most `max_parallel` at once.
    Remote (not the local entrypoint) so a detached batch keeps launching clips after the
    laptop disconnects."""
    from concurrent.futures import ThreadPoolExecutor

    def run(clip: dict) -> dict:
        start = time.time()
        try:
            code = control.remote(clip["steps"], source_revision, False, f"{batch_id}-{clip['name']}", app_id, options)
        except Exception as exc:  # noqa: BLE001  - one clip failing must not stop the batch
            code = f"{type(exc).__name__}: {exc}"[:200]
        result = {"name": clip["name"], "code": code, "start": start, "end": time.time()}
        print(f"[v2d-batch {time.strftime('%H:%M:%SZ', time.gmtime())}] {clip['name']}: "
              f"{'ok' if code == 0 else 'FAILED ' + str(code)} in {result['end'] - start:.0f}s", flush=True)
        progress_store.put(f"batch:{batch_id}:{clip['name']}", result)
        return result

    options = dict(options or {})
    slots = int(options.pop("gpu_slots", 0) or 0)
    if slots > 0:
        queue = modal.Queue.from_name(C.GPU_SLOTS_QUEUE, create_if_missing=True)
        options["gpu_slots_key"] = batch_id
        queue.clear(partition=batch_id)
        queue.put_many(list(range(slots)), partition=batch_id)
    print(f"[v2d-batch] {batch_id}: {len(clips)} clips, {max_parallel} at a time, options {options}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
        results = list(pool.map(run, clips))
    from tools.modal_v2d import batch as B

    summary = B.summarize(results)
    print(summary, flush=True)
    progress_store.put(f"batch:{batch_id}", {"results": results, "summary": summary})
    return results


@app.function(
    image=IMAGES["control"],
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=C.CONTROL_TIMEOUT,
    retries=0,
    cpu=8,  # one run_e2e.sh orchestration per clip in flight
    memory=32 * 1024,
)
def control_batch(clips: list[dict], max_parallel: int, source_revision: str, batch_id: str, app_id: str = "",
                  options: dict | None = None) -> list[dict]:
    """All clips of a batch in one control container (runtime.run_batch_role)."""
    from tools.modal_v2d import runtime

    roots = runtime._clip_roots(clips)
    argv = ["init", *[token for root in roots for token in ("--run-root", root)]]
    token, keys = _acquire_lock("control", argv, False, owner_id=batch_id)
    try:
        return runtime.run_batch_role(
            clips, source_revision=source_revision, state_volume=state_volume, weights_volume=weights_volume,
            functions=IMAGE_FUNCTIONS, cpu_functions=CPU_FUNCTIONS, hedge_functions=HEDGE_FUNCTIONS,
            progress_store=progress_store, batch_id=batch_id, app_id=app_id, options=options,
            max_parallel=max_parallel,
        )
    finally:
        _release_lock(token, keys)


@app.function(
    image=IMAGES["control"],
    volumes=VOLUMES,
    timeout=C.CONTROL_TIMEOUT,
    retries=0,
    cpu=1,
    memory=1024,
)
def batch_pipeline(clips: list[dict], max_parallel: int, source_revision: str, batch_id: str, app_id: str = "",
                   options: dict | None = None, mode: str = "shared", robot: list[dict] | None = None,
                   render_every: int = 20, robot_parallel: int = 4) -> dict:
    """The whole batch, remotely (a detached batch keeps going after the laptop disconnects):

    1. reconstruct: every clip's control steps (control_batch, or batch_driver per clip);
    2. robot: per reconstructed clip, its robot steps (retarget, simulate) in one rg container,
       `robot_parallel` clips at a time;
    3. render: the optional environment video for flagged clips plus a 1-in-`render_every`
       sample, one clip at a time, after everything else.
    """
    from concurrent.futures import ThreadPoolExecutor

    from tools.modal_v2d import batch as B
    from tools.modal_v2d import sync

    def stamp() -> str:
        return time.strftime("%H:%M:%SZ", time.gmtime())

    t0 = time.time()
    if mode == "shared":
        recon = control_batch.remote(clips, max_parallel, source_revision, batch_id, app_id, options)
    elif mode == "per_clip":
        recon = batch_driver.remote(clips, max_parallel, source_revision, batch_id, app_id, options)
    else:
        raise ValueError(f"unknown batch mode {mode!r}")
    print(B.summarize(recon), flush=True)
    t_recon = time.time()
    ok = [r["name"] for r in recon if r["code"] == 0]
    steps = {item["name"]: item["steps"] for item in (robot or [])}

    def robot_one(name: str) -> dict:
        start = time.time()
        try:
            code = rg.remote(steps[name], source_revision, False, f"{batch_id}-{name}-robot", app_id, options)
        except Exception as exc:  # noqa: BLE001 - one clip failing must not stop the batch
            code = f"{type(exc).__name__}: {exc}"[:200]
        result = {"name": name, "code": code, "start": start, "end": time.time()}
        print(f"[v2d-batch {stamp()}] {name}: robot steps {'ok' if code == 0 else 'FAILED ' + str(code)} "
              f"in {result['end'] - start:.0f}s", flush=True)
        progress_store.put(f"batch:{batch_id}:{name}:robot", result)
        return result

    robot_results = []
    todo = [name for name in ok if steps.get(name)]
    if todo:
        with ThreadPoolExecutor(max_workers=max(1, robot_parallel)) as pool:
            robot_results = list(pool.map(robot_one, todo))
    t_robot = time.time()

    state_volume.reload()
    qa: dict[str, dict] = {}
    for result in robot_results:
        if result["code"] != 0:
            continue
        path = sync.mirror_path(C.MIRROR_ROOT, B.qa_report_path(result["name"]))
        try:
            report = json.loads(Path(path).read_text())
            qa[result["name"]] = {"flag": bool(report.get("flag")), "reasons": report.get("flag_reasons", [])}
        except (OSError, ValueError) as exc:
            qa[result["name"]] = {"flag": None, "reasons": [f"no QA report: {exc}"]}
    flagged = [name for name, item in qa.items() if item["flag"]]
    simulated = [r["name"] for r in robot_results if r["code"] == 0]
    queue = B.render_queue(simulated, render_every, flagged)
    print(f"[v2d-batch {stamp()}] QA: {len(flagged)}/{len(qa)} flagged {flagged}; "
          f"render queue (flagged + 1 in {render_every}): {queue}", flush=True)

    renders = []
    for name, argv in zip(queue, B.render_commands(queue)):  # low priority: one at a time, last
        start = time.time()
        try:
            code = rg.remote(argv, source_revision, False, f"{batch_id}-{name}-render", app_id, options)
        except Exception as exc:  # noqa: BLE001
            code = f"{type(exc).__name__}: {exc}"[:200]
        renders.append({"name": name, "code": code, "seconds": round(time.time() - start, 1)})
        print(f"[v2d-batch {stamp()}] {name}: render {'ok' if code == 0 else 'FAILED ' + str(code)}", flush=True)
    t_end = time.time()

    def ok_code(name: str, results: list[dict]) -> bool:
        return any(r["name"] == name and r["code"] == 0 for r in results)

    delivered = [name for name in ok if not steps.get(name) or ok_code(name, robot_results)]
    summary = {
        "batch_id": batch_id,
        "clips": len(clips),
        "reconstructed": ok,
        "robot": robot_results,
        "delivered": delivered,
        "qa": qa,
        "flagged": flagged,
        "render_every": render_every,
        "render_queue": queue,
        "renders": renders,
        "phase_s": {"reconstruct": round(t_recon - t0), "robot": round(t_robot - t_recon),
                    "render": round(t_end - t_robot)},
        "delivered_per_h": round(len(delivered) / max(t_robot - t0, 1e-9) * 3600, 2),
    }
    print(f"[v2d-batch {stamp()}] {len(delivered)}/{len(clips)} delivered in {(t_robot - t0) / 60:.1f} min "
          f"(render after delivery: {(t_end - t_robot) / 60:.1f} min) | {json.dumps(summary['phase_s'])}", flush=True)
    progress_store.put(f"batch:{batch_id}:pipeline", summary)
    return {"recon": recon, "pipeline": summary}


@app.function(image=IMAGES["control"], volumes=VOLUMES, timeout=1800, retries=0, cpu=1, memory=1024)
def startup_probe_driver(images: list[str], rounds: int) -> list[dict]:
    """Container start latency of real module images: one no-op call per image per round,
    all concurrent, each on a fresh container that exits right after (no idle billing)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(job: tuple[str, int]) -> dict:
        image, round_ = job
        spec = {"image": image, "argv": ["true"], "env": {}, "workdir": None, "mounts": [],
                "call_id": f"probe-{uuid.uuid4().hex[:8]}", "exit_after": True}
        t0 = time.time()
        try:
            result = IMAGE_FUNCTIONS[image].remote(spec)
            timing = result.get("timing") or {}
            total = time.time() - t0
            age = timing.get("container_age_s")
            return {"image": image, "round": round_, "total_s": round(total, 1),
                    "boot_s": age, "wait_s": round(total - timing.get("worker_s", 0) - (age or 0), 1)}
        except Exception as exc:  # noqa: BLE001
            return {"image": image, "round": round_, "error": f"{type(exc).__name__}: {exc}"[:200]}

    jobs = [(image, r) for r in range(rounds) for image in images]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        return list(pool.map(one, jobs))


DOCTOR_SCRIPT = r"""
set -u -o pipefail
section() { printf '\n===== %s =====\n' "$1"; }
section "nvidia-smi"; nvidia-smi || echo "FAIL: nvidia-smi"
section "driver capabilities"; echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-unset}"
section "Vulkan ICD files"; ls -l /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1
# Informational: Modal may inject driver libraries without refreshing the ldconfig cache.
section "NVIDIA graphics libraries"; ldconfig -p | grep -E 'libGLX_nvidia|libEGL_nvidia|libnvidia-glcore|libnvidia-vulkan' || echo "WARN: no NVIDIA graphics libraries in the ldconfig cache"
section "vulkaninfo"; vulkaninfo --summary > /tmp/vulkaninfo.txt 2>&1 && head -40 /tmp/vulkaninfo.txt || { head -40 /tmp/vulkaninfo.txt; echo "WARN: vulkaninfo failed (the Kit boot below is the authoritative check)"; }
section "NVIDIA driver libraries present (RTX needs rtcore/OptiX/gpucomp)"
ls /usr/lib/x86_64-linux-gnu 2>/dev/null | grep -E 'libnvidia|libnvoptix|libcuda|libGLX_nvidia|libEGL_nvidia' | sort | tr '\n' ' '; echo
for lib in libnvoptix.so.1 libnvidia-rtcore.so libnvidia-gpucomp.so libnvidia-glvkspirv.so libnvidia-ngx.so.1; do
  if ls /usr/lib/x86_64-linux-gnu/${lib}* >/dev/null 2>&1; then echo "present: $lib"; else echo "WARN: missing $lib"; fi
done
section "USD provenance (pip usd-core can shadow Kit's own USD)"
/opt/isaac/bin/python -m pip show usd-core 2>/dev/null | grep -E '^(Name|Version|Location)' || echo "usd-core: not pip-installed"
/opt/isaac/bin/python -c "import pxr, os; print('pxr from', os.path.dirname(pxr.__file__))" 2>&1 | tail -2
kit_boot() {
  cd /workspace/isaaclab && timeout 1500 ./isaaclab.sh -p -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app
print('$1')
app.close()
" > "$2" 2>&1
  local code=$?
  echo "exit=$code; log lines=$(wc -l < "$2")"
  grep -m1 -E "^$1" "$2" || {
    echo "--- GPU/RTX/driver lines before the crash ---"
    sed '/Crash detected/q' "$2" | grep -i -E "gpu|vulkan|rtx|driver|device|optix|rtcore|\[Error\]|\[Warning\]" | head -60 | cut -c1-260
    echo "--- first errors ---"; grep -m25 -E "\[Error\]|\[Fatal\]|Traceback|Error:" "$2" | cut -c1-260
    echo "--- last lines ---"; tail -8 "$2" | cut -c1-260
    return 1
  }
}
section "Isaac Kit headless boot"
if ! kit_boot "V2D_DOCTOR: Isaac Kit started headless with cameras enabled" /tmp/kit_boot.log; then
  echo "FAIL: Isaac Kit headless boot"
  # Isolation: is it the camera/RTX renderer? (train-expert and the collect pilot are camera-free)
  section "A/B: Kit boot without cameras"
  cd /workspace/isaaclab && timeout 900 ./isaaclab.sh -p -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=False).app
print('V2D_AB: Kit boots headless WITHOUT cameras')
app.close()
" > /tmp/kit_nocam.log 2>&1; echo "exit=$?"
  grep -m1 "^V2D_AB" /tmp/kit_nocam.log || { echo "V2D_AB: Kit fails even without cameras"; sed '/Crash detected/q' /tmp/kit_nocam.log | grep -E "\[Error\]" | head -10 | cut -c1-260; }
fi
section "Isaac Lab E2E python deps"; /opt/isaac/bin/python -c "import msgpack_numpy, zmq, pinocchio, pink; print('V2D_DOCTOR: isaac python deps OK')" 2>&1 | tail -5
section "Isaac-GR00T env"; cd /opt/Isaac-GR00T && uv run python -c "import gr00t, torch; print('V2D_DOCTOR: gr00t', torch.__version__, torch.cuda.is_available())" 2>&1 | tail -5
section "volumes"; ls -la /vol /weights; ls -la /weights/mano/models 2>&1
"""


@app.function(
    image=IMAGES[C.RG_IMAGE],
    gpu=C.RG_GPU,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=C.DOCTOR_TIMEOUT,
    retries=0,
)
def doctor_modal() -> str:
    """Cheapest gate: prove the Modal GPU can boot Isaac Kit before any paid pipeline run."""
    import threading

    from tools.modal_v2d import containerenv, progress

    # Same environment the pipeline gives Isaac/GR00T commands (image PYTHONPATH, not Modal's).
    env = containerenv.container_env(image=C.RG_IMAGE)
    started = time.time()
    rate = progress.hourly_rate(C.RG_GPU, 0.125, 1)  # default function size: GPU dominates
    section = {"name": "starting"}
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(30):
            print(
                f"[doctor {progress.utc()}] ♥ section '{section['name']}' | elapsed "
                f"{progress.hms(time.time() - started)} | cap {C.DOCTOR_TIMEOUT // 60} min | "
                f"~${(time.time() - started) / 3600 * rate:.2f}",
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    lines: list[str] = []
    process = subprocess.Popen(
        ["bash", "-c", DOCTOR_SCRIPT], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace"
    )
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        text = line.rstrip("\n")
        if text.startswith("===== "):
            section["name"] = text.strip("= ")
            print(f"\n[doctor {progress.utc()}] ━━━━ {section['name']} ━━━━", flush=True)
        else:
            print(f"[doctor {progress.utc()}] {text}", flush=True)
    process.wait()
    stop.set()
    print(f"[doctor {progress.utc()}] finished in {progress.hms(time.time() - started)} (~${(time.time() - started) / 3600 * rate:.2f})", flush=True)
    return "".join(lines)


# --------------------------------------------------------------------------- local entry
def _source_revision() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        return head + ("+dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _tag(tags: dict[str, str]) -> None:
    try:
        app.set_tags(tags)
        print(f"[v2d-modal] app {app.app_id} tagged {tags}")
    except Exception as exc:  # billing attribution falls back to the app id
        print(f"[v2d-modal] WARNING: could not tag app {app.app_id}: {type(exc).__name__}: {exc}")


@app.local_entrypoint()
def main(cmd: str = "", argv_json: str = "", break_lock: bool = False) -> None:
    """Run one run_e2e.sh invocation: --cmd "status" or --argv-json '["status"]'."""
    if bool(cmd) == bool(argv_json):
        raise SystemExit("pass exactly one of --cmd or --argv-json")
    argv = json.loads(argv_json) if argv_json else shlex.split(cmd)
    if not (isinstance(argv, list) and argv and all(isinstance(item, str) for item in argv)):
        raise SystemExit(f"invalid run_e2e.sh argv: {argv!r}")
    role = C.role_for(argv)
    function = control if role == "control" else rg
    revision = _source_revision()
    timeout_h = (C.CONTROL_TIMEOUT if role == "control" else C.RG_TIMEOUT) / 3600
    invocation = time.strftime("%m%d-%H%M", time.gmtime()) + "-" + uuid.uuid4().hex[:4]
    print(f"[v2d-modal] invocation {invocation}: role={role} timeout={timeout_h:g}h source={revision} argv={argv}")
    print(f"[v2d-modal] watch: MODAL_PROFILE={os.environ.get('MODAL_PROFILE', '<profile>')} python -m tools.modal_v2d.board --watch")
    # Tags put this invocation on its own rows in `modal billing report` (real cost per step).
    _tag({"v2d_invocation": invocation, "v2d_command": C.subcommand(argv), "v2d_role": role})
    code = function.remote(argv, revision, break_lock, invocation, app.app_id or "")
    if code != 0:
        raise SystemExit(f"run_e2e.sh {C.subcommand(argv)} failed with exit code {code}")


@app.local_entrypoint()
def doctor() -> None:
    _tag({"v2d_invocation": "doctor-" + time.strftime("%m%d-%H%M", time.gmtime()), "v2d_command": "modal-doctor", "v2d_role": "doctor"})
    report = doctor_modal.remote()
    failures = [line for line in report.splitlines() if "FAIL" in line]
    markers = [line for line in report.splitlines() if line.startswith(("V2D_DOCTOR", "V2D_AB"))]
    print("\n".join(["", "== summary ==", *markers, *failures]))
    if failures or sum(m.startswith("V2D_DOCTOR") for m in markers) < 3:
        raise SystemExit("Modal platform doctor failed; do not launch pipeline stages")


@app.local_entrypoint()
def batch(manifest: str, max_parallel: int = 4) -> None:
    """Run a manifest of clips concurrently (tools/modal_v2d/batch.py)."""
    from pathlib import Path

    from tools.modal_v2d import batch as B

    spec = B.load_manifest(Path(manifest).read_text())
    clips = [{"name": clip["name"], "steps": B.clip_steps(clip, spec)} for clip in spec["clips"]]
    batch_id = "b" + time.strftime("%m%d-%H%M", time.gmtime()) + "-" + uuid.uuid4().hex[:4]
    print(f"[v2d-batch] {batch_id}: {len(clips)} clips, max {max_parallel} in parallel, app {app.app_id}")
    for clip in clips:
        for step in clip["steps"]:
            print(f"  {clip['name']}: run_e2e.sh {' '.join(step)}")
    _tag({"v2d_invocation": batch_id, "v2d_command": "batch", "v2d_role": "batch"})
    robot = [{"name": clip["name"], "steps": B.robot_steps(clip["name"], spec)} for clip in spec["clips"]]
    for item in robot:
        for step in item["steps"]:
            print(f"  {item['name']}: run_e2e.sh {' '.join(step)}  (robot phase, rg container)")
    render_every = int(spec.get("render_every", B.RENDER_EVERY))
    options = {"scaledown_s": C.MODULE_SCALEDOWN_S, "scaledown_per_image": C.SCALEDOWN_PER_IMAGE,
               **(spec.get("options") or {})}
    mode = spec.get("mode", "shared")
    print(f"[v2d-batch] mode {mode} | options {options} | module scaledown {C.MODULE_SCALEDOWN_S}s")
    if mode not in ("shared", "per_clip"):  # shared: one control container, one GPU ledger
        raise SystemExit(f"unknown batch mode {mode!r} (shared | per_clip)")
    out = batch_pipeline.remote(clips, max_parallel, _source_revision(), batch_id, app.app_id or "", options,
                                mode, robot, render_every, int(spec.get("robot_parallel", 4)))
    results, pipeline = out["recon"], out["pipeline"]
    print(B.summarize(results))
    record = Path(__file__).resolve().parents[2] / ".e2e" / "batch" / batch_id / "pipeline.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(pipeline, indent=1))
    print(f"[v2d-batch] delivered {len(pipeline['delivered'])}/{len(results)} | flagged {pipeline['flagged']} | "
          f"rendered {[r['name'] for r in pipeline['renders'] if r['code'] == 0]} -> {record}")
    if len(pipeline["delivered"]) != len(results) or any(r["code"] != 0 for r in pipeline["renders"]):
        raise SystemExit("some clips failed")


@app.local_entrypoint()
def startup_probe(images: str = "", rounds: int = 1) -> None:
    """Cold-start latency per module image: `modal run -m tools.modal_v2d.app::startup_probe`."""
    names = [name for name in images.split(",") if name] or [n for n in C.MODULE_IMAGES if n != "v2d_task_library_loader"]
    _tag({"v2d_invocation": "probe-" + time.strftime("%m%d-%H%M", time.gmtime()), "v2d_command": "startup-probe", "v2d_role": "probe"})
    for row in startup_probe_driver.remote(names, rounds):
        print(json.dumps(row), flush=True)


@app.local_entrypoint()
def hedge_check(image: str = "v2d_moge", primary_delay_s: float = 0.0) -> None:
    """Real-infra check of hedged dispatch: the primary claims `primary_delay_s` after it
    starts (simulated stall). Expect the hedge to run it when the delay exceeds
    C.HEDGE_AFTER_S plus the hedge's own start, the primary otherwise; never both."""
    from tools.modal_v2d.runtime import StateSync

    state = StateSync.__new__(StateSync)
    state.tracker = None
    state.hedge_functions = HEDGE_FUNCTIONS
    spec = {"image": image, "argv": ["true"], "env": {}, "workdir": None, "mounts": [],
            "call_id": f"hedgecheck-{uuid.uuid4().hex[:6]}", "exit_after": True,
            "debug_delay_claim_s": primary_delay_s}
    t0 = time.time()
    result = state._call_hedged(image, IMAGE_FUNCTIONS[image], spec, lambda text: print(text, end=""), on_cpu=False)
    print(json.dumps({"image": image, "primary_delay_s": primary_delay_s, "hedge_after_s": C.HEDGE_AFTER_S,
                      "wall_s": round(time.time() - t0, 1), **{k: result["timing"].get(k) for k in ("lane", "hedged", "claim_s", "container_age_s")},
                      "returncode": result["returncode"]}))


@app.local_entrypoint()
def warm_check(spec_file: str, calls: int = 2) -> None:
    """Real-infra check of keep-warm preloading: run one recorded call spec `calls` times on
    the same function, one after another. Calls after the first should land on the same
    container and run on the preloaded process (timing.resident)."""
    from pathlib import Path

    spec = json.loads(Path(spec_file).read_text())
    for index in range(calls):
        t0 = time.time()
        result = IMAGE_FUNCTIONS[spec["image"]].remote({**spec, "call_id": f"warmcheck-{index}", "keep_warm": True,
                                                        "exit_after": index == calls - 1})
        timing = result.get("timing") or {}
        print(json.dumps({"call": index, "returncode": result["returncode"], "wall_s": round(time.time() - t0, 1),
                          **{k: timing.get(k) for k in ("exec_s", "resident", "preload_s", "container_age_s")}}), flush=True)


@app.function(image=IMAGES["control"], timeout=1800, retries=0, cpu=1, memory=1024)
def cap_probe_driver(images: list[str], hold_s: float, gap_s: float, reuse: bool) -> list[dict]:
    """Wave 1: one container per image runs `sleep hold_s` and exits (reuse=False) or stays
    (reuse=True). Wave 2 starts gap_s after wave 1 ends, one no-op call per image. Wave 2's
    start latency shows whether exited containers still count against the GPU cap."""
    from concurrent.futures import ThreadPoolExecutor

    def one(job: tuple[str, int]) -> dict:
        image, wave = job
        argv = ["sleep", str(hold_s)] if wave == 1 else ["true"]
        spec = {"image": image, "argv": argv, "env": {}, "workdir": None, "mounts": [],
                "call_id": f"cap{wave}-{uuid.uuid4().hex[:6]}", "exit_after": (wave == 1 and not reuse) or wave == 2}
        t0 = time.time()
        timing = (IMAGE_FUNCTIONS[image].remote(spec).get("timing") or {})
        return {"image": image, "wave": wave, "t0": round(t0, 1), "end": round(time.time(), 1),
                "start_s": round(time.time() - t0 - timing.get("worker_s", 0), 1), "age_s": timing.get("container_age_s")}

    with ThreadPoolExecutor(len(images)) as pool:
        first = list(pool.map(one, [(i, 1) for i in images]))
    time.sleep(gap_s)
    with ThreadPoolExecutor(len(images)) as pool:
        second = list(pool.map(one, [(i, 2) for i in images]))
    return first + second


@app.local_entrypoint()
def cap_probe(n: int = 10, hold_s: float = 20, gap_s: float = 5, reuse: bool = False) -> None:
    images = [i for i in C.MODULE_IMAGES if i != "v2d_task_library_loader"][:n]
    for row in cap_probe_driver.remote(images, hold_s, gap_s, reuse):
        print(json.dumps(row), flush=True)


GPU_SMOKE_COMMON = (
    "import torch; assert torch.cuda.is_available(); "
    "print('torch', torch.__version__, 'cuda', torch.version.cuda, 'sm', torch.cuda.get_device_capability(), torch.cuda.get_device_name()); "
    "x = torch.randn(256, 256, device='cuda'); assert torch.isfinite((x @ x).sum()); "
    "import torch.nn.functional as F; F.conv2d(torch.randn(1, 3, 32, 32, device='cuda'), torch.randn(4, 3, 3, 3, device='cuda')); "
)
# One line per image: load every compiled CUDA extension the image's code paths use and run
# a tiny kernel where that is cheap. A Blackwell port fails here, per image, in seconds.
GPU_SMOKE = {
    "v2d_sam3d": "import gsplat; from gsplat.cuda._backend import _C; assert _C is not None; "
                 "import spconv.pytorch as spconv; "
                 "f = torch.randn(8, 4, device='cuda'); i = torch.randint(0, 8, (8, 4), device='cuda', dtype=torch.int32); i[:, 0] = 0; "
                 "t = spconv.SparseConvTensor(f, i, [8, 8, 8], 1); spconv.SubMConv3d(4, 4, 3).cuda()(t); "
                 "from pytorch3d import _C as p3d; import nvdiffrast.torch as dr; dr.RasterizeCudaContext(); import kaolin",
    "v2d_foundation_pose": "from pytorch3d import _C as p3d; import nvdiffrast.torch as dr; dr.RasterizeCudaContext(); import kaolin; "
                           "import tensorrt as trt; print('tensorrt', trt.__version__); import warp as wp; wp.init()",
    "v2d_droid_slam": "import torch_scatter, lietorch, droid_backends; print(lietorch.SE3.Identity(1, device='cuda'))",
    "v2d_gsplat_refinement": "import gsplat; from gsplat.rendering import rasterization; "
                             "m = torch.zeros(1, 3, device='cuda'); q = torch.tensor([[1., 0, 0, 0]], device='cuda'); "
                             "s = torch.full((1, 3), .1, device='cuda'); o = torch.ones(1, device='cuda'); c = torch.ones(1, 3, device='cuda'); "
                             "v = torch.eye(4, device='cuda')[None]; v[0, 2, 3] = 2; K = torch.tensor([[[50., 0, 32], [0, 50, 32], [0, 0, 1]]], device='cuda'); "
                             "img, _, _ = rasterization(m, q, s, o, c, v, K, 64, 64); print('gsplat render', tuple(img.shape))",
    "v2d_grounding_dino": "import groundingdino; from groundingdino import _C; print('groundingdino _C ok')",
    "v2d_sam2": "import sam2; print('sam2 ok')",
    "v2d_hamer": "import hamer; print('hamer ok')",
    "v2d_wilor": "import wilor_mini; print('wilor ok')",
    "v2d_moge": "import moge; print('moge ok')",
    "v2d_anycalib": "import anycalib; print('anycalib ok')",
    "v2d_geocalib": "import geocalib; print('geocalib ok')",
    "v2d_task_library_loader": "from pytorch3d import _C as p3d; print('pytorch3d ok')",
}


@app.function(image=IMAGES["control"], timeout=3600, retries=0, cpu=1, memory=1024)
def gpu_smoke_driver(images: list[str]) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    def one(image: str) -> dict:
        spec = {"image": image, "argv": ["python", "-c", GPU_SMOKE_COMMON + GPU_SMOKE[image]], "env": {},
                "workdir": None, "mounts": [], "call_id": f"gpusmoke-{uuid.uuid4().hex[:6]}", "exit_after": True}
        try:
            result = IMAGE_FUNCTIONS[image].remote(spec)
            log = ""
            try:
                with open(result["log"], encoding="utf-8", errors="replace") as stream:
                    log = stream.read()[-1500:]
            except OSError:
                pass
            return {"image": image, "returncode": result["returncode"], "tail": log}
        except Exception as exc:  # noqa: BLE001
            return {"image": image, "returncode": -1, "tail": f"{type(exc).__name__}: {exc}"[:1500]}

    with ThreadPoolExecutor(max_workers=4) as pool:  # stay well below the workspace GPU cap
        return list(pool.map(one, images))


@app.local_entrypoint()
def gpu_smoke(images: str = "") -> None:
    """Per image: load its compiled CUDA extensions on the module GPU (`--images a,b`)."""
    names = [n for n in images.split(",") if n] or list(GPU_SMOKE)
    for row in gpu_smoke_driver.remote(names):
        print(json.dumps({"image": row["image"], "returncode": row["returncode"]}), flush=True)
        if row["returncode"] != 0:
            print(row["tail"], flush=True)
