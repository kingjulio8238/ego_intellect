"""Modal image definitions, derived from the repository's own Dockerfiles.

* Reconstruction module images: built from each module's unmodified Dockerfile with the
  same build context (`reconstruction/modules`) that `docker/build.py` uses, so they always
  contain the current checkout (which is why `--dev` mounts are unnecessary on Modal).
* Loader image: same, plus the robotic_grounding python package that its build.py
  stages into the context (added here as an image layer, so nothing is written into the
  checkout).
* robotic-grounding image: the Isaac Lab base image with the steps of
  `robotic_grounding/workflow/Dockerfile` replayed on top, plus an Isaac-GR00T checkout.
  The Dockerfile's `/usr/local/bin/python` wrapper is relocated to C.ISAAC_BIN because
  Modal's own runtime Python must own `/usr/local/bin/python` (`add_python`), the same
  way Modal's Isaac Lab example builds its image.
* control image: small CPU image for the host-side launcher and pipeline orchestration.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import modal

from tools.modal_v2d import config as C

# Pin Modal's image builder. Builders before 2025.06 silently skip Dockerfile WORKDIR
# ("Skipping WORKDIR instruction"), which runs every later RUN step and the container in
# the wrong directory. Pinning here (read lazily by Modal at build time) also keeps image
# hashes identical across machines whatever the workspace default is.
IMAGE_BUILDER_VERSION = "2025.06"
_requested = os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", IMAGE_BUILDER_VERSION)
if _requested != IMAGE_BUILDER_VERSION:
    raise RuntimeError(
        f"MODAL_IMAGE_BUILDER_VERSION={_requested}; these images need {IMAGE_BUILDER_VERSION} (Dockerfile WORKDIR support)"
    )

REPO = Path(__file__).resolve().parents[2]
MODULES = REPO / "reconstruction" / "modules"
RG_ROOT = REPO / "robotic_grounding"
ISAAC_LAB_BASE = os.environ.get("V2D_ISAAC_LAB_IMAGE", "nvcr.io/nvidia/isaac-lab:2.3.2")
GROOT_REPO = os.environ.get("V2D_GROOT_REPO", "https://github.com/NVIDIA/Isaac-GR00T.git")
# Isaac-GR00T n1.7-release tag commit (verified to contain every gr00t module the pipeline calls).
GROOT_REF = os.environ.get("V2D_GROOT_REF", "23ace64f17aa5015259b8609d371eb61a357c776")

# What gets uploaded as the repo for control/rg containers (and the shim/runtime code).
REPO_IGNORE = [
    ".git",
    ".e2e",
    "docs",
    "video_ingestion_agent",
    "robotic_grounding/flash_chord",
    "reconstruction/data",
    "**/__pycache__",
    "**/*.pyc",
    "**/*.egg-info",
    "**/.venv",
    "**/node_modules",
    "**/_rg_pkg",
    "**/.pytest_cache",
    "**/htmlcov",
    # This package's own docs and tests: editing them must not rebuild the baked layers.
    "tools/modal_v2d/README.md",
    "tools/modal_v2d/tests",
    "tools/fastrecon",  # local benchmarking harness; never needed inside a container
]
# Same exclusions as build.py's shutil.ignore_patterns (which match at any depth).
LOADER_RG_PKG_IGNORE = ["**/assets", "**/__pycache__", "**/*.pyc", "**/*.egg-info"]
# Record the image's own PYTHONPATH and WORKDIR: shimmed commands run with these, not with
# the Modal runtime's environment (see worker.py / docker_shim.py).
CAPTURE_IMAGE_ENV = (
    f"sh -c 'pwd > {C.IMAGE_WORKDIR_FILE}; printf %s \"${{PYTHONPATH:-}}\" > {C.IMAGE_PYTHONPATH_FILE}'"
)

# Host-side wrapper dependencies (see reconstruction/scripts/install_ego_reconstruction_packages.sh
# and each wrapper's pyproject). Installed with deps here; the wrappers themselves are
# installed --no-deps at container start from the uploaded checkout.
HOST_PYTHON_DEPS = [
    "numpy",
    "Pillow",
    "scipy>=1.13,<2",
    "opencv-python",
    "trimesh",
    "OpenEXR",
    "boto3",
    "h5py",
    "av",
    "imageio",
    "imageio-ffmpeg",
    "tqdm",
    "setuptools",
    "wheel",
]


def _with_repo(image: modal.Image, role: str) -> modal.Image:
    """Add the checkout and bake the per-invocation bootstrap into the image (bake.py).

    Content-addressed: an unchanged checkout reuses the cached layers; a change rebuilds
    only these last two layers (copy, git index, editable installs: about a minute on a
    CPU builder) instead of costing ~80 s at the start of every invocation.
    """
    return image.add_local_dir(REPO, C.REPO_UPLOAD, ignore=REPO_IGNORE, copy=True).run_commands(
        f"cd {C.REPO_UPLOAD} && /usr/local/bin/python -m tools.modal_v2d.bake {role}"
    )


# Upstream drift fix (not Modal-specific; a fresh `docker build` hits it too). The module
# Dockerfiles leave huggingface_hub unpinned; 2.x downloads through httpx2, whose brotli
# decoder passes `output_buffer_limit=`, which Brotli < 1.2 rejects:
#   TypeError: process() takes no keyword arguments      (setup, SAM2 weights, 2026-11-17)
# Upgrade Brotli only where it is installed and older than 1.2; touch nothing else.
BROTLI_COMPAT = (
    'PY="$(command -v python || command -v python3)"; '
    '"$PY" -c "import importlib.metadata as m, subprocess, sys; '
    "v = next((m.version(n) for n in ('brotli', 'Brotli') if __import__('importlib.util').util.find_spec('brotli')), None); "
    "old = v is not None and tuple(int(x) for x in v.split('.')[:2]) < (1, 2); "
    "print('v2d compat: Brotli', v, '-> >=1.2' if old else '(ok)'); "
    "old and subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'brotli>=1.2'])\""
)


# Upstream drift fix: the SAM3D Dockerfile installs open3d unpinned; current open3d wheels
# link libusb-1.0, which that image lacks:
#   ImportError: libusb-1.0.so.0: cannot open shared object file  (reconstruct, SAM3D, 2026-11-17)
# Only where importing open3d fails on libusb: install it, then require that error to be
# gone (other import errors, e.g. libEGL, which the GPU runtime provides, are left alone).
OPEN3D_COMPAT = (
    'PY="$(command -v python || command -v python3)"; '
    'err="$("$PY" -c "import open3d" 2>&1 >/dev/null)"; '
    'case "$err" in '
    '*libusb-1.0.so*) apt-get update && apt-get install -y --no-install-recommends libusb-1.0-0 '
    '&& rm -rf /var/lib/apt/lists/* && ! "$PY" -c "import open3d" 2>&1 | grep -q libusb-1.0.so '
    '&& echo "v2d compat: installed libusb-1.0-0 for open3d" ;; '
    '*) echo "v2d compat: open3d needs no libusb fix" ;; '
    'esac'
)


# SAM3D renders through gsplat, which JIT-compiles its CUDA extension on first use: 65-75 s
# inside every SAM3D call (fastrecon P-sam3d profile; the per-call volume cache never hit,
# since ninja sees image-layer sources newer than the cached objects). Compile once at image
# build, for the GPU the module functions run on, and point torch at it. Same sources and
# flags as the runtime build, so the kernels are the same.
GSPLAT_JIT_IMAGES = {"v2d_sam3d"}
GSPLAT_PREBUILD_ENV = {"TORCH_EXTENSIONS_DIR": "/opt/torch_extensions", "TORCH_CUDA_ARCH_LIST": "8.9"}
GSPLAT_PREBUILD = (
    "mkdir -p /opt/torch_extensions && cd /tmp && "
    "TORCH_EXTENSIONS_DIR=/opt/torch_extensions TORCH_CUDA_ARCH_LIST=8.9 "
    "python -c 'from gsplat.cuda._backend import _C; assert _C is not None; print(\"gsplat_cuda prebuilt\")'"
)


def _dockerfile_instructions(path: Path) -> list[str]:
    """Dockerfile -> instructions (comments dropped, `\\` continuations joined verbatim)."""
    out: list[str] = []
    current = ""
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        if not current and (not line.strip() or line.lstrip().startswith("#")):
            continue
        if current and line.lstrip().startswith("#"):
            continue  # comment lines inside a continued RUN
        current = f"{current}\n{line}" if current else line
        if not line.endswith("\\"):
            out.append(current)
            current = ""
    if current:
        out.append(current)
    return out


SPLIT_DOCKERFILE_DIR = REPO / ".e2e" / "modal_dockerfiles"


def _reused_images() -> dict[str, str]:
    """V2D_MODAL_REUSE_IMAGES="v2d_foundation_pose=im-...,..." (see module_image)."""
    pairs = (item.split("=", 1) for item in os.environ.get("V2D_MODAL_REUSE_IMAGES", "").split(",") if "=" in item)
    return {name.strip(): image_id.strip() for name, image_id in pairs}


def module_image(image_name: str) -> modal.Image:
    """Module image = the module's own Dockerfile, built as two parts.

    Everything before the first COPY/ADD (base image, apt, pip, CUDA builds) needs no build
    context, so it is built from a generated Dockerfile with an empty context and stays
    cached across source edits. The COPY of the module sources and the steps after it
    (editable installs, import checks) become one Modal layer per instruction. Before this
    split, `from_dockerfile(context_dir=modules/)` re-ran the whole Dockerfile on any edit
    under reconstruction/modules: 10-85 min per image (fastrecon round 2).
    """
    module_dir = MODULES / C.MODULE_IMAGES[image_name]
    dockerfile = module_dir / "docker" / "Dockerfile"
    if not dockerfile.exists():
        # Inside a running container app.py is re-imported, but images are never built
        # there and the module sources are absent: return the lazy definition unread.
        return modal.Image.from_dockerfile(dockerfile, context_dir=MODULES)
    instructions = _dockerfile_instructions(dockerfile)
    first_copy = next(
        (k for k, ins in enumerate(instructions) if ins.split(None, 1)[0].upper() in {"COPY", "ADD"}),
        len(instructions),
    )
    reuse = _reused_images().get(image_name)
    if reuse:
        # Escape hatch while a base layer is still building: start from an earlier complete
        # image of this module (same base steps) and re-apply only the source layers.
        image = modal.Image.from_id(reuse)
        for instruction in instructions[first_copy:]:
            image = image.dockerfile_commands([instruction], context_dir=MODULES)
        return image.entrypoint([]).run_commands(CAPTURE_IMAGE_ENV)
    pre_dir = SPLIT_DOCKERFILE_DIR / image_name
    (pre_dir / "context").mkdir(parents=True, exist_ok=True)
    pre_file = pre_dir / "Dockerfile"
    pre_text = "\n".join(instructions[:first_copy]) + "\n"
    if not pre_file.exists() or pre_file.read_text() != pre_text:
        pre_file.write_text(pre_text)
    image = modal.Image.from_dockerfile(pre_file, context_dir=pre_dir / "context").entrypoint([])
    # Source-independent fixes go before the sources, so they stay cached too.
    image = image.run_commands(BROTLI_COMPAT, OPEN3D_COMPAT)
    if image_name in GSPLAT_JIT_IMAGES:
        image = image.run_commands(GSPLAT_PREBUILD, gpu=C.MODULE_GPU).env(GSPLAT_PREBUILD_ENV)
    for instruction in instructions[first_copy:]:
        # Same build context as docker/build.py; Modal honours modules/.dockerignore.
        image = image.dockerfile_commands([instruction], context_dir=MODULES)
    image = image.entrypoint([])
    if image_name == "v2d_task_library_loader":
        # build.py stages robotic_grounding's python package (no assets) into the context;
        # the Dockerfile's ENV PYTHONPATH already points at this location.
        image = image.add_local_dir(
            RG_ROOT / "source" / "robotic_grounding" / "robotic_grounding",
            "/workspace/v2d_task_library_loader/_rg_pkg/robotic_grounding",
            copy=True,
            ignore=LOADER_RG_PKG_IGNORE,
        )
    # No `--dev` upload: the image was just built from this checkout (see C.DEV_MOUNT).
    return image.run_commands(CAPTURE_IMAGE_ENV)


# Upstream drift fix: workflow/Dockerfile installs `datasets` (pulls the newest pyarrow,
# 26.x, which requires NumPy >= 2) and then pins numpy<2 for Isaac Lab, leaving pyarrow
# unimportable: "ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4"
# (retarget.vega, 2026-11-17). 25.0.0 is the newest release that imports with NumPy 1.26;
# verified with every retarget-stage script's --help in this image. Applied only when the
# installed pyarrow fails to import, and the import is required to succeed afterwards.
ISAAC_KIT_PYTHON = "/workspace/isaaclab/_isaac_sim/kit/python/bin/python3"
ISAAC_PYARROW_COMPAT = (
    f'if {ISAAC_KIT_PYTHON} -c "import pyarrow" >/dev/null 2>&1; then echo "v2d compat: pyarrow imports (ok)"; '
    f'else {ISAAC_KIT_PYTHON} -m pip install --no-cache-dir --no-deps "pyarrow==25.0.0" '
    f'&& {ISAAC_KIT_PYTHON} -c "import pyarrow, pyarrow.parquet; print(\'v2d compat: pyarrow\', pyarrow.__version__)"; fi'
)

GENERATED_RG_DOCKERFILE = Path(tempfile.gettempdir()) / "v2d_modal" / "robotic_grounding.Dockerfile"


def render_robotic_grounding_dockerfile(source: str) -> str:
    """Rewrite robotic_grounding/workflow/Dockerfile for Modal.

    Only two edits: the single-stage FROM is pinned/validated, and the Isaac Lab `python`
    wrapper moves from /usr/local/bin to C.ISAAC_BIN. Fails loudly if the Dockerfile
    changes shape, instead of silently building a different image than
    `workflow/run.sh build` would.
    """
    lines = source.splitlines()
    from_lines = [line for line in lines if line.strip().upper().startswith("FROM ")]
    if len(from_lines) != 1 or ISAAC_LAB_BASE.split("/")[-1] not in from_lines[0]:
        raise RuntimeError(
            f"robotic_grounding/workflow/Dockerfile base changed ({from_lines}); "
            f"update V2D_ISAAC_LAB_IMAGE (currently {ISAAC_LAB_BASE})"
        )
    kept = [line for line in lines if not line.strip().upper().startswith(("FROM ", "CMD "))]
    body = "\n".join(kept)
    wrapper = "/usr/local/bin/python"
    count = len(re.findall(re.escape(wrapper) + r"\b", body))
    if count != 3:
        raise RuntimeError(
            f"expected the python wrapper to be written to {wrapper} three times in "
            f"workflow/Dockerfile, found {count}; update tools/modal_v2d/images.py"
        )
    body = re.sub(re.escape(wrapper) + r"\b", f"{C.ISAAC_BIN}/python", body)
    header = [
        f"FROM {ISAAC_LAB_BASE}",
        "ENV ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES",
        # Isaac Sim renders through Vulkan; request every driver capability.
        "ENV NVIDIA_DRIVER_CAPABILITIES=all",
        f"RUN mkdir -p {C.ISAAC_BIN}",
    ]
    return "\n".join([*header, body, ""])


def robotic_grounding_image() -> modal.Image:
    if modal.is_local():  # inside containers the image is already built; nothing to render
        GENERATED_RG_DOCKERFILE.parent.mkdir(parents=True, exist_ok=True)
        GENERATED_RG_DOCKERFILE.write_text(
            render_robotic_grounding_dockerfile((RG_ROOT / "workflow" / "Dockerfile").read_text())
        )
    return (
        # add_python installs Modal's runtime Python at /usr/local (the wrapper moved away).
        modal.Image.from_dockerfile(GENERATED_RG_DOCKERFILE, context_dir=RG_ROOT, add_python="3.11")
        .entrypoint([])
        .run_commands(
            "apt-get update && apt-get install -y --no-install-recommends git git-lfs curl vulkan-tools "
            "&& rm -rf /var/lib/apt/lists/*",
            "curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh",
            f"git clone {GROOT_REPO} {C.GROOT_DIR} && cd {C.GROOT_DIR} && git checkout {GROOT_REF} "
            "&& git submodule update --init --recursive",
            # Python 3.10 environment managed by uv inside the checkout (`uv run python` from
            # its root is what tools/e2e expects).
            f"cd {C.GROOT_DIR} && UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_CACHE_DIR=/tmp/uv-cache "
            "uv sync --python 3.10 && rm -rf /tmp/uv-cache",
            f"cd {C.GROOT_DIR} && uv run --no-sync python -c 'import gr00t; print(\"gr00t import OK\")'",
        )
        # GR00T's trainer imports DeepSpeed (via accelerate), which refuses to import without
        # a CUDA toolkit: "MissingCUDAException: CUDA_HOME does not exist" (finetune, Vast,
        # 2026-11-17). The Isaac Lab base image ships no toolkit; add nvcc 12.8 (torch cu128).
        .run_commands(
            '. /etc/os-release && cd /tmp && wget -q "https://developer.download.nvidia.com/compute/cuda/repos/'
            'ubuntu$(echo $VERSION_ID | tr -d .)/x86_64/cuda-keyring_1.1-1_all.deb" && dpkg -i cuda-keyring_1.1-1_all.deb '
            "&& apt-get update && apt-get install -y --no-install-recommends cuda-nvcc-12-8 && rm -rf /var/lib/apt/lists/*",
            f"cd {C.GROOT_DIR} && CUDA_HOME=/usr/local/cuda-12.8 uv run --no-sync python -c 'import deepspeed; print(\"deepspeed OK\")'",
        )
        # The repo's closed-loop client speaks GR00T main's msgpack_numpy wire format; the
        # n1.7-release server (pinned for checkpoint compatibility) does not. Swap only its
        # MsgSerializer for main's, which also decodes the old format.
        .add_local_file(str(Path(__file__).parent / "patches" / "groot_serializer.py"), "/opt/v2d/groot_serializer.py", copy=True)
        .run_commands(
            f"cd {C.GROOT_DIR} && uv pip install --python .venv/bin/python msgpack-numpy "
            f"&& .venv/bin/python /opt/v2d/groot_serializer.py {C.GROOT_DIR}",
        )
        .env({"UV_NO_SYNC": "1", "UV_PYTHON_INSTALL_DIR": "/opt/uv-python", "CUDA_HOME": "/usr/local/cuda-12.8"})
        .pip_install(*HOST_PYTHON_DEPS)
        .run_commands(
            f"mkdir -p {C.SHIM_BIN} {C.LOCAL_STATE_ROOT}",
            # `docker exec <rg> bash -lc "... python ..."` (run_eval.sh) must still resolve
            # `python` to the Isaac Lab wrapper if a login profile rewrites PATH.
            f"echo 'export PATH={C.ISAAC_BIN}:$PATH' > /etc/profile.d/zz-v2d-isaac-python.sh",
            ISAAC_PYARROW_COMPAT,
            CAPTURE_IMAGE_ENV,
        )
    )


def control_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "wget", "bash")
        .pip_install(*HOST_PYTHON_DEPS)
        # Placeholder so `setup --isaac-groot-dir` validates here; the real checkout lives
        # at the same path in the robotic-grounding image, where it is used.
        .run_commands(
            f"mkdir -p {C.GROOT_DIR} {C.SHIM_BIN} {C.LOCAL_STATE_ROOT}",
            f"echo 'placeholder: Isaac-GR00T lives in the robotic-grounding image' > {C.GROOT_DIR}/PLACEHOLDER",
            CAPTURE_IMAGE_ENV,
        )
    )


def build_images() -> dict[str, modal.Image]:
    # tools/modal_v2d itself reaches every container through Modal's entrypoint package
    # mount (`modal run -m tools.modal_v2d.app` mounts the top-level `tools` package).
    images = {name: module_image(name) for name in C.MODULE_IMAGES}
    images[C.RG_IMAGE] = _with_repo(robotic_grounding_image(), "rg")
    images["control"] = _with_repo(control_image(), "control")
    return images
