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
# build and point torch at it. Same sources and flags as the runtime build, so the kernels
# are the same. Built for every architecture the module images target (A100 8.0, A10G 8.6,
# L40S 8.9, H100 9.0), so hedged calls can run on another GPU type (batch mode).
GSPLAT_JIT_IMAGES = {"v2d_sam3d"}
GSPLAT_ARCHS = "12.0" if C.ARCH == "blackwell" else "8.0;8.6;8.9;9.0"
GSPLAT_PREBUILD_ENV = {"TORCH_EXTENSIONS_DIR": "/opt/torch_extensions", "TORCH_CUDA_ARCH_LIST": GSPLAT_ARCHS}
GSPLAT_PREBUILD = (
    "mkdir -p /opt/torch_extensions && cd /tmp && "
    f"TORCH_EXTENSIONS_DIR=/opt/torch_extensions TORCH_CUDA_ARCH_LIST='{GSPLAT_ARCHS}' "
    "python -c 'from gsplat.cuda._backend import _C; assert _C is not None; print(\"gsplat_cuda prebuilt\")'"
)


# Blackwell (sm 12.0) needs CUDA >= 12.8 and libraries built for it. The module Dockerfiles
# target PyTorch 2.5.1 + CUDA 12.4 (no sm 12.0 kernels); this variant swaps the version pins
# for their nearest Blackwell-capable equivalents and compiles CUDA code for 12.0 only.
# Verified wheel indexes (2026-11-20): kaolin 0.18.0 torch-2.7.1_cu128, torch-scatter
# pt27cu128; TensorRT 10.9.0.34 (+cuda12.8) is the first 10.x with Blackwell in our apt repo.
BLACKWELL_SUBS = [
    (r"pytorch/pytorch:2\.5\.1-cuda12\.4-cudnn9-devel", "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel"),
    (r'TORCH_CUDA_ARCH_LIST=("?)[0-9.; ]+\1', 'TORCH_CUDA_ARCH_LIST="12.0"'),
    (r"torch-2\.5\.1_cu124\.html", "torch-2.7.1_cu128.html"),
    (r"torch-2\.5\.1\+cu124\.html", "torch-2.7.1+cu128.html"),
    (r"cumm-cu124", "cumm-cu126"),
    (r"spconv-cu124", "spconv-cu126"),
    (r"10\.7\.0\.23-1\+cuda12\.6", "10.9.0.34-1+cuda12.8"),
    (r"tensorrt-cu12==10\.7\.0\.post1", "tensorrt-cu12==10.9.0.34"),
    # GroundingDINO's deformable-attention op dispatches on `Tensor.type()`, removed from
    # the dispatch macros in PyTorch 2.7 (build error "no suitable conversion function from
    # DeprecatedTypeProperties to c10::ScalarType"); `scalar_type()` is the same type.
    (r"cd /tmp/GroundingDINO && ",
     lambda m: "cd /tmp/GroundingDINO && sed -i 's/value[.]type(),/value.scalar_type(),/g' "
               "groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu && "),
    # Same removed dispatch form in FoundationPose's own CUDA ops (bundlesdf/mycuda/common.cu).
    (r"cd /workspace/v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda && ",
     lambda m: "cd /workspace/v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda && "
               "sed -i 's/[.]type(), \"/.scalar_type(), \"/g' *.cu && "),
    # sam-3d-objects imports flash_attn only on A100/H100/H200; other GPUs use sdpa.
    (r"MAX_JOBS=4 pip install flash-attn==2\.7\.4\.post1 --no-build-isolation",
     "echo 'flash-attn skipped: sdpa attention on Blackwell RTX'"),
    # WiLoR-mini pins torch<=2.5 and pip silently swapped in torch 2.5.0 (no sm 12.0 kernels:
    # "no kernel image is available"). Same package, its other requirements, the image's torch.
    (r"pip install --no-cache-dir git\+https://github\.com/warmshao/WiLoR-mini\.git$",
     "pip install --no-cache-dir --no-deps git+https://github.com/warmshao/WiLoR-mini.git && "
     "pip install --no-cache-dir smplx==0.1.28 timm einops ultralytics==8.1.34 opencv-python "
     "huggingface_hub scikit-image roma"),
]


def blackwell_instruction(instruction: str) -> str:
    import re  # noqa: PLC0415

    for pattern, replacement in BLACKWELL_SUBS:
        instruction = re.sub(pattern, replacement, instruction)
    return instruction


def _dockerfile_instructions(path: Path, *, arch_rewrite: bool = True) -> list[str]:
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
    if C.ARCH == "blackwell" and arch_rewrite:
        out = [blackwell_instruction(instruction) for instruction in out]
        if any("cmake " in instruction for instruction in out):
            # The PyTorch 2.7.1 image ships CMake 4, which rejects projects declaring
            # cmake_minimum_required < 3.5 (pybind11 v2.10.0, FoundationPose's mycpp).
            out.insert(1, "ENV CMAKE_POLICY_VERSION_MINIMUM=3.5")
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
    pre_dir = SPLIT_DOCKERFILE_DIR / (image_name if C.ARCH == "ada" else f"{image_name}-{C.ARCH}")
    (pre_dir / "context").mkdir(parents=True, exist_ok=True)
    pre_file = pre_dir / "Dockerfile"
    pre_text = "\n".join(instructions[:first_copy]) + "\n"
    if not pre_file.exists() or pre_file.read_text() != pre_text:
        pre_file.write_text(pre_text)
    image = modal.Image.from_dockerfile(pre_file, context_dir=pre_dir / "context").entrypoint([])
    # Source-independent fixes go before the sources, so they stay cached too.
    image = image.run_commands(BROTLI_COMPAT, OPEN3D_COMPAT)
    if image_name in GSPLAT_JIT_IMAGES:
        image = image.run_commands(GSPLAT_PREBUILD, gpu="L40S").env(GSPLAT_PREBUILD_ENV)  # fixed: kernels target GSPLAT_ARCHS whatever the builder GPU
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
    if C.ARCH == "blackwell":
        # PyTorch >= 2.6 defaults torch.load to weights_only=True, which rejects the pickled
        # checkpoints these modules load (WiLoR: "Weights only load failed"). The Ada images'
        # PyTorch 2.5.1 loads them fully; restore that behaviour.
        image = image.env({"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"})
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
    """robotic_grounding/workflow/Dockerfile plus GR00T, built in three parts.

    1. Everything in the Dockerfile before its first COPY (Isaac Lab base, apt, pip) from a
       generated Dockerfile with an empty context.
    2. The source-independent additions (GR00T checkout and uv env, nvcc, serializer
       patch, host packages).
    3. The Dockerfile's COPY of the robotic_grounding sources and the steps after it
       (editable install, python wrapper), one Modal layer per instruction.
    Parts 1-2 stay cached across source edits; an edit under robotic_grounding/ rebuilds
    only part 3 (about a minute). Before this split the whole Dockerfile was one layer with
    robotic_grounding/ as its context, and GR00T's layers sat after it: every edit,
    even a stray __pycache__, rebuilt everything (~18 min, render overlay phase C).
    """
    if not modal.is_local():  # inside containers the image is already built; nothing to render
        return modal.Image.from_dockerfile(GENERATED_RG_DOCKERFILE, context_dir=RG_ROOT, add_python="3.11")
    GENERATED_RG_DOCKERFILE.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_RG_DOCKERFILE.write_text(
        render_robotic_grounding_dockerfile((RG_ROOT / "workflow" / "Dockerfile").read_text())
    )
    instructions = _dockerfile_instructions(GENERATED_RG_DOCKERFILE, arch_rewrite=False)
    first_copy = next(
        (k for k, ins in enumerate(instructions) if ins.split(None, 1)[0].upper() in {"COPY", "ADD"}),
        len(instructions),
    )
    pre_dir = SPLIT_DOCKERFILE_DIR / C.RG_IMAGE
    (pre_dir / "context").mkdir(parents=True, exist_ok=True)
    pre_file = pre_dir / "Dockerfile"
    pre_text = "\n".join(instructions[:first_copy]) + "\n"
    if not pre_file.exists() or pre_file.read_text() != pre_text:
        pre_file.write_text(pre_text)
    image = (
        # add_python installs Modal's runtime Python at /usr/local (the wrapper moved away).
        modal.Image.from_dockerfile(pre_file, context_dir=pre_dir / "context", add_python="3.11")
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
        )
    )
    for instruction in instructions[first_copy:]:
        # Same build context as `workflow/run.sh build`; Modal honours robotic_grounding/.dockerignore.
        image = image.dockerfile_commands([instruction], context_dir=RG_ROOT)
    # The editable install can move pyarrow/numpy, so the pyarrow check runs after it.
    image = image.entrypoint([]).run_commands(ISAAC_PYARROW_COMPAT, CAPTURE_IMAGE_ENV)
    # Isaac warm-up baked into the image: the first Kit launch in a fresh container fetches
    # and unpacks extensions (URDF importer) into ~/.local/share/ov, compiles every
    # extension's .pyc and fills Kit's shader/derived-data caches: ~24 s that every clip's
    # `simulate` paid (59 s cold vs 35 s warm, simulate phase 2026-11-20). One short replay
    # of the bundled sample clip on a GPU builder leaves those caches in the image; the run's
    # own outputs (/tmp) are dropped. Reruns with part 3, i.e. on robotic_grounding edits.
    return image.run_commands(RG_ISAAC_WARMUP, gpu=C.RG_GPU)


# One 20-step physics replay of the bundled tissue_box_simple sample (the `simulate` command
# line, CPU physics as there); fails the build if the replay does not finish.
RG_ISAAC_WARMUP = (
    "cd /workspace/video_to_data/robotic_grounding "
    "&& export ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES "
    "&& /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/replay_metrics.py --headless --device cpu "
    "--task VegaSharpa-WholeBody-Manip-v0 "
    "--motion_file ego_recon/processed/sequence_id=tissue_box_simple/robot_name=vega_sharpa "
    "--output /tmp/v2d_warm/replay_metrics.json --states_out none --max_steps 20 > /tmp/v2d_warm.log 2>&1; "
    "grep -q REPLAY_METRICS /tmp/v2d_warm.log || { tail -40 /tmp/v2d_warm.log; exit 1; }; "
    "rm -rf /tmp/v2d_warm /tmp/v2d_warm.log /tmp/IsaacLab /tmp/isaaclab && echo 'isaac warm-up OK'"
)


# Triage gate (tools/gates/triage.py): MediaPipe hand landmarker on CPU, model baked in.
# Environment packages (tools/envpkg): parquet, schema validation, the MuJoCo export and replay.
GATE_PYTHON_DEPS = ["mediapipe==1.2.3", "pyarrow", "jsonschema", "mujoco==3.8.0"]
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def control_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "wget", "bash")
        .pip_install(*HOST_PYTHON_DEPS)
        # MediaPipe's task runtime dlopens EGL/GLES even on CPU (missing: "libEGL.so.1: cannot open").
        .apt_install("libegl1", "libgles2")
        .pip_install(*GATE_PYTHON_DEPS)
        .run_commands(
            "mkdir -p /opt/mediapipe_tasks",
            f"wget -q -O /opt/mediapipe_tasks/hand_landmarker.task {HAND_LANDMARKER_URL}",
            f"echo '{HAND_LANDMARKER_SHA256}  /opt/mediapipe_tasks/hand_landmarker.task' | sha256sum -c -",
            # A broken install must fail the build, not the first clip's triage: build the
            # detector and run it once, the way tools/gates/triage.py does.
            "python -c \"import numpy as np, mediapipe as mp, pyarrow, jsonschema, mujoco; "
            "from mediapipe.tasks.python import BaseOptions; "
            "from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode; "
            "d = HandLandmarker.create_from_options(HandLandmarkerOptions(base_options=BaseOptions("
            "model_asset_path='/opt/mediapipe_tasks/hand_landmarker.task'), running_mode=RunningMode.IMAGE)); "
            "r = d.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((240, 320, 3), np.uint8))); "
            "d.close(); print('gates deps OK', mp.__version__, len(r.hand_landmarks), 'hands on a blank frame')\"",
        )
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
