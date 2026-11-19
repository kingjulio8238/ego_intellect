"""Image-build step: prepare the repo copy once, instead of in every container.

    python -m tools.modal_v2d.bake <control|rg>      (run by images._with_repo)

Bakes into the image what runtime.bootstrap would otherwise redo on every invocation:
the writable repo copy with content-derived mtimes, the synthetic git index tools/e2e
fingerprints, and the editable host-wrapper installs. Measured on the first runs this was
~80 s per invocation (repo copy 4 s, pip 27 s, git snapshot 48 s), paid on an L40S in the
rg role. Stdlib only.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tools.modal_v2d import runtime


def main(role: str) -> None:
    if role not in ("control", "rg"):
        raise SystemExit(f"unknown role {role!r}")
    runtime._copy_repo()
    runtime._synthetic_git("image build")
    runtime._install_host_packages(role)
    Path(runtime.BAKED_MARKER).write_text(role + "\n")
    runtime.log(f"baked repo copy, git index and {role} host packages into the image")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
