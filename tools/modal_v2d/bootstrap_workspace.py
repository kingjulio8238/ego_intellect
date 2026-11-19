"""Prepare a fresh Modal workspace for batch runs by copying a prepared one.

    python -m tools.modal_v2d.bootstrap_workspace --src sales-32662 --dst julian-41138 \
        [--steps secret,weights,verify,build,probe,inputs] [--cap 25]

Steps (each idempotent; run any subset):
  secret   create `v2d-hf` on dst from $HF_TOKEN or ~/.cache/huggingface/token (never printed)
  weights  copy v2d-weights recon/ cache/ mano/ from src to dst: a CPU sandbox on src mounts
           the source volume and uploads as dst (dst token from ~/.modal.toml); cap 1 h
  verify   md5 every weights file on both workspaces; any difference fails
  build    build all images on dst (`run_e2e_modal.sh --build`)
  probe    one no-op call per module image on dst, so each image is loaded once (first
           loads on a new workspace take minutes) and its cold start is measured
  inputs   upload batch inputs (task profiles) from --inputs DIR to fs/e2e/inputs/

Spend: dst month-to-date is checked first (profile pinned; a failed query stops the run,
never read as $0); src pays only for the copy and md5 sandboxes (cents).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WEIGHT_DIRS = ("recon", "cache", "mano")


def run(cmd: list[str], profile: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env={**os.environ, "MODAL_PROFILE": profile}, cwd=REPO, text=True, **kw)


def month_to_date(profile: str) -> float:
    result = run(["modal", "billing", "report", "--for", "this month", "--json"], profile, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f"{profile}: billing query failed (spend UNKNOWN, not $0):\n{result.stderr.strip()}")
    return sum(float(row.get("cost", row.get("Cost"))) for row in json.loads(result.stdout))


def step_secret(dst: str) -> None:
    token = os.environ.get("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").read_text().strip()
    listing = run(["modal", "secret", "list", "--json"], dst, capture_output=True)
    if listing.returncode == 0 and '"v2d-hf"' in listing.stdout:
        print(f"{dst}: secret v2d-hf exists")
        return
    result = run(["modal", "secret", "create", "v2d-hf", f"HF_TOKEN={token}"], dst, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f"{dst}: could not create v2d-hf: {result.stderr.strip()[-300:]}")
    print(f"{dst}: secret v2d-hf created")


COPY_SCRIPT = r'''
import time, modal
t = time.time()
vol = modal.Volume.from_name("v2d-weights", create_if_missing=True)
with vol.batch_upload(force=True) as batch:
    for name in DIRS:
        batch.put_directory(f"/src/{name}", f"/{name}")
print("UPLOAD_DONE in %.0fs" % (time.time() - t), flush=True)
'''


def _internal(step: str, profile: str, *args: str) -> str:
    """Run a step that uses the Modal client in-process, in a child pinned to `profile`
    (the client reads its profile once, at import)."""
    result = run([sys.executable, "-m", "tools.modal_v2d.bootstrap_workspace", "--internal", step, *args], profile,
                 stdout=subprocess.PIPE)
    if result.returncode != 0:
        raise SystemExit(f"{step} on {profile} failed (exit {result.returncode})")
    return result.stdout


def step_weights(src: str, dst: str) -> None:
    print(_internal("copy", src, dst), end="")


def _copy(dst: str) -> None:
    import modal

    config = tomllib.loads((Path.home() / ".modal.toml").read_text())
    if dst not in config:
        raise SystemExit(f"profile {dst} not in ~/.modal.toml")
    src = os.environ["MODAL_PROFILE"]
    app = modal.App.lookup("v2d-weights-copy", create_if_missing=True)
    secret = modal.Secret.from_dict({"MODAL_TOKEN_ID": config[dst]["token_id"],
                                     "MODAL_TOKEN_SECRET": config[dst]["token_secret"]})
    script = f"DIRS = {list(WEIGHT_DIRS)!r}\n" + COPY_SCRIPT
    sandbox = modal.Sandbox.create(
        "python", "-c", script, image=modal.Image.debian_slim().pip_install("modal"), app=app,
        cpu=8, memory=16384, timeout=3600, secrets=[secret],
        volumes={"/src": modal.Volume.from_name("v2d-weights")},
    )
    for line in sandbox.stdout:
        print(f"[copy {src}->{dst}] {line}", end="", flush=True)
    sandbox.wait()
    if sandbox.returncode != 0:
        raise SystemExit(f"weights copy failed: {sandbox.stderr.read()[-2000:]}")


MD5 = "cd /w && find -L {dirs} -type f -print0 | sort -z | xargs -0 -P 8 -n 50 md5sum | sort -k2"


def _md5() -> dict[str, str]:
    import modal

    app = modal.App.lookup("v2d-md5", create_if_missing=True)
    sandbox = modal.Sandbox.create("bash", "-c", MD5.format(dirs=" ".join(WEIGHT_DIRS)), image=modal.Image.debian_slim(),
                                   app=app, timeout=1800, cpu=8, volumes={"/w": modal.Volume.from_name("v2d-weights")})
    out = sandbox.stdout.read()
    sandbox.wait()
    sums = {}
    for line in out.splitlines():
        digest, path = line.split(None, 1)
        sums[path] = digest
    return sums


def step_verify(src: str, dst: str) -> None:
    a, b = json.loads(_internal("md5", src)), json.loads(_internal("md5", dst))
    # .pyc and other caches written at runtime may legitimately differ; everything else must match.
    keys = {k for k in a if not k.endswith(".pyc") and "/__pycache__/" not in k}
    missing = sorted(k for k in keys if k not in b)
    differ = sorted(k for k in keys if k in b and a[k] != b[k])
    print(f"verify: {len(keys)} files on {src}; missing on {dst}: {len(missing)}; differ: {len(differ)}")
    if missing or differ:
        raise SystemExit(f"weights differ, first: {(missing + differ)[:5]}")


def step_build(dst: str) -> None:
    if run([str(REPO / "tools/modal_v2d/run_e2e_modal.sh"), "--build"], dst).returncode != 0:
        raise SystemExit("image build failed (see .e2e/modal_build_logs/)")


def step_probe(dst: str) -> None:
    if run([str(REPO / "tools/modal_v2d/run_e2e_modal.sh"), "--probe"], dst).returncode != 0:
        raise SystemExit("probe failed")


def step_inputs(dst: str, inputs: Path | None) -> None:
    if not inputs:
        print("inputs: none given")
        return
    run(["modal", "volume", "create", "v2d-e2e"], dst, capture_output=True)  # exists already: fine
    for path in sorted(inputs.rglob("*")):
        if path.is_file():
            remote = f"fs/e2e/inputs/{path.relative_to(inputs)}"
            if run(["modal", "volume", "put", "--force", "v2d-e2e", str(path), remote], dst).returncode != 0:
                raise SystemExit(f"upload failed: {path}")
            print(f"inputs: {path} -> /e2e/inputs/{path.relative_to(inputs)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--internal", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--src")
    parser.add_argument("--dst")
    parser.add_argument("--steps", default="secret,weights,verify,build,probe,inputs")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--cap", type=float, default=25.0)
    args = parser.parse_args()
    if args.internal:  # child of _internal, MODAL_PROFILE already pinned
        if args.internal[0] == "copy":
            _copy(args.internal[1])
        elif args.internal[0] == "md5":
            print(json.dumps(_md5()))
        return 0
    if not (args.src and args.dst):
        parser.error("--src and --dst are required")
    spent = month_to_date(args.dst)
    print(f"{args.dst}: MTD ${spent:.2f}, headroom ${args.cap - spent:.2f} below ${args.cap:.0f}")
    if spent >= args.cap:
        raise SystemExit(f"{args.dst} is at its ${args.cap:.0f} stop line")
    for step in args.steps.split(","):
        print(f"== {step}", flush=True)
        if step == "secret":
            step_secret(args.dst)
        elif step == "weights":
            step_weights(args.src, args.dst)
        elif step == "verify":
            step_verify(args.src, args.dst)
        elif step == "build":
            step_build(args.dst)
        elif step == "probe":
            step_probe(args.dst)
        elif step == "inputs":
            step_inputs(args.dst, args.inputs)
        else:
            raise SystemExit(f"unknown step {step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
