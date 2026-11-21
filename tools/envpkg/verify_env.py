#!/usr/bin/env python3
"""Verify a delivered environment package (or its .tar.gz archive) before using it.

Ships inside every package as ``verify/verify_env.py`` (with the validator it needs in
``verify/envpkg/``) and at the root of every delivery, so it runs without our repository:

    python3 verify/verify_env.py <package-or-archive> [--mujoco] [--isaaclab --rg-root DIR --isaac-python CMD]

Checks, in order:
1. archive: sha256 against the delivery's SHA256SUMS when one sits beside it; safe extraction
   (no absolute paths, no "..", no links or devices, one top-level folder);
2. the package contract, strict (delivery) mode: every file and hash, references, frames,
   motion-vs-reconstruction agreement, QA verdicts, task spec, Isaac Lab and MuJoCo exports;
3. ``--mujoco``: re-runs the MuJoCo physics replay and compares it with the recorded one;
4. ``--isaaclab``: re-runs the Isaac Lab replay (inside an Isaac Lab install with the
   robotic_grounding extension) and requires the same environment and verdict.

Requires Python 3.10+, numpy, pyarrow, jsonschema; mujoco for --mujoco. Exit 0 = verified.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # never write __pycache__ into the package being verified

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import tarfile  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent


def _envpkg():
    """The validator: vendored beside this script in a package or delivery, else the repo's."""
    for root in (HERE, HERE.parent / "verify"):
        if (root / "envpkg" / "validate.py").is_file():
            sys.path.insert(0, str(root))
            import envpkg  # noqa: F401

            return "envpkg"
    sys.path.insert(0, str(HERE.parents[1]))
    return "tools.envpkg"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, target: Path) -> Path:
    """Extract ``archive`` into ``target``; refuse anything that could escape it. Returns the
    package folder (the archive's single top-level directory)."""
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        tops = set()
        for member in members:
            name = member.name
            parts = Path(name).parts
            if name.startswith("/") or ".." in parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"unsafe archive member: {name!r}")
            tops.add(parts[0])
        if len(tops) != 1:
            raise ValueError(f"archive must hold one top-level folder, found {sorted(tops)}")
        try:
            tar.extractall(target, members=members, filter="data")  # members vetted above
        except TypeError:  # Python < 3.12: no extraction filters
            tar.extractall(target, members=members)
    return target / tops.pop()


def check_checksum(archive: Path) -> str | None:
    """Error text when SHA256SUMS beside the archive disagrees; None when it matches or is absent."""
    sums = archive.parent / "SHA256SUMS"
    if not sums.is_file():
        return None
    expected = {line.split()[1].lstrip("*"): line.split()[0] for line in sums.read_text().splitlines() if line.strip()}
    if archive.name not in expected:
        return f"{archive.name} is not listed in {sums}"
    if _sha256(archive) != expected[archive.name]:
        return f"{archive.name}: sha256 does not match SHA256SUMS"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="package folder or .tar.gz archive")
    parser.add_argument("--mujoco", action="store_true", help="re-run the MuJoCo physics replay")
    parser.add_argument("--isaaclab", action="store_true", help="re-run the Isaac Lab replay (Isaac Lab install)")
    parser.add_argument("--rg-root", type=Path, help="robotic_grounding checkout (for --isaaclab)")
    parser.add_argument("--isaac-python", default="python", help="Isaac Lab's python command (for --isaaclab)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    mod = _envpkg()
    validate = __import__(f"{mod}.validate", fromlist=["validate_package"])
    result = {"target": str(args.target), "errors": [], "warnings": [], "checks": {}}
    with tempfile.TemporaryDirectory(prefix="verify_env_") as tmp:
        package = args.target
        if args.target.is_file():
            problem = check_checksum(args.target)
            has_sums = (args.target.parent / "SHA256SUMS").is_file()
            result["checks"]["checksum"] = "failed" if problem else ("ok" if has_sums else "no SHA256SUMS beside the archive")
            if problem:
                result["errors"].append(problem)
            try:
                package = safe_extract(args.target, Path(tmp))
            except (ValueError, tarfile.TarError) as exc:
                result["errors"].append(f"archive: {exc}")
                package = None
        if package is not None:
            report = validate.validate_package(package, strict=True)
            result["errors"] += report.errors
            result["warnings"] += report.warnings
            result["checks"]["contract"] = "ok" if report.ok else f"{len(report.errors)} errors"
            manifest = json.loads((package / "manifest.json").read_text()) if (package / "manifest.json").is_file() else {}
            result["env_id"] = manifest.get("env_id")
            if args.mujoco and report.ok:
                replay = __import__(f"{mod}.mujoco_replay", fromlist=["main"])
                for robot in manifest.get("robots", {}):
                    code = replay.main([str(package), "--robot", robot, "--check"])
                    result["checks"][f"mujoco:{robot}"] = "ok" if code == 0 else "differs"
                    if code:
                        result["errors"].append(f"MuJoCo replay of {robot} differs from the recorded one")
            if args.isaaclab and report.ok:
                if args.rg_root is None:
                    result["errors"].append("--isaaclab needs --rg-root")
                else:
                    check = __import__(f"{mod}.isaaclab_check", fromlist=["main"])
                    code = check.main([str(package), "--rg-root", str(args.rg_root), "--isaac-python", args.isaac_python])
                    result["checks"]["isaaclab"] = "ok" if code == 0 else "differs"
                    if code:
                        result["errors"].append("Isaac Lab replay differs from the package")
    result["ok"] = not result["errors"]
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        for line in result["errors"]:
            print(f"ERROR   {line}")
        for line in result["warnings"]:
            print(f"warning {line}")
        print(f"{result.get('env_id') or args.target}: {'VERIFIED' if result['ok'] else 'FAILED'} {result['checks']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
