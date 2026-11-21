"""CLI: intake | build | validate (see tools/envpkg/__init__.py)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import ContractError


def _print(report, as_json: bool, extra: dict | None = None) -> int:
    if as_json:
        print(json.dumps({"ok": report.ok, "errors": report.errors, "warnings": report.warnings, **(extra or {})}, indent=2))
    else:
        for line in report.errors:
            print(f"ERROR   {line}")
        for line in report.warnings:
            print(f"warning {line}")
        print("OK" if report.ok else f"FAILED ({len(report.errors)} errors)")
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m tools.envpkg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    intake = sub.add_parser("intake", help="validate a vendor delivery manifest (intake/v1)")
    intake.add_argument("manifest", type=Path)
    intake.add_argument("--no-probe", action="store_true", help="skip ffprobe checks (not for acceptance)")
    intake.add_argument("--json", action="store_true")

    build = sub.add_parser("build", help="assemble an env_package/v1 core from a run root")
    build.add_argument("--run-root", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--robot", action="append", dest="robots", help="robot to include (default: all retargeted)")
    build.add_argument("--sequence-id")
    build.add_argument("--instruction")
    build.add_argument("--object-prompt")
    build.add_argument("--allow-flagged", action="store_true", help="package a clip the physics QA flagged")
    build.add_argument("--allow-missing-provenance", action="store_true", help="non-deliverable package without run records")
    build.add_argument("--force", action="store_true", help="replace an existing package")
    build.add_argument("--task-profile", type=Path, help="task profile JSON (default: the run config's contracts.task)")
    build.add_argument("--simulators", default="isaaclab,mujoco", help="exports to write (comma-separated)")

    export = sub.add_parser("export", help="build, strictly validate and archive one clip's package (exit 3: QA refused)")
    export.add_argument("--run-root", type=Path, required=True)
    export.add_argument("--task-profile", type=Path)
    export.add_argument("--no-strict", action="store_true", help="skip delivery-mode validation (development only)")
    export.add_argument("--exit-zero", action="store_true", help="exit 0 when the QA refused the clip (pipeline stages)")

    archive = sub.add_parser("archive", help="deterministic <env_id>.tar.gz of a package")
    archive.add_argument("package", type=Path)
    archive.add_argument("--out", type=Path, required=True)

    deliver = sub.add_parser("deliver", help="collect accepted exports into a delivery folder")
    deliver.add_argument("run_roots", type=Path, nargs="+")
    deliver.add_argument("--out", type=Path, required=True)
    deliver.add_argument("--delivery-id", required=True)
    deliver.add_argument("--external", action="store_true", help="refuse robots whose asset redistribution is unconfirmed")

    validate = sub.add_parser("validate", help="validate an env_package/v1 directory")
    validate.add_argument("package", type=Path)
    validate.add_argument("--strict", action="store_true", help="also require what delivery requires")
    validate.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "intake":
            from .intake import validate_intake

            report = validate_intake(args.manifest, probe=not args.no_probe)
            return _print(report, args.json, {"clips": report.clips})
        if args.command == "build":
            from .build import build_package

            out = build_package(
                args.run_root, args.out, robots=args.robots, sequence_id=args.sequence_id,
                instruction=args.instruction, object_prompt=args.object_prompt,
                allow_flagged=args.allow_flagged, allow_missing_provenance=args.allow_missing_provenance,
                force=args.force, task_profile=args.task_profile,
                simulators=tuple(x for x in args.simulators.split(",") if x),
            )
            print(f"built {out}")
            return 0
        if args.command == "export":
            from .delivery import QARefused, export_run

            try:
                report = export_run(args.run_root, task_profile=args.task_profile, strict=not args.no_strict)
            except QARefused as exc:
                print(f"export: REJECTED ({exc})")
                return 0 if args.exit_zero else 3
            print(f"export: {report['env_id']} -> {report['archive']} ({report['bytes']} bytes, sha256 {report['sha256'][:12]})")
            return 0
        if args.command == "archive":
            from .delivery import archive

            print(archive(args.package, args.out))
            return 0
        if args.command == "deliver":
            from .delivery import deliver

            manifest = deliver(args.run_roots, args.out, delivery_id=args.delivery_id, external=args.external)
            print(f"delivered {len(manifest['environments'])} environment(s) to {args.out}; skipped {manifest['skipped']}")
            return 0
        from .validate import validate_package

        return _print(validate_package(args.package, strict=args.strict), args.json)
    except ContractError as exc:
        print(f"ERROR   {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
