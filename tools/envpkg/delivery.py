"""Delivery: export one clip's package, archive it, and collect accepted clips into a delivery.

    python3 -m tools.envpkg export --run-root R [--exit-zero]      # one clip: R/package/ + R/qa/export.json
    python3 -m tools.envpkg deliver --delivery-id ID --out DIR R1 R2 ... [--external]

``export`` builds the package (both simulator exports and the MuJoCo check), validates it in
strict (delivery) mode, and archives it deterministically. A clip the QA rejected or flagged
gets ``verdict: reject`` and no package; that is a result (exit 3, or 0 with --exit-zero),
like the gates. Anything else that stops a package (a missing input, a strict-mode error such
as a dirty source tree) is a failure (exit 2).

``deliver`` copies the accepted archives into DIR with ``delivery.json``, ``SHA256SUMS``,
``verify_env.py`` (+ its validator) and ``README.md``. ``--external`` refuses robots whose
asset redistribution is not confirmed in robot_assets.json. There is no upload step yet: the
delivery folder is the hand-off (decided 2026-11-21). Contract: docs/pipeline/DELIVERY.md.
"""

from __future__ import annotations

import gzip
import io
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import ContractError, sha256_file, write_json

DELIVERY_SCHEMA = "delivery/v1"
EXPORT_SCHEMA = "export/v1"
HERE = Path(__file__).resolve().parent
# What the lab-side verifier needs (verify_env.py imports these through envpkg.validate).
VERIFIER_MODULES = ("__init__.py", "common.py", "validate.py", "build.py", "task.py", "success.py",
                    "isaaclab.py", "isaaclab_check.py", "mujoco_replay.py")
EXPORT_REPORT = Path("qa/export.json")
PACKAGE_DIR = Path("package")


class QARefused(ContractError):
    """The clip's QA rejected or flagged it: no package (a result, not a failure)."""


def vendor_verifier(target: Path) -> None:
    """Write verify_env.py and the validator it imports into ``target`` (a package's verify/)."""
    (target / "envpkg/schemas").mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "verify_env.py", target / "verify_env.py")
    for name in VERIFIER_MODULES:
        shutil.copy2(HERE / name, target / "envpkg" / name)
    for schema in sorted((HERE / "schemas").glob("*.json")):
        shutil.copy2(schema, target / "envpkg/schemas" / schema.name)


def archive(package: Path, out_dir: Path) -> Path:
    """``<env_id>.tar.gz`` of the package: one top-level ``<env_id>/`` folder, sorted entries,
    zeroed times and owners, normalised modes, so the same package always gives the same bytes."""
    manifest = json.loads((package / "manifest.json").read_text())
    env_id = manifest["env_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{env_id}.tar.gz"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        paths = sorted(p for p in package.rglob("*"))
        for path in [package, *paths]:
            rel = Path(env_id) / path.relative_to(package)
            if path.is_symlink():
                raise ContractError(f"{path}: links are not archived")
            info = tar.gettarinfo(str(path), arcname=str(rel))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if path.is_dir() else 0o644
            if path.is_file():
                with open(path, "rb") as stream:
                    tar.addfile(info, stream)
            else:
                tar.addfile(info)
    with open(target, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
        gz.write(buffer.getvalue())
    return target


def export_run(run_root: Path, *, task_profile: Path | None = None, strict: bool = True,
               simulators: tuple[str, ...] = ("isaaclab", "mujoco")) -> dict[str, Any]:
    """Build, validate (strict) and archive one clip's package; returns the export report
    (also written to RUN_ROOT/qa/export.json). Raises QARefused for a rejected/flagged clip."""
    from .build import build_package
    from .validate import validate_package

    run_root = run_root.resolve()
    out = run_root / PACKAGE_DIR
    report: dict[str, Any] = {"schema": EXPORT_SCHEMA, "run_root": run_root.name,
                              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        staging = out / "env"
        package = build_package(run_root, staging, task_profile=task_profile, force=True, simulators=simulators,
                                allow_missing_provenance=not strict)  # development exports only
    except ContractError as exc:
        if str(exc).startswith("QA "):  # build_package refuses rejected/flagged clips with "QA <verdict> ..."
            report.update(verdict="reject", reasons=[str(exc)])
            write_json(run_root / EXPORT_REPORT, report)
            raise QARefused(str(exc)) from exc
        raise
    checked = validate_package(package, strict=strict)
    if not checked.ok:
        raise ContractError("package fails delivery (strict) validation:\n  " + "\n  ".join(checked.errors))
    manifest = json.loads((package / "manifest.json").read_text())
    for old in out.glob("*.tar.gz"):
        old.unlink()
    target = archive(package, out)
    report.update(
        verdict="accept", reasons=[], env_id=manifest["env_id"], package=str((PACKAGE_DIR / "env")),
        archive=str(target.relative_to(run_root)), sha256=sha256_file(target), bytes=target.stat().st_size,
        task_id=json.loads((package / manifest["task"]["spec"]).read_text())["task_id"],
        robots=sorted(manifest["robots"]), simulators=sorted(manifest["simulators"]),
        qa=manifest["qa"]["verdict"], warnings=checked.warnings,
    )
    write_json(run_root / EXPORT_REPORT, report)
    return report


def deliver(run_roots: list[Path], out: Path, *, delivery_id: str, external: bool = False) -> dict[str, Any]:
    """Collect the accepted exports of ``run_roots`` into the delivery folder ``out``."""
    registry = json.loads((HERE / "robot_assets.json").read_text())
    if out.exists() and any(out.iterdir()):
        raise ContractError(f"{out} is not empty; a delivery folder is written once")
    envs, skipped = [], []
    for run_root in run_roots:
        report_path = run_root / EXPORT_REPORT
        if not report_path.is_file():
            skipped.append({"run": run_root.name, "reason": "not exported"})
            continue
        report = json.loads(report_path.read_text())
        if report.get("verdict") != "accept":
            skipped.append({"run": run_root.name, "reason": "; ".join(report.get("reasons", [])) or "rejected"})
            continue
        archive_path = run_root / report["archive"]
        if sha256_file(archive_path) != report["sha256"]:
            raise ContractError(f"{archive_path}: archive changed since export")
        for robot in report["robots"]:
            entry = registry.get(robot)
            if external and not (entry and entry.get("redistribution_confirmed")):
                raise ContractError(f"external delivery refused: redistribution of {robot}'s assets is not confirmed "
                                    f"(tools/envpkg/robot_assets.json)")
        envs.append({k: report[k] for k in ("env_id", "sha256", "bytes", "task_id", "robots", "simulators", "qa")}
                    | {"archive": Path(report["archive"]).name, "run": run_root.name})
    if not envs:
        raise ContractError(f"nothing to deliver: {skipped}")
    out.mkdir(parents=True, exist_ok=True)
    for env in envs:
        run_root = next(r for r in run_roots if r.name == env["run"])
        shutil.copy2(run_root / PACKAGE_DIR / env["archive"], out / env["archive"])
    (out / "SHA256SUMS").write_text("".join(f"{e['sha256']}  {e['archive']}\n" for e in sorted(envs, key=lambda e: e["archive"])))
    vendor_verifier(out / "verify")
    robots = sorted({r for e in envs for r in e["robots"]})
    manifest = {
        "schema": DELIVERY_SCHEMA,
        "delivery_id": delivery_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audience": "external" if external else "internal",
        "environments": sorted(envs, key=lambda e: e["env_id"]),
        "skipped": skipped,
        "robot_assets": {r: registry.get(r, {}) for r in robots},
        "verify": "python3 verify/verify_env.py <archive> [--mujoco]",
    }
    write_json(out / "delivery.json", manifest)
    (out / "README.md").write_text(_readme(manifest))
    return manifest


def _readme(manifest: dict[str, Any]) -> str:
    rows = "\n".join(f"| `{e['archive']}` | {e['task_id']} | {', '.join(e['robots'])} | {', '.join(e['simulators'])} |"
                     for e in manifest["environments"])
    return f"""# Delivery {manifest['delivery_id']}

{len(manifest['environments'])} environment package(s). Each archive unpacks to one folder whose
`manifest.json` describes it (env_package/v1).

| Archive | Task | Robots | Simulators |
|---|---|---|---|
{rows}

## Verify before use

    python3 verify/verify_env.py <archive>.tar.gz            # checksum, safe unpack, full contract
    python3 verify/verify_env.py <archive>.tar.gz --mujoco   # + re-run the MuJoCo physics replay

Needs Python 3.10+, numpy, pyarrow, jsonschema (and mujoco for `--mujoco`). Every package also
carries the same verifier as `verify/verify_env.py`.

## Use

- Isaac Lab: `isaaclab/env.json` in each package (task id, motion argument, train/play commands;
  needs the robotic_grounding extension at the recorded revision).
- MuJoCo: `mujoco/<robot>/scene.xml` (`mujoco.MjModel.from_xml_path`); `mujoco/env.json` lists
  the servo gains, the arm command delay and the control timing.
- Task and rewards: `task/task.json`; success scoring: `verify/envpkg/success.py`.
"""
