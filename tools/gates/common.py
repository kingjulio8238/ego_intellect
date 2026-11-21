"""Shared pieces of the triage gate and the reconstruction QA gate."""

from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

LIMITS_PATH = Path(__file__).resolve().parent / "limits.json"
REJECTED_EXIT = 3  # run_e2e.sh exit code for "the gate ran and rejected the clip" (not a failure)


@lru_cache(maxsize=None)
def load_limits(section: str) -> dict[str, dict[str, Any]]:
    """Limits for one gate: {check: {"min"|"max": number, "why": text}} (tools/gates/limits.json)."""
    return json.loads(LIMITS_PATH.read_text())[section]


def evaluate(metrics: dict[str, Any], section: str) -> tuple[dict[str, Any], list[str]]:
    """Apply the section's limits to ``metrics``: (checks, reasons). A metric that is None
    (not measurable for this clip, e.g. no masks for an imported bundle) is skipped and
    reported; a missing metric is a programming error."""
    checks: dict[str, Any] = {}
    reasons: list[str] = []
    for name, limit in load_limits(section).items():
        if name.startswith("_"):
            continue
        if name not in metrics:
            raise KeyError(f"{section}: limit for unknown metric {name!r}")
        value = metrics[name]
        if value is None:
            checks[name] = {"value": None, "ok": None, **{k: limit[k] for k in ("min", "max") if k in limit}}
            continue
        ok = ("min" not in limit or value >= limit["min"]) and ("max" not in limit or value <= limit["max"])
        checks[name] = {"value": value, "ok": ok, **{k: limit[k] for k in ("min", "max") if k in limit}}
        if not ok:
            bound = f">= {limit['min']}" if "min" in limit and value < limit["min"] else f"<= {limit['max']}"
            reasons.append(f"{name} = {value:.4g} (needs {bound}): {limit['why']}")
    return checks, reasons


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(tmp, path)


def verdict_of(path: Path) -> tuple[bool, list[str]]:
    """(accepted, reasons) from a gate report written by this package."""
    report = json.loads(path.read_text())
    return report["verdict"] in ("accept", "pass"), list(report.get("reasons", []))
