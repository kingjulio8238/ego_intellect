"""Seed the resident-hint Dict from a recorded run's journal (call_spec records).

    MODAL_PROFILE=<p> python -m tools.fastrecon.seed_resident_hints <journal.jsonl> [--clear] [--disable]

The runtime records hints itself after every successful call; this only lets the first
run on a fresh workspace start with preloading containers.
"""

from __future__ import annotations

import argparse
import json

import modal

from tools.modal_v2d import config as C


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("journal", nargs="?")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--disable", action="store_true", help="turn residents off (plain warm-ups)")
    parser.add_argument("--enable", action="store_true")
    args = parser.parse_args()
    store = modal.Dict.from_name(C.RESIDENT_HINTS_DICT, create_if_missing=True)
    if args.clear:
        store.clear()
    if args.disable:
        store["disabled"] = True
    if args.enable:
        store.pop("disabled", None)
    if args.journal:
        hints: dict[str, dict] = {}
        for line in open(args.journal):
            row = json.loads(line)
            if row.get("kind") != "call_spec" or row.get("on_cpu"):
                continue
            spec = row["spec"]
            argv = spec.get("argv") or []
            if len(argv) > 2 and argv[1] == "-m":
                hints.setdefault(spec["image"], {})[argv[2]] = {
                    "argv": argv, "env": spec.get("env") or {}, "workdir": spec.get("workdir"),
                    "mounts": [m for m in spec.get("mounts") or [] if not m.get("mirrored")]}
        for image, modules in hints.items():
            store[image] = modules
    for key, value in store.items():
        print(key, sorted(value) if isinstance(value, dict) else value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
