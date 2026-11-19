"""Pre-build every Modal image with a live progress table and per-image logs.

    MODAL_PROFILE=<profile> python -m tools.modal_v2d.build            # all 14 images
    MODAL_PROFILE=<profile> python -m tools.modal_v2d.build --only robotic-grounding control

Why: `modal run` builds every image in the app at once with interleaved logs, and one
transient failure (e.g. a PyPI-mirror timeout) aborts all of them. Here each image builds
in its own subprocess with its own log file, transient failures are retried (Modal caches
every finished layer, so a retry only redoes the failed layer), and a table is printed on
every change. Later `modal run` calls reuse the cached images.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.modal_v2d import config as C
from tools.modal_v2d.progress import hms, utc

REPO = Path(__file__).resolve().parents[2]
BUILD_APP = "v2d-e2e-build"
TRANSIENT = (
    "Read timed out",
    "Max retries exceeded",
    "Connection reset",
    "Temporary failure in name resolution",
    "502 Bad Gateway",
    "503 Service Unavailable",
    "TLS handshake timeout",
    "i/o timeout",
)


def all_names() -> list[str]:
    return [C.RG_IMAGE, "control", *C.MODULE_IMAGES]


def build_one(name: str) -> None:
    """Child process: build one image with Modal's output streamed to stdout."""
    import modal

    from tools.modal_v2d.images import build_images

    image = build_images()[name]
    app = modal.App.lookup(BUILD_APP, create_if_missing=True)
    with modal.enable_output():
        image.build(app)
    print(f"V2D_BUILD_OK {name} {image.object_id}", flush=True)


class Board:
    def __init__(self, names: list[str], log_dir: Path):
        self.rows = {n: {"status": "queued", "attempt": 0, "started": None, "elapsed": 0.0, "last": "", "log": str(log_dir / f"{n}.log")} for n in names}
        self.lock = threading.Lock()
        self.started = time.time()

    def update(self, name: str, **fields) -> None:
        with self.lock:
            self.rows[name].update(fields)

    def render(self) -> str:
        with self.lock:
            width = max(len(n) for n in self.rows)
            lines = [f"[v2d-build {utc()}] elapsed {hms(time.time() - self.started)}", f"  {'image':<{width}}  {'status':<10} {'try':>3}  {'elapsed':>8}  last log line"]
            for name, row in self.rows.items():
                elapsed = row["elapsed"] if row["status"] in ("built", "FAILED") else (time.time() - row["started"] if row["started"] else 0)
                lines.append(f"  {name:<{width}}  {row['status']:<10} {row['attempt']:>3}  {hms(elapsed):>8}  {row['last'][:90]}")
            done = sum(r["status"] == "built" for r in self.rows.values())
            failed = sum(r["status"] == "FAILED" for r in self.rows.values())
            lines.append(f"  built {done}/{len(self.rows)}, failed {failed}")
            return "\n".join(lines)


def run_build(name: str, board: Board, retries: int, env: dict[str, str]) -> bool:
    log_path = Path(board.rows[name]["log"])
    for attempt in range(1, retries + 2):
        board.update(name, status="building", attempt=attempt, started=time.time())
        print(f"[v2d-build {utc()}] ▶ {name} (attempt {attempt}) → {log_path}", flush=True)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n===== attempt {attempt} at {utc()} =====\n")
            log.flush()
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "tools.modal_v2d.build", "--child", name],
                cwd=REPO,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            tail: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                stripped = line.strip()
                if stripped and not stripped.startswith(("╭", "╰", "│")):
                    board.update(name, last=stripped)
                tail = (tail + [line])[-200:]
            code = process.wait()
        elapsed = time.time() - board.rows[name]["started"]
        text = "".join(tail)
        ok = code == 0 and "V2D_BUILD_OK" in text
        with open(log_path.parent / "builds.jsonl", "a", encoding="utf-8") as journal:  # benchmark record
            journal.write(json.dumps({
                "ts": time.time(), "image": name, "attempt": attempt, "ok": ok, "seconds": round(elapsed, 3),
                "cached": ok and elapsed < 30, "profile": os.environ.get("MODAL_PROFILE"),
            }) + "\n")
        if ok:
            board.update(name, status="built", elapsed=elapsed)
            print(f"[v2d-build {utc()}] ✔ {name} built in {hms(elapsed)}", flush=True)
            return True
        transient = any(marker in text for marker in TRANSIENT)
        print(
            f"[v2d-build {utc()}] ✘ {name} attempt {attempt} failed after {hms(elapsed)}"
            f"{' (transient network error)' if transient else ''}; see {log_path}",
            flush=True,
        )
        if not transient:
            break
    board.update(name, status="FAILED", elapsed=time.time() - board.rows[name]["started"])
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=all_names(), help="build only these images")
    parser.add_argument("--parallel", type=int, default=4, help="concurrent builds (default 4)")
    parser.add_argument("--retries", type=int, default=2, help="retries for transient network failures only")
    parser.add_argument("--log-dir", type=Path, default=REPO / ".e2e" / "modal_build_logs")
    parser.add_argument("--table-every", type=int, default=60, help="seconds between progress tables")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        build_one(args.child)
        return 0

    if not os.environ.get("MODAL_PROFILE"):
        raise SystemExit("set MODAL_PROFILE explicitly (never rely on the globally active profile)")
    names = args.only or all_names()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    board = Board(names, args.log_dir)
    print(f"[v2d-build {utc()}] profile {os.environ['MODAL_PROFILE']}: building {len(names)} images, {args.parallel} at a time; logs in {args.log_dir}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    stop = threading.Event()

    def ticker() -> None:
        while not stop.wait(args.table_every):
            print(board.render(), flush=True)

    threading.Thread(target=ticker, daemon=True).start()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(lambda n: run_build(n, board, args.retries, env), names))
    stop.set()
    print(board.render(), flush=True)
    if not all(results):
        print(f"[v2d-build {utc()}] FAILED: {', '.join(n for n, ok in zip(names, results) if not ok)}", flush=True)
        return 1
    print(f"[v2d-build {utc()}] all {len(names)} images built", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
