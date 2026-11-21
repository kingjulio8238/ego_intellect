"""Formatted, live progress for Modal runs (stdlib only; imported in every image).

Three outputs, all derived from the same events:

* console: every line is timestamped and tagged, stage transitions get banners, and a
  heartbeat line every ``C.HEARTBEAT_S`` says what is running, for how long, the stage
  count, the ETA, and the estimated spend (`modal app logs` shows exactly this);
* live snapshot: one record per invocation (``inv:<id>``) and per module call
  (``call:<id>``) in a Modal Dict, read by ``python -m tools.modal_v2d.board``;
* journal: ``/vol/progress/<id>.jsonl`` on the state volume (durable history).

Costs are estimates from ``C.RATES`` (list prices), not billing data; the board shows the
workspace's real month-to-date spend next to them.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from typing import Any, Callable

from tools.modal_v2d import config as C

STAGE_RUN = re.compile(r"^\[RUN\] (\S+)")
STAGE_DONE = re.compile(r"^\[DONE\] (\S+)")
STAGE_SKIP = re.compile(r"^\[SKIP\] (\S+)")
STAGE_PLAN = re.compile(r"^\[DRY-RUN\] (\S+):")
STAGE_FAIL = re.compile(r"stage '([^']+)' (?:failed|did not produce)")
# Inner progress counters worth surfacing (RSL-RL, tqdm-style "n/m", GR00T steps).
INNER = (
    re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)"),
    re.compile(r"\bstep[s]?\s*[:=]?\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)\s*/\s*(\d+)\s*\[\d"),  # tqdm: " 12/390 [00:03<..."
)


def utc(ts: float | None = None, seconds: bool = True) -> str:
    return time.strftime("%H:%M:%SZ" if seconds else "%H:%MZ", time.gmtime(ts or time.time()))


def hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def hourly_rate(gpu: str | None, cpu: float, memory_gib: float) -> float:
    rate = cpu * C.RATES["cpu_core_h"] + memory_gib * C.RATES["mem_gib_h"]
    if gpu:
        rate += C.RATES["gpu_h"].get(gpu.split(":")[0].upper(), C.RATES["gpu_h_unknown"])
    return rate


def inner_progress(line: str) -> tuple[int, int] | None:
    for pattern in INNER:
        match = pattern.search(line)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            # A finished bar (n == m) says nothing about the time left in the stage.
            if 0 < total and done < total:
                return done, total
    return None


class GpuSampler:
    """Samples `nvidia-smi` every few seconds; reports utilisation and memory per window.

    Benchmark evidence for "was this GPU worth paying for": mean/peak utilisation and peak
    memory of each stage and module call. Absent GPU or nvidia-smi -> empty stats.
    """

    QUERY = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]

    def __init__(self, every_s: float = 5.0):
        import shutil

        self.every_s = every_s
        self.available = shutil.which("nvidia-smi") is not None
        self.lock = threading.Lock()
        self.windows: dict[str, list[tuple[float, float]]] = {}
        self.memory_total = 0.0
        self._stop = threading.Event()
        if self.available:
            threading.Thread(target=self._loop, name="v2d-gpu-sampler", daemon=True).start()

    def _sample(self) -> tuple[float, float] | None:
        import subprocess

        try:
            out = subprocess.run(self.QUERY, capture_output=True, text=True, timeout=10).stdout
            util, used, total = (float(x) for x in out.strip().splitlines()[0].split(","))
        except Exception:
            return None
        self.memory_total = total
        return util, used

    def _loop(self) -> None:
        while not self._stop.wait(self.every_s):
            sample = self._sample()
            if sample:
                with self.lock:
                    for values in self.windows.values():
                        values.append(sample)

    def open(self, key: str) -> None:
        with self.lock:
            self.windows[key] = []

    def close(self, key: str) -> dict[str, float]:
        with self.lock:
            values = self.windows.pop(key, [])
        if not values:
            return {}
        utils = [u for u, _ in values]
        return {
            "gpu_util_mean": round(sum(utils) / len(utils), 1),
            "gpu_util_max": max(utils),
            "gpu_mem_max_mib": max(m for _, m in values),
            "gpu_mem_total_mib": self.memory_total,
            "gpu_samples": len(values),
        }

    def stop(self) -> None:
        self._stop.set()


class _Store:
    """Live snapshot writes, off the caller's thread (latest value per key wins).

    Writes are network calls to a Modal Dict; doing them in a background thread keeps log
    output and the launcher from ever waiting on them. Failures are reported, not hidden.
    """

    def __init__(self, store, emit: Callable[[str], None]):
        self.store = store
        self.emit = emit
        self.failures = 0
        self.pending: dict[str, dict] = {}
        self.inflight = 0
        self.cond = threading.Condition()
        if store is not None:
            threading.Thread(target=self._writer, name="v2d-progress-store", daemon=True).start()

    def put(self, key: str, value: dict) -> None:
        if self.store is None:
            return
        with self.cond:
            self.pending[key] = value
            self.cond.notify()

    def flush(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        with self.cond:
            while (self.pending or self.inflight) and time.time() < deadline:
                self.cond.wait(0.1)

    def _writer(self) -> None:
        while True:
            with self.cond:
                while not self.pending:
                    self.cond.wait()
                key, value = next(iter(self.pending.items()))
                del self.pending[key]
                self.inflight += 1
            try:
                self.store[key] = value
            except Exception as exc:  # live view degrades; the run and the journal continue
                self.failures += 1
                if self.failures in (1, 10, 100):
                    self.emit(f"progress store write failed ({self.failures}x): {type(exc).__name__}: {exc}")
            with self.cond:
                self.inflight -= 1
                self.cond.notify_all()


class Tracker:
    """Progress of one run_e2e.sh invocation inside its role container."""

    def __init__(
        self,
        *,
        invocation: str,
        role: str,
        argv: list[str],
        timeout_s: int,
        gpu: str | None,
        cpu: float,
        memory_gib: float,
        store=None,
        journal_dir: str | None = None,
        out=None,
        heartbeat_s: int | None = None,
        meta: dict | None = None,
        gpu_sampler: "GpuSampler | None" = None,
    ):
        self.invocation = invocation
        self.meta = {"gpu": gpu, "cpu": cpu, "memory_gib": memory_gib, **(meta or {})}
        self.gpu_sampler = gpu_sampler
        self.role = role
        self.argv = list(argv)
        self.command = C.subcommand(argv)
        self.started = time.time()
        self.deadline = self.started + timeout_s
        self.rate = hourly_rate(gpu, cpu, memory_gib)
        self.out = out or sys.stdout
        self.heartbeat_s = heartbeat_s or C.HEARTBEAT_S
        self.lock = threading.RLock()
        self.store = _Store(store, self._say)
        self.journal = (
            os.path.join(journal_dir, f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(self.started))}-{invocation}.jsonl")
            if journal_dir
            else None
        )
        self.planned: list[str] = []
        self.stage: str | None = None
        self.stage_started = 0.0
        self.done: list[str] = []
        self.skipped: list[str] = []
        self.durations: list[float] = []
        self.failed: str | None = None
        self.calls: dict[str, dict[str, Any]] = {}  # in-flight module calls (several during prefetch)
        self.calls_done = 0
        self.call_seconds = 0.0
        self.call_cost = 0.0
        self.last_line = ""
        self.inner: tuple[int, int] | None = None
        self.inner_started: tuple[float, int] | None = None
        self.status = "running"
        self._stop = threading.Event()
        self.stage_calls: list[dict] = []  # module calls made during the current stage

    # -- output -----------------------------------------------------------------------
    def _say(self, text: str) -> None:
        self.out.write(f"[v2d {utc()}] {text}\n")
        self.out.flush()

    def _banner(self, text: str) -> None:
        self.out.write(f"\n[v2d {utc()}] ━━━━ {text} ━━━━\n")
        self.out.flush()

    def event(self, kind: str, **fields) -> None:
        """Append a structured record to the durable journal (no console output)."""
        self._journal(kind, **fields)

    def _journal(self, kind: str, **fields) -> None:
        if not self.journal:
            return
        record = {"ts": time.time(), "kind": kind, "invocation": self.invocation, **fields}
        # The volume is briefly unwritable while another thread reloads it (EPERM on
        # /vol/progress; batch b2 lost a clip's last events that way), so retry.
        for attempt in range(8):
            try:
                os.makedirs(os.path.dirname(self.journal), exist_ok=True)
                with open(self.journal, "a", encoding="utf-8") as stream:  # closed at once: volume commits need that
                    stream.write(json.dumps(record) + "\n")
                return
            except OSError as exc:
                if attempt == 7:
                    self._say(f"journal write failed: {exc}")
                    return
                time.sleep(0.25 * (attempt + 1))

    # -- derived numbers --------------------------------------------------------------
    def cost(self) -> float:
        return (time.time() - self.started) / 3600 * self.rate + self.call_cost + self._running_call_cost()

    def _running_call_cost(self) -> float:
        now = time.time()
        return sum((now - call["started"]) / 3600 * call["rate"] for call in list(self.calls.values()))

    def counts(self) -> tuple[int, int | None]:
        finished = len(self.done) + len(self.skipped)
        return finished, (len(self.planned) or None)

    def eta(self) -> float | None:
        """Absolute ETA: inner counter first, else average stage time x remaining stages."""
        now = time.time()
        if self.inner and self.inner_started:
            done, total = self.inner
            t0, d0 = self.inner_started
            if done > d0 and now > t0:
                stage_end = now + (total - done) * (now - t0) / (done - d0)
                remaining_stages = self._remaining_after_current()
                average = (sum(self.durations) / len(self.durations)) if self.durations else 0
                return stage_end + remaining_stages * average
        finished, total = self.counts()
        if total and self.durations:
            average = sum(self.durations) / len(self.durations)
            running = 1 if self.stage else 0
            left = max(total - finished - running, 0)
            current_left = max(average - (now - self.stage_started), 0) if self.stage else 0
            return now + current_left + left * average
        return None

    def _remaining_after_current(self) -> int:
        finished, total = self.counts()
        if not total:
            return 0
        return max(total - finished - (1 if self.stage else 0), 0)

    def snapshot(self) -> dict[str, Any]:
        finished, total = self.counts()
        eta = self.eta()
        return {
            "invocation": self.invocation,
            "role": self.role,
            "command": self.command,
            "argv": self.argv,
            "status": self.status,
            "started": self.started,
            "deadline": self.deadline,
            "updated": time.time(),
            "stage": self.stage,
            "stage_started": self.stage_started or None,
            "stages_finished": finished,
            "stages_total": total,
            "stages_skipped": len(self.skipped),
            "failed_stage": self.failed,
            "inner": list(self.inner) if self.inner else None,
            "call": {k: v for k, v in self.call.items() if k != "rate"} if self.call else None,
            "calls_running": len(self.calls),
            "calls_done": self.calls_done,
            "eta": eta,
            "cost_est": round(self.cost(), 4),
            "last_line": self.last_line[-200:],
        }

    def publish(self) -> None:
        self.store.put(f"inv:{self.invocation}", self.snapshot())

    def status_line(self) -> str:
        finished, total = self.counts()
        parts = [self.command]
        parts.append(f"stage {finished}/{total if total else '?'}")
        if self.stage:
            parts.append(f"{self.stage} {hms(time.time() - self.stage_started)}")
        if self.inner:
            parts.append(f"{self.inner[0]}/{self.inner[1]}")
        if len(self.calls) > 1:
            names = ", ".join(f"{c['image']} {hms(time.time() - c['started'])}" for c in list(self.calls.values())[:4])
            more = len(self.calls) - 4
            parts.append(f"{len(self.calls)} modules in parallel: {names}{f' +{more}' if more > 0 else ''}")
        elif self.call:
            parts.append(f"module {self.call['image']} {hms(time.time() - self.call['started'])}")
        if self.call:
            remote = self.call.get("remote_last")
            if remote:
                parts.append(f"› {remote[-90:]}")
        parts.append(f"elapsed {hms(time.time() - self.started)}")
        eta = self.eta()
        parts.append(f"ETA {utc(eta, seconds=False)}" if eta else f"cap {utc(self.deadline, seconds=False)}")
        parts.append(f"~${self.cost():.2f}")
        return " | ".join(parts)

    # -- events -----------------------------------------------------------------------
    def begin(self, planned: list[str]) -> None:
        with self.lock:
            self.planned = planned
            self._banner(
                f"{self.command} ({self.role}) invocation {self.invocation} | "
                f"{len(planned) if planned else '?'} stages | cap {utc(self.deadline, seconds=False)} | "
                f"~${self.rate:.2f}/h for this container"
            )
            for index, name in enumerate(planned, start=1):
                self.out.write(f"    {index:>2}. {name}\n")
            self._journal(
                "begin",
                role=self.role,
                command=self.command,
                argv=self.argv,
                planned=planned,
                started=self.started,
                deadline=self.deadline,
                rate_per_h=self.rate,
                **self.meta,
            )
            self.publish()

    def line(self, text: str) -> None:
        """Feed one line of launcher output; echo it timestamped and react to markers."""
        text = text.rstrip("\n")
        with self.lock:
            if match := STAGE_RUN.match(text):
                self._stage_start(match.group(1))
            elif match := STAGE_DONE.match(text):
                self._stage_end(match.group(1), "done")
            elif match := STAGE_SKIP.match(text):
                self._stage_end(match.group(1), "skip")
            elif match := STAGE_FAIL.search(text):
                self._stage_end(match.group(1), "failed")
            elif text.strip():
                self.last_line = text.strip()
                counter = inner_progress(text)
                if counter:
                    if not self.inner or counter[1] != self.inner[1] or counter[0] < self.inner[0]:
                        self.inner_started = (time.time(), counter[0])  # new counter (or restart)
                    self.inner = counter
        self.out.write(f"  {utc()} │ {text}\n")
        self.out.flush()

    def _stage_start(self, name: str) -> None:
        finished, total = self.counts()
        self._stage_start_common(name, f"▶ stage {finished + 1}/{total or '?'} {name}")

    def _stage_start_common(self, name: str, banner: str) -> None:
        self.stage, self.stage_started = name, time.time()
        self.inner = self.inner_started = None
        self.stage_calls = []
        if self.gpu_sampler:
            self.gpu_sampler.open(f"stage:{name}")
        self._banner(banner)
        self._journal("stage_start", stage=name)
        self.publish()

    def _stage_end(self, name: str, how: str) -> None:
        now = time.time()
        if how == "skip":
            self.skipped.append(name)
            self._say(f"↷ {name} skipped (completed with matching inputs)")
        elif how == "done":
            took = now - self.stage_started if self.stage == name else 0.0
            self.done.append(name)
            self.durations.append(took)
            finished, total = self.counts()
            self._banner(f"✔ {name} done in {hms(took)} ({finished}/{total or '?'}) | ~${self.cost():.2f} so far")
        else:
            self.failed = name
            self._banner(f"✘ {name} FAILED after {hms(now - self.stage_started)}")
        seconds = now - self.stage_started if self.stage == name else 0.0
        metrics: dict[str, Any] = {}
        if how != "skip" and self.stage == name:
            call_cost = sum(c["cost_est"] for c in self.stage_calls)
            metrics = {
                "container_cost_est": round(seconds / 3600 * self.rate, 6),
                "calls": len(self.stage_calls),
                "call_seconds": round(sum(c["seconds"] for c in self.stage_calls), 3),
                "call_cost_est": round(call_cost, 6),
                "cost_est": round(seconds / 3600 * self.rate + call_cost, 6),
                **(self.gpu_sampler.close(f"stage:{name}") if self.gpu_sampler else {}),
            }
        self._journal(f"stage_{how}", stage=name, seconds=round(seconds, 3), started=self.stage_started if self.stage == name else None, **metrics)
        if self.stage == name:
            self.stage = None
            self.inner = self.inner_started = None
        self.publish()

    @property
    def call(self) -> dict[str, Any] | None:
        """The most recently started in-flight call."""
        return list(self.calls.values())[-1] if self.calls else None

    def side_stage_start(self, name: str, banner: str) -> None:
        """A stage this package adds (e.g. the weight prefetch): journaled and benchmarked
        like a launcher stage, but not counted against the launcher's planned stages."""
        with self.lock:
            self._stage_start_common(name, banner)

    def side_stage_end(self, name: str, ok: bool) -> None:
        with self.lock:
            now = time.time()
            seconds = now - self.stage_started
            call_cost = sum(c["cost_est"] for c in self.stage_calls)
            self._journal(
                "stage_done" if ok else "stage_failed",
                stage=name,
                seconds=round(seconds, 3),
                started=self.stage_started,
                side_stage=True,
                container_cost_est=round(seconds / 3600 * self.rate, 6),
                calls=len(self.stage_calls),
                call_seconds=round(sum(c["seconds"] for c in self.stage_calls), 3),
                call_cost_est=round(call_cost, 6),
                cost_est=round(seconds / 3600 * self.rate + call_cost, 6),
            )
            self._banner(f"{'✔' if ok else '✘'} {name} {'done' if ok else 'had failures'} in {hms(seconds)} | ~${self.cost():.2f} so far")
            self.stage = None
            self.publish()

    def call_start(self, image: str, call_id: str, argv: list[str], gpu: str | None, cpu: float, memory_gib: float) -> None:
        with self.lock:
            self.calls[call_id] = {
                "image": image,
                "call_id": call_id,
                "started": time.time(),
                "rate": hourly_rate(gpu, cpu, memory_gib),
                "gpu": gpu or "cpu",
                "argv": " ".join(argv)[:200],
            }
            self.inner = self.inner_started = None  # counters belong to one command, not the stage
            self._say(f"→ module {image} (call {call_id}, {gpu or 'cpu'}): {' '.join(argv)[:160]}")
            self._journal("call_start", image=image, call_id=call_id, argv=argv, gpu=gpu or "cpu")
            self.publish()

    def call_remote_status(self, record: dict | None, call_id: str | None = None) -> None:
        with self.lock:
            call = self.calls.get(call_id) if call_id else self.call
            if call and record:
                call["remote_last"] = record.get("last", "")
                call["remote_lines"] = record.get("lines", 0)

    def call_end(self, returncode: int, seconds: float, details: dict | None = None, call_id: str | None = None) -> None:
        with self.lock:
            call = self.calls.pop(call_id, None) if call_id else (self.calls.pop(self.call["call_id"]) if self.call else None)
            if call:
                # The module container only bills while the remote call is in flight.
                billable = (details or {}).get("remote_s", seconds)
                cost = billable / 3600 * call["rate"]
                self.call_cost += cost
                self.call_seconds += seconds
                self.calls_done += 1
                mark = "✔" if returncode == 0 else "✘"
                extra = ""
                if details and "overhead_s" in details:
                    extra = f"; exec {hms(details.get('exec_s', 0))}, startup+sync {hms(details['overhead_s'])}"
                self._say(f"{mark} module {call['image']} ({call['gpu']}) exit={returncode} in {hms(seconds)} (~${cost:.2f}{extra})")
                record = {
                    "image": call["image"],
                    "call_id": call["call_id"],
                    "stage": self.stage,
                    "returncode": returncode,
                    "started": call["started"],
                    "seconds": round(seconds, 3),
                    "gpu": call["gpu"],
                    "rate_per_h": call["rate"],
                    "cost_est": round(cost, 6),
                    **(details or {}),
                }
                self.stage_calls.append(record)
                self._journal("call_end", **record)
            self.publish()

    def finish(self, returncode: int) -> None:
        with self.lock:
            self._stop.set()
            # Exit 3 = a gate (triage, reconstruction QA) ran and rejected the clip: a result, not a failure.
            self.status = "succeeded" if returncode == 0 else "rejected" if returncode == 3 else "failed"
            finished, total = self.counts()
            self._banner(
                f"{'✔' if returncode == 0 else '✘'} {self.command} {self.status} (exit {returncode}) | "
                f"stages {finished}/{total or '?'} ({len(self.skipped)} skipped) | "
                f"module calls {self.calls_done} | elapsed {hms(time.time() - self.started)} | ~${self.cost():.2f}"
            )
            if self.gpu_sampler:
                self.gpu_sampler.stop()
            self._journal(
                "finish",
                returncode=returncode,
                status=self.status,
                seconds=round(time.time() - self.started, 3),
                container_cost_est=round((time.time() - self.started) / 3600 * self.rate, 6),
                calls=self.calls_done,
                call_seconds=round(self.call_seconds, 3),
                call_cost_est=round(self.call_cost, 6),
                cost_est=round(self.cost(), 6),
                stages_done=self.done,
                stages_skipped=self.skipped,
                failed_stage=self.failed,
                active_run=self.meta.get("active_run"),
            )
            self.publish()
        self.store.flush()

    # -- heartbeat --------------------------------------------------------------------
    def start_heartbeat(self, remote_lookup: Callable[[str], dict | None] | None = None) -> None:
        def beat() -> None:
            while not self._stop.wait(self.heartbeat_s):
                for call in list(self.calls.values()):
                    if not remote_lookup:
                        break
                    try:  # network round trip: outside the lock so output never stalls on it
                        record = remote_lookup(call["call_id"])
                    except Exception as exc:
                        record = None
                        self._say(f"remote status lookup failed: {type(exc).__name__}: {exc}")
                    self.call_remote_status(record, call["call_id"])
                with self.lock:
                    if self._stop.is_set():
                        return
                    self._say("♥ " + self.status_line())
                    self.publish()

        threading.Thread(target=beat, name="v2d-heartbeat", daemon=True).start()


class CallReporter:
    """Worker side: tag remote output lines and publish a throttled live status."""

    def __init__(self, *, image: str, call_id: str, store=None, out=None, every_s: float = 15.0):
        self.image = image
        self.call_id = call_id
        self.out = out or sys.stdout
        self.store = _Store(store, lambda text: self.out.write(f"[{image} {utc()}] {text}\n"))
        self.every_s = every_s
        self.lines = 0
        self.last = ""
        self.started = time.time()
        self._published = 0.0

    def line(self, text: str) -> None:
        text = text.rstrip("\n")
        self.lines += 1
        if text.strip():
            self.last = text.strip()
        self.out.write(f"[{self.image} {utc()}] {text}\n")
        now = time.time()
        if now - self._published >= self.every_s:
            self.out.flush()
            self.publish("running")

    def publish(self, status: str, returncode: int | None = None, *, wait: bool = False) -> None:
        self._published = time.time()
        self.store.put(
            f"call:{self.call_id}",
            {
                "image": self.image,
                "status": status,
                "returncode": returncode,
                "lines": self.lines,
                "last": self.last[-200:],
                "started": self.started,
                "updated": self._published,
            },
        )
        if wait:
            self.store.flush()
