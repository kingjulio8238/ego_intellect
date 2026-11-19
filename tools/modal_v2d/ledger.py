"""Client-side model of the GPU containers one invocation holds (single-clip mode).

Modal's Starter plan caps a workspace at 10 concurrent GPUs, counting containers that sit
idle in their scaledown window, and a call past the cap waits silently (~65 s after a slot
frees). `reconstruct --fast` starts speculative containers (warm-ups, resident preloads)
to cut latency; unbudgeted, they push the run past the cap and its own critical-path calls
wait behind them (fast3: first calls waited 47-58 s).

The ledger tracks one entry per container this invocation believes alive:
  busy  running a call, or a resident preloading/waiting for its call;
  idle  warm, or finished a call; counts until its scaledown expiry.
Real calls reuse an idle container of their image, else take a new entry, waiting while
the ledger is full. Speculation only takes an entry while `reserve` entries stay free, so it
can never delay a real call. Stdlib only.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass(eq=False)  # identity: two containers of one image are distinct
class Entry:
    image: str
    busy: bool
    expires: float | None = None
    since: float = field(default_factory=time.time)


class GpuLedger:
    def __init__(self, capacity: int, reserve: int, scaledown_s: float, boot_s: float = 30.0,
                 scaledown_per_image: dict[str, float] | None = None):
        self.capacity, self.reserve = capacity, reserve
        self.scaledown_s, self.boot_s = scaledown_s, boot_s
        self.scaledown_per_image = dict(scaledown_per_image or {})
        self.entries: list[Entry] = []
        self.cv = threading.Condition()
        self.skipped = 0  # speculative containers not started for lack of budget

    def _expire(self, now: float) -> None:
        self.entries = [e for e in self.entries if e.busy or (e.expires or 0) > now]

    def _next_expiry(self, now: float) -> float:
        idle = [e.expires for e in self.entries if not e.busy and e.expires]
        return max(0.05, min(idle) - now) if idle else 5.0

    def acquire(self, image: str) -> tuple[Entry, float]:
        """A container for a real call of `image`: (entry, seconds waited)."""
        t0 = time.time()
        with self.cv:
            while True:
                now = time.time()
                self._expire(now)
                for entry in self.entries:
                    if entry.image == image and not entry.busy:
                        entry.busy, entry.expires = True, None  # reuse the warm/idle container
                        return entry, time.time() - t0
                if len(self.entries) < self.capacity:
                    entry = Entry(image, busy=True)
                    self.entries.append(entry)
                    return entry, time.time() - t0
                self.cv.wait(timeout=min(self._next_expiry(now), 5.0))

    def speculate(self, image: str, *, busy: bool, reserve: int | None = None) -> Entry | None:
        """A speculative container (warm-up: idle; resident or hedge: busy), only with room to
        spare (`reserve` entries left free; default self.reserve)."""
        with self.cv:
            now = time.time()
            self._expire(now)
            if len(self.entries) >= self.capacity - (self.reserve if reserve is None else reserve):
                self.skipped += 1
                return None
            entry = Entry(image, busy=busy, expires=None if busy else now + self.boot_s + self.scaledown(image))
            self.entries.append(entry)
            return entry

    def done(self, entry: Entry | None) -> None:
        """The call finished: its container idles until its scaledown expiry."""
        if entry is None:
            return
        with self.cv:
            entry.busy, entry.expires = False, time.time() + self.scaledown(entry.image)
            self.cv.notify_all()

    def drop(self, entry: Entry | None) -> None:
        """The container is gone (cancelled resident, exited container)."""
        if entry is None:
            return
        with self.cv:
            if entry in self.entries:
                self.entries.remove(entry)
            self.cv.notify_all()

    TEARDOWN_S = 10.0  # a container still counts against the cap briefly after its scaledown

    def scaledown(self, image: str) -> float:
        return float(self.scaledown_per_image.get(image, self.scaledown_s)) + self.TEARDOWN_S

    def held(self) -> int:
        with self.cv:
            self._expire(time.time())
            return len(self.entries)
