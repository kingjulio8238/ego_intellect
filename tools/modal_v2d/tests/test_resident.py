"""Resident execution: a preloaded zygote runs the call exactly like `python -m` would."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from tools.modal_v2d import worker


class _Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, text: str) -> None:
        self.lines.append(text)


def _package(tmp: Path) -> None:
    pkg = tmp / "zpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "state.py").write_text("loaded = None\n")
    (pkg / "_resident.py").write_text(textwrap.dedent("""
        def preload(module, argv):
            import zpkg.state as st
            st.loaded = ("model", argv[argv.index("--w") + 1])
    """))
    (pkg / "entry.py").write_text(textwrap.dedent("""
        import os, sys
        from zpkg import state
        print("argv", sys.argv[1:], "loaded", state.loaded, "env", os.environ.get("ZFLAG"))
        if __name__ == "__main__":
            sys.exit(int(sys.argv[sys.argv.index("--rc") + 1]))
    """))


def test_zygote_runs_call_with_preloaded_state(tmp_path, monkeypatch):
    _package(tmp_path)
    monkeypatch.setattr(worker.containerenv, "image_workdir", lambda: str(tmp_path))
    argv = [sys.executable, "-m", "zpkg.entry", "--w", "W1", "--rc", "3"]
    hint = {"argv": argv, "env": {"ZFLAG": "on"}, "workdir": None}
    zygote = worker.Zygote("test_image", hint)
    assert zygote.matches({"argv": argv, "env": {"ZFLAG": "on"}, "workdir": None})
    assert not zygote.matches({"argv": argv, "env": {"ZFLAG": "off"}, "workdir": None})
    reporter = _Reporter()
    log = tmp_path / "logs" / "call.log"
    code = zygote.run([*argv[:4], "W1", "--rc", "3"], str(log), reporter)
    assert code == 3
    text = log.read_text()
    assert "loaded ('model', 'W1')" in text and "env on" in text
    assert "argv ['--w', 'W1', '--rc', '3']" in text
    assert zygote.preload_s is not None

    # Same command without a zygote: identical output apart from the preloaded state.
    plain = subprocess.run(argv, cwd=tmp_path, env={**os.environ, "ZFLAG": "on"}, capture_output=True, text=True)
    assert plain.returncode == 3
    assert plain.stdout.replace("None", "('model', 'W1')") == text


class _FakeQueue:
    def __init__(self, items: dict, take_after: float | None = None):
        import threading
        self.items = items
        self.lock = threading.Lock()
        if take_after is not None:  # a resident that picks the spec up after `take_after` s
            threading.Timer(take_after, self._resident_take).start()

    def _resident_take(self) -> None:
        with self.lock:
            for key in list(self.items):
                self.items[key] = []

    def len(self, partition):
        with self.lock:
            return len(self.items.get(partition, []))

    def get(self, partition, block=True):
        with self.lock:
            values = self.items.get(partition, [])
            return values.pop(0) if values else None


class _FakeCall:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def test_claim_taken_back_when_resident_never_runs(monkeypatch):
    from tools.modal_v2d import config as C
    from tools.modal_v2d.runtime import StateSync
    monkeypatch.setattr(C, "RESIDENT_CLAIM_S", 1)
    queue, call = _FakeQueue({"k": [{"argv": ["x"]}]}), _FakeCall()
    assert StateSync._claimed(queue, "k", call) is False
    assert call.cancelled and queue.len("k") == 0  # spec taken back, resident cancelled


def test_claim_detected_when_resident_takes_spec(monkeypatch):
    from tools.modal_v2d import config as C
    from tools.modal_v2d.runtime import StateSync
    monkeypatch.setattr(C, "RESIDENT_CLAIM_S", 5)
    queue, call = _FakeQueue({"k": [{"argv": ["x"]}]}, take_after=0.3), _FakeCall()
    assert StateSync._claimed(queue, "k", call) is True
    assert not call.cancelled


class _Vol:
    def reload(self):
        pass

    def commit(self):
        pass


def test_keep_warm_preloads_next_call_in_same_container(tmp_path, monkeypatch):
    from tools.modal_v2d import config as C

    _package(tmp_path)
    monkeypatch.setattr(worker.containerenv, "image_workdir", lambda: str(tmp_path))
    monkeypatch.setattr(C, "RESIDENT_MODULES", frozenset({"zpkg.entry"}))
    monkeypatch.setattr(C, "CALL_LOG_ROOT", str(tmp_path / "calls"))
    monkeypatch.setattr(worker, "_WARM", {})
    argv = [sys.executable, "-m", "zpkg.entry", "--w", "W1", "--rc", "0"]
    spec = {"image": "test_image", "argv": argv, "env": {"ZFLAG": "on"}, "workdir": str(tmp_path), "mounts": [],
            "keep_warm": True}
    first = worker.execute({**spec, "call_id": "c1"}, state_volume=_Vol(), weights_volume=_Vol())
    assert first["returncode"] == 0 and not first["timing"].get("resident")
    assert "loaded None" in Path(first["log"]).read_text()  # fresh process: nothing preloaded
    assert "zpkg.entry" in worker._WARM
    second = worker.execute({**spec, "call_id": "c2"}, state_volume=_Vol(), weights_volume=_Vol())
    assert second["returncode"] == 0 and second["timing"].get("resident")
    assert "loaded ('model', 'W1')" in Path(second["log"]).read_text()  # ran on the preloaded process
    # A non-matching call kills the preload and runs fresh.
    third = worker.execute({**spec, "env": {"ZFLAG": "off"}, "call_id": "c3", "keep_warm": False},
                           state_volume=_Vol(), weights_volume=_Vol())
    assert not third["timing"].get("resident") and "loaded None" in Path(third["log"]).read_text()
    assert worker._WARM == {}
