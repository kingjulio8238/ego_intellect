"""Batch mode: concurrent invocations on one state volume, hedged dispatch, clip steps."""

from __future__ import annotations

import json
import os
import threading
import unittest
import time

import pytest

from tools.modal_v2d import batch as B
from tools.modal_v2d import config as C
from tools.modal_v2d import sync


# --------------------------------------------------------------------------- index partitions
def _write(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as stream:
        stream.write(text)


def test_partitioned_index_keeps_concurrent_invocations_apart(tmp_path):
    fs, parts, legacy = str(tmp_path / "fs"), str(tmp_path / "parts"), str(tmp_path / "legacy.json")
    run_a, run_b, inputs = str(tmp_path / "e2e/runs/a"), str(tmp_path / "e2e/runs/b"), str(tmp_path / "e2e/inputs")
    _write(f"{run_a}/f1"), _write(f"{run_b}/g1"), _write(f"{inputs}/clip_a.mp4")

    # Two invocations load at the same time (neither sees the other's saves) ...
    first = sync.Mirror(fs, legacy, parts)
    first.load({run_a: False, inputs: True})
    second = sync.Mirror(fs, legacy, parts)
    second.load({run_b: False, inputs: True})
    first.push(run_a), first.push(inputs, delete=False)
    _write(f"{inputs}/clip_b.mp4")
    second.push(run_b), second.push(inputs, delete=False)
    # ... and save in either order: nothing of the other is lost.
    second.save()
    first.save()
    third = sync.Mirror(fs, legacy, parts)
    third.load({run_a: False, run_b: False, inputs: True})
    assert f"{run_a}/f1" in third.index and f"{run_b}/g1" in third.index
    assert f"{inputs}/clip_a.mp4" in third.index and f"{inputs}/clip_b.mp4" in third.index


def test_partition_seeded_from_legacy_index_and_never_blanked(tmp_path):
    fs, parts, legacy = str(tmp_path / "fs"), str(tmp_path / "parts"), str(tmp_path / "legacy.json")
    run_a = str(tmp_path / "e2e/runs/a")
    _write(f"{run_a}/f1")
    old = sync.Mirror(fs, legacy)  # an invocation from before partitioning
    old.load()
    old.push(run_a)
    old.save()
    stamp = old.index[f"{run_a}/f1"]

    mirror = sync.Mirror(fs, legacy, parts)
    mirror.load({run_a: False})
    assert mirror.index[f"{run_a}/f1"] == stamp  # mtimes restored on checkout come from here
    mirror.save()

    # An invocation that registers run_a without having synced it must not blank its file.
    other = sync.Mirror(fs, legacy, parts)
    other.load({})
    other.roots[run_a] = False
    other.save()
    again = sync.Mirror(fs, legacy, parts)
    again.load({run_a: False})
    assert again.index[f"{run_a}/f1"] == stamp


# --------------------------------------------------------------------------- hedged dispatch
class _Dict:
    def __init__(self):
        self.data, self.lock = {}, threading.Lock()

    def put(self, key, value, skip_if_exists=False):
        with self.lock:
            if skip_if_exists and key in self.data:
                return False
            self.data[key] = value
            return True

    def get(self, key, default=None):
        with self.lock:
            return self.data.get(key, default)

    def pop(self, key, default=None):
        with self.lock:
            return self.data.pop(key, default)


class _Call:
    """A spawned call whose container starts after `start_after` s and then claims."""

    def __init__(self, claims, spec, start_after, lane_runs):
        self.claims, self.spec, self.cancelled = claims, spec, False
        self.result, self.done = None, threading.Event()
        threading.Timer(start_after, self._start, args=(lane_runs,)).start()

    def _start(self, lane_runs):
        if self.cancelled:
            self.done.set()
            return
        if self.claims.put(self.spec["claim"], {"lane": self.spec["lane"], "t": time.time()}, skip_if_exists=True):
            lane_runs.append(self.spec["lane"])
            self.result = {"returncode": 0, "changed": [], "deleted": [], "dirs": [], "log": "", "timing": {"worker_s": 1}}
        else:
            self.result = {"lost_claim": True}
        self.done.set()

    def get(self, timeout=None):
        if not self.done.wait(timeout):
            raise TimeoutError
        return self.result

    def cancel(self):
        self.cancelled = True


class _Function:
    def __init__(self, claims, start_after, lane_runs):
        self.claims, self.start_after, self.lane_runs, self.spawned = claims, start_after, lane_runs, 0

    def spawn(self, spec):
        self.spawned += 1
        return _Call(self.claims, spec, self.start_after, self.lane_runs)

    def remote(self, spec):  # pragma: no cover - only used when hedging is off
        raise AssertionError("hedged path expected")


def _state_with(monkeypatch, claims):
    import modal

    from tools.modal_v2d.runtime import StateSync

    monkeypatch.setattr(modal.Dict, "from_name", staticmethod(lambda *a, **k: claims))
    state = StateSync.__new__(StateSync)
    state.tracker = None
    return state


@pytest.mark.parametrize("primary_after,expect_lane,expect_hedge_spawned", [(0.2, "primary", 0), (5.0, "hedge", 1)])
def test_hedge_runs_exactly_once(monkeypatch, primary_after, expect_lane, expect_hedge_spawned):
    monkeypatch.setattr(C, "HEDGE_AFTER_S", 1.0)
    claims, runs, lines = _Dict(), [], []
    state = _state_with(monkeypatch, claims)
    primary, hedge = _Function(claims, primary_after, runs), _Function(claims, 0.2, runs)
    state.hedge_functions = {"img": hedge}
    result = state._call_hedged("img", primary, {"call_id": "c1"}, lines.append, on_cpu=False)
    time.sleep(primary_after + 0.5 if expect_lane == "hedge" else 0.3)  # let the loser start too
    assert result["returncode"] == 0 and result["timing"]["lane"] == expect_lane
    assert hedge.spawned == expect_hedge_spawned
    assert runs == [expect_lane]  # the other container never ran the command, even starting late
    assert len(claims.data) == 1  # the claim stays as a tombstone for late starters


def test_hedge_off_for_cpu_calls(monkeypatch):
    claims = _Dict()
    state = _state_with(monkeypatch, claims)
    state.hedge_functions = {"img": object()}

    class Plain:
        def remote(self, spec):
            return {"returncode": 0, "plain": True}

    assert state._call_hedged("img", Plain(), {"call_id": "c"}, print, on_cpu=True)["plain"]


# --------------------------------------------------------------------------- manifest
MANIFEST = {
    "clips": [
        {"name": "tb1", "video": "/v/tb.mp4", "task_profile": "/p/tb.json", "sequence_id": "tissue_box"},
        {"name": "ap1", "video": "/v/ap.mp4", "task_profile": "/p/ap.json", "sequence_id": "airplane",
         "reconstruct_args": ["--hand-tracking", "hamer", "--fast"]},
    ],
}


def test_clip_steps_use_explicit_config_never_the_active_pointer():
    manifest = B.load_manifest(json.dumps(MANIFEST))
    steps = B.clip_steps(manifest["clips"][0], manifest)
    assert [s[0] if s[0] != "--config" else s[2] for s in steps] == ["init", "setup", "triage", "reconstruct", "inspect"]
    assert steps[0][:3] == ["init", "--run-root", "/e2e/runs/tb1"]
    for step in steps[1:]:
        assert step[:2] == ["--config", "/e2e/runs/tb1/e2e_config.json"]
    assert steps[2][2:] == ["triage", "--video", "/v/tb.mp4"]
    assert steps[3][-len(B.DEFAULT_RECONSTRUCT_ARGS):] == B.DEFAULT_RECONSTRUCT_ARGS
    assert B.clip_steps(manifest["clips"][1], manifest)[3][-3:] == ["--hand-tracking", "hamer", "--fast"]
    gates_off = {**manifest, "triage": False, "recon_qa": False}
    assert [s[2] for s in B.clip_steps(manifest["clips"][0], gates_off)[1:]] == ["setup", "reconstruct"]
    from tools.modal_v2d import runtime

    for step in steps:  # every step locks and syncs exactly this clip's run root
        assert runtime.runs_touched(step) == {"/e2e/runs/tb1"}
        assert C.role_for(step) == "control"


def test_manifest_rejects_duplicate_names():
    bad = {"clips": [MANIFEST["clips"][0], MANIFEST["clips"][0]]}
    with pytest.raises(ValueError, match="unique"):
        B.load_manifest(json.dumps(bad))


def test_summary_reports_makespan_and_throughput():
    text = B.summarize([
        {"name": "a", "code": 0, "start": 0.0, "end": 600.0},
        {"name": "b", "code": 1, "start": 10.0, "end": 700.0},
    ])
    assert "makespan 11:40" in text and "1/2 ok" in text and "FAIL 1" in text


# --------------------------------------------------------------------------- GPU slots
class _Queue:
    def __init__(self, tokens):
        import queue as q

        self.q = q.Queue()
        for token in tokens:
            self.q.put(token)

    def get(self, partition, timeout=None):
        import queue as q

        try:
            return self.q.get(timeout=timeout)
        except q.Empty:
            return None

    def put(self, value, partition):
        self.q.put(value)


def test_gpu_slots_bound_calls_in_flight(monkeypatch):
    import modal

    from tools.modal_v2d.runtime import StateSync

    slots = _Queue([0, 1])
    monkeypatch.setattr(modal.Queue, "from_name", staticmethod(lambda *a, **k: slots))
    state = StateSync.__new__(StateSync)
    state.options = {"gpu_slots_key": "b1"}
    in_flight, peak, lock = [0], [0], threading.Lock()

    def call():
        slot = state._acquire_slot("img", lambda text: None)
        with lock:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        time.sleep(0.2)
        with lock:
            in_flight[0] -= 1
        state._release_slot(slot)

    threads = [threading.Thread(target=call) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 2 and slots.q.qsize() == 2  # never above the budget; every token returned


def test_no_slots_without_batch_key():
    from tools.modal_v2d.runtime import StateSync

    state = StateSync.__new__(StateSync)
    state.options = {}
    assert state._acquire_slot("img", print) is None


def test_shard_assignment_follows_headroom():
    from tools.modal_v2d.batch_shard import assign

    clips = [{"name": f"c{i}"} for i in range(6)]
    shards = assign(clips, {"a": 10.0, "b": 2.0}, 1.0)
    assert len(shards["a"]) == 5 and len(shards["b"]) == 1
    with pytest.raises(SystemExit):
        assign(clips, {"a": 2.0, "b": 2.0}, 1.0)


# --------------------------------------------------------------------------- GPU ledger
def test_ledger_reuses_idle_containers_and_caps_new_ones():
    from tools.modal_v2d.ledger import GpuLedger

    ledger = GpuLedger(capacity=2, reserve=1, scaledown_s=0.3)
    ledger.TEARDOWN_S = 0.0
    a, waited = ledger.acquire("moge")
    assert waited < 0.1 and ledger.held() == 1
    ledger.done(a)  # idle until its scaledown expires
    again, _ = ledger.acquire("moge")
    assert again is a and ledger.held() == 1  # same container, no new GPU
    b, _ = ledger.acquire("sam2")
    assert ledger.held() == 2
    ledger.done(again)
    t0 = time.time()
    c, waited = ledger.acquire("hamer")  # full: waits for moge's idle container to expire
    assert 0.2 < waited < 1.5 and time.time() - t0 >= 0.2
    assert ledger.held() == 2
    ledger.done(b), ledger.done(c)


def test_ledger_speculation_never_takes_the_reserve():
    from tools.modal_v2d.ledger import GpuLedger

    ledger = GpuLedger(capacity=3, reserve=2, scaledown_s=60)
    warm = ledger.speculate("moge", busy=False)
    assert warm is not None
    assert ledger.speculate("sam2", busy=True) is None and ledger.skipped == 1
    call, waited = ledger.acquire("moge")  # lands on the warm container
    assert call is warm and waited < 0.1
    other, _ = ledger.acquire("sam2")
    assert ledger.held() == 2
    ledger.drop(other)
    assert ledger.held() == 1


def test_hedge_needs_budget(monkeypatch):
    from tools.modal_v2d.ledger import GpuLedger

    monkeypatch.setattr(C, "HEDGE_AFTER_S", 0.3)
    claims, runs, lines = _Dict(), [], []
    state = _state_with(monkeypatch, claims)
    primary, hedge = _Function(claims, 1.5, runs), _Function(claims, 0.1, runs)
    state.hedge_functions = {"img": hedge}
    budget = GpuLedger(capacity=1, reserve=0, scaledown_s=60)
    entry, _ = budget.acquire("img")  # the call's own container fills the budget
    result = state._call_hedged("img", primary, {"call_id": "c2"}, lines.append, False, None, budget)
    assert hedge.spawned == 0 and result["timing"]["lane"] == "primary"
    assert any("no GPU budget" in line for line in lines)
    assert budget.held() == 1


def test_init_step_is_skipped_when_run_already_initialized(tmp_path, monkeypatch):
    from tools.modal_v2d import runtime

    monkeypatch.setattr(runtime, "RUNS_ROOT", str(tmp_path / "runs"))
    root = tmp_path / "runs" / "c1"
    step = ["init", "--run-root", str(root), "--sequence-id", "x"]
    assert not runtime.init_done(step)
    root.mkdir(parents=True)
    (root / "e2e_config.json").write_text("{}")
    assert runtime.init_done(step)
    assert not runtime.init_done(["--config", str(root / "e2e_config.json"), "setup"])


def test_extension_cache_is_per_gpu_arch(tmp_path, monkeypatch):
    from tools.modal_v2d import worker

    root, local = tmp_path / "cache", tmp_path / "local"
    monkeypatch.setattr(C, "EXT_CACHE_ROOT", str(root))
    monkeypatch.setattr(C, "LOCAL_EXT_DIR", str(local))
    (root / "img" / "gsplat_cuda").mkdir(parents=True)  # pre-split cache, built on L40S
    (root / "img" / "gsplat_cuda" / "k.so").write_text("sm89")
    monkeypatch.setattr(worker, "_gpu_sm", lambda: "89")
    env = {}
    assert worker._ext_cache_in("img", env) == str(root / "img" / "sm89")
    assert (local / "gsplat_cuda" / "k.so").read_text() == "sm89"  # seeded from the legacy cache
    import shutil

    shutil.rmtree(local)
    monkeypatch.setattr(worker, "_gpu_sm", lambda: "90")
    assert worker._ext_cache_in("img", {}) == str(root / "img" / "sm90")
    assert not (local / "gsplat_cuda").exists()  # H100 never loads L40S binaries


def test_gpu_spec_lists():
    assert C.gpu_spec("L40S") == "L40S"
    assert C.gpu_spec("L40S, H100,A100-80GB") == ["L40S", "H100", "A100-80GB"]
    assert C.gpu_label("L40S,H100") == "L40S"


def test_cancelled_call_kills_its_command(tmp_path):
    """An exception in the worker while the command runs (Modal input cancellation) must not
    leave the command running: it would write into the next call's mounts."""
    import signal
    import subprocess as sp

    from tools.modal_v2d import worker

    marker = tmp_path / "alive"
    script = f"import time\nwhile True:\n    open({str(marker)!r}, 'w').write(str(time.time()))\n    time.sleep(0.1)\n"

    class Boom(Exception):
        pass

    class Reporter:
        def line(self, text):
            raise Boom  # first output line: simulate the cancellation arriving mid-call

    import sys

    spec = {"argv": [sys.executable, "-u", "-c", "print('start', flush=True)\n" + script]}
    with pytest.raises(Boom):
        worker._run_subprocess(spec, dict(os.environ), str(tmp_path), str(tmp_path / "log.txt"), Reporter())
    time.sleep(0.5)
    before = marker.stat().st_mtime if marker.exists() else 0
    time.sleep(0.5)
    after = marker.stat().st_mtime if marker.exists() else 0
    assert before == after  # nothing is writing any more


class RenderQueueTests(unittest.TestCase):
    def test_sample_is_stable_and_about_one_in_n(self):
        names = [f"clip_{i:04d}" for i in range(2000)]
        chosen = B.render_queue(names, every=20)
        self.assertEqual(chosen, B.render_queue(list(reversed(names)), every=20)[::-1])
        self.assertTrue(70 <= len(chosen) <= 130, len(chosen))  # ~100 expected
        self.assertEqual(B.render_queue(names[:500], every=20), [n for n in chosen if n in names[:500]])

    def test_flagged_always_rendered_and_zero_disables_sampling(self):
        names = ["a", "b", "c"]
        self.assertEqual(B.render_queue(names, every=0, flagged=["c"]), ["c"])
        self.assertEqual(B.render_queue(names, every=0), [])
        self.assertEqual(B.render_queue(names, every=1), names)

    def test_render_commands_target_each_run_root(self):
        self.assertEqual(B.render_commands(["x"]), [["--config", "/e2e/runs/x/e2e_config.json", "render"]])


class RobotStepsTests(unittest.TestCase):
    def test_default_robot_steps_retarget_then_simulate(self):
        self.assertEqual(B.robot_steps("c1", {}), [
            ["--config", "/e2e/runs/c1/e2e_config.json", "retarget"],
            ["--config", "/e2e/runs/c1/e2e_config.json", "simulate"],
        ])

    def test_robot_steps_can_be_disabled(self):
        self.assertEqual(B.robot_steps("c1", {"robot_steps": []}), [])

    def test_robot_steps_run_in_the_rg_role(self):
        for step in B.robot_steps("c1", {}) + B.render_commands(["c1"]):
            self.assertEqual(C.role_for(step), "rg")

    def test_qa_report_path_matches_the_simulate_stage(self):
        self.assertEqual(B.qa_report_path("c1"), "/e2e/runs/c1/simulate/vega/replay_metrics.json")
