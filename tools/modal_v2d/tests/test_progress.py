"""Progress tracker, worker reporter, and board rendering (no Modal needed)."""

from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
from pathlib import Path

from tools.modal_v2d import progress


class FakeStore(dict):
    """Stands in for modal.Dict (supports item assignment and .items())."""


def make_tracker(store=None, journal_dir=None, argv=("reconstruct", "--video", "x.mp4")):
    out = io.StringIO()
    tracker = progress.Tracker(
        invocation="inv1",
        role="control",
        argv=list(argv),
        timeout_s=3600,
        gpu=None,
        cpu=4,
        memory_gib=16,
        store=store,
        journal_dir=journal_dir,
        out=out,
        heartbeat_s=3600,
    )
    return tracker, out


class TrackerTests(unittest.TestCase):
    def test_stage_markers_drive_counts_banners_and_journal(self) -> None:
        store = FakeStore()
        with tempfile.TemporaryDirectory() as tmp:
            tracker, out = make_tracker(store, tmp)
            tracker.begin(["reconstruct.prepare", "reconstruct.run", "reconstruct.validate"])
            tracker.line("[SKIP] reconstruct.prepare: completed with matching inputs\n")
            tracker.line("[RUN] reconstruct.run\n")
            tracker.line("  [1/1] reconstruct: python run.py\n")
            self.assertEqual(tracker.stage, "reconstruct.run")
            tracker.call_start("v2d_moge", "c1", ["python", "moge.py"], "L40S", 8, 32)
            tracker.call_end(0, 120.0)
            tracker.line("[DONE] reconstruct.run\n")
            tracker.line("[RUN] reconstruct.validate\n")
            tracker.line("error: stage 'reconstruct.validate' failed in check with exit code 1; see log\n")
            tracker.finish(1)
            tracker.store.flush()

            self.assertEqual(tracker.counts(), (2, 3))
            self.assertEqual(tracker.failed, "reconstruct.validate")
            text = out.getvalue()
            self.assertIn("▶ stage 2/3 reconstruct.run", text)
            self.assertIn("✔ reconstruct.run done", text)
            self.assertIn("✔ module v2d_moge (L40S) exit=0", text)
            self.assertIn("✘ reconstruct.validate FAILED", text)
            self.assertIn("│   [1/1] reconstruct: python run.py", text)  # launcher lines are timestamped
            snap = store["inv:inv1"]
            self.assertEqual(snap["status"], "failed")
            self.assertEqual(snap["stages_total"], 3)
            # one L40S module call of 120 s dominates the estimate
            self.assertGreater(snap["cost_est"], 120 / 3600 * 1.95)
            journal = list(Path(tmp).glob("*-inv1.jsonl"))[0].read_text().splitlines()
            kinds = [json.loads(line)["kind"] for line in journal]
            self.assertEqual(
                kinds,
                ["begin", "stage_skip", "stage_start", "call_start", "call_end", "stage_done", "stage_start", "stage_failed", "finish"],
            )

    def test_concurrent_calls_are_tracked_and_costed_separately(self) -> None:
        tracker, out = make_tracker()
        tracker.begin(["setup.reconstruction-host"])
        tracker.line("[RUN] setup.reconstruction-host\n")
        tracker.call_start("v2d_sam2", "a", ["python", "dl"], None, 4, 16)
        tracker.call_start("v2d_hamer", "b", ["python", "dl"], None, 4, 16)
        self.assertIn("2 modules in parallel", tracker.status_line())
        tracker.call_end(0, 60.0, {"remote_s": 60.0}, call_id="a")
        self.assertEqual(list(tracker.calls), ["b"])
        tracker.call_end(0, 120.0, {"remote_s": 120.0}, call_id="b")
        self.assertEqual(tracker.calls_done, 2)
        self.assertAlmostEqual(tracker.call_cost, 180 / 3600 * progress.hourly_rate(None, 4, 16), places=6)
        self.assertIn("✔ module v2d_hamer (cpu) exit=0", out.getvalue())

    def test_inner_counter_gives_eta_and_status_line(self) -> None:
        tracker, _ = make_tracker(argv=("train-expert", "--max-iterations", "100"))
        tracker.begin(["train-expert.vega"])
        tracker.line("[RUN] train-expert.vega\n")
        tracker.line("Learning iteration 10/100\n")
        tracker.inner_started = (time.time() - 10, 10)  # 10 s ago at iteration 10
        tracker.line("Learning iteration 20/100\n")
        eta = tracker.eta()
        self.assertIsNotNone(eta)
        self.assertAlmostEqual(eta - time.time(), 80, delta=5)  # 1 s/iteration, 80 left
        line = tracker.status_line()
        self.assertIn("stage 0/1", line)
        self.assertIn("20/100", line)
        self.assertIn("ETA", line)

    def test_inner_progress_patterns(self) -> None:
        self.assertEqual(progress.inner_progress("Learning iteration 5/20000"), (5, 20000))
        self.assertEqual(progress.inner_progress(" 37%|███ | 144/390 [00:12<00:20, 11.9it/s]"), (144, 390))
        self.assertIsNone(progress.inner_progress("loaded 3 files"))
        self.assertIsNone(progress.inner_progress("ratio 9/3"))  # done > total is not progress
        self.assertIsNone(progress.inner_progress("Fetching 3 files: 100%|███| 3/3 [00:06<00:00]"))  # finished bar

    def test_store_failures_are_reported_not_raised(self) -> None:
        class Broken:
            def __setitem__(self, key, value):
                raise ConnectionError("dict unavailable")

        tracker, out = make_tracker(Broken())
        tracker.publish()
        tracker.store.flush()
        time.sleep(0.05)
        self.assertIn("progress store write failed (1x): ConnectionError", out.getvalue())


class CallReporterTests(unittest.TestCase):
    def test_tags_lines_and_publishes_final_status(self) -> None:
        store = FakeStore()
        out = io.StringIO()
        reporter = progress.CallReporter(image="v2d_sam2", call_id="c9", store=store, out=out, every_s=0)
        reporter.line("frame 1\n")
        reporter.line("frame 2\n")
        reporter.publish("succeeded", 0, wait=True)
        self.assertIn("[v2d_sam2 ", out.getvalue())
        self.assertEqual(store["call:c9"]["lines"], 2)
        self.assertEqual(store["call:c9"]["last"], "frame 2")
        self.assertEqual(store["call:c9"]["status"], "succeeded")


class BoardTests(unittest.TestCase):
    def test_rows_and_stale_detection(self) -> None:
        from tools.modal_v2d import board

        store = FakeStore()
        tracker, _ = make_tracker(store)
        tracker.begin(["a", "b"])
        tracker.line("[RUN] a\n")
        tracker.store.flush()
        now = time.time()
        cells = board.row(store["inv:inv1"], now)
        self.assertEqual(cells[0], "reconstruct [inv1]")
        self.assertTrue(cells[1].startswith("a"))
        self.assertTrue(cells[2].startswith("0/2 stages"))
        stale = board.row(store["inv:inv1"], now + 10 * 3600)
        self.assertTrue(stale[3].startswith("STALE"))


if __name__ == "__main__":
    unittest.main()


class BenchTests(unittest.TestCase):
    def test_journal_to_report_with_billing_reconciliation(self) -> None:
        from tools.modal_v2d import bench

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            tracker, _ = make_tracker(journal_dir=str(raw))
            tracker.meta.update({"app_id": "ap-1", "run_roots": ["/e2e/runs/tissue_box"]})
            tracker.begin(["reconstruct.run"])
            tracker.line("[RUN] reconstruct.run\n")
            tracker.call_start("v2d_moge", "c1", ["python"], "L40S", 8, 32)
            tracker.call_end(0, 100.0, {"remote_s": 90.0, "exec_s": 70.0, "startup_s": 15.0, "push_s": 5.0, "pull_s": 5.0, "reload_s": 1.0, "commit_s": 2.0, "push_bytes": 2_000_000, "gpu_util_mean": 12.0, "gpu_util_max": 40.0, "gpu_mem_max_mib": 9000.0})
            tracker.line("[DONE] reconstruct.run\n")
            tracker.finish(0)

            invocations = bench.load_invocations(raw)
            self.assertTrue(bench.belongs_to(invocations["inv1"], "tissue_box"))
            self.assertFalse(bench.belongs_to(invocations["inv1"], "other"))
            billing = [
                {"object_id": "ap-1", "app": "v2d-e2e", "interval_start": "x", "cost": 0.10, "v2d_invocation": "inv1", "v2d_command": "reconstruct", "v2d_role": "control"},
                {"object_id": "ap-9", "app": "v2d-e2e-build", "interval_start": "x", "cost": 0.02, "v2d_invocation": "", "v2d_command": "", "v2d_role": ""},
            ]
            tables = bench.build_tables(invocations, billing, None)
            inv = tables["invocations"][0]
            self.assertEqual(inv["billed_cost"], 0.1)
            self.assertAlmostEqual(tables["calls"][0]["cost_est"], 90 / 3600 * progress.hourly_rate("L40S", 8, 32), places=5)
            self.assertEqual(tables["unattributed"], {"image builds": 0.02})
            stage = tables["stages"][0]
            self.assertEqual(stage["module_calls"], 1)
            self.assertNotEqual(stage["cost_calibrated"], "")
            summary = bench.summarize(tables, None, 0.12)
            self.assertEqual(summary["steps"][0]["command"], "reconstruct")
            out = Path(tmp) / "out"
            out.mkdir()
            bench.write_report(out, "tissue_box", tables, summary, [])
            report = (out / "report.md").read_text()
            self.assertIn("## Per stage", report)
            self.assertIn("v2d_moge", report)
            self.assertIn("a smaller GPU may do", report)  # 12% mean util over 60 s
            self.assertIsNone(bench.write_charts(out, tables, summary))
            self.assertGreater((out / "charts.pdf").stat().st_size, 1000)

            unknown = bench.build_tables(invocations, [], "PermissionError: nope")
            self.assertEqual(unknown["invocations"][0]["billed_cost"], "UNKNOWN")
