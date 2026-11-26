"""自动重扫：按作业状态和 rollout 落盘间隔决定要不要扫。"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from service.rollout_auto_scan_service import (
    cadence_from_mtimes,
    decide_autoscan,
    run_autoscan_tick,
    worth_watching,
)


class TestCadence(unittest.TestCase):
    def test_median_of_recent_gaps(self) -> None:
        base = 1_700_000_000.0
        mtimes = [base + i * 100 for i in range(6)]
        self.assertEqual(cadence_from_mtimes(mtimes), 100.0)

    def test_clamps_fast_and_slow(self) -> None:
        base = 1_700_000_000.0
        self.assertEqual(cadence_from_mtimes([base, base + 10, base + 20]), 45.0)
        self.assertEqual(
            cadence_from_mtimes([base, base + 5000]),
            900.0,
        )

    def test_default_when_unknown(self) -> None:
        self.assertEqual(cadence_from_mtimes([]), 120.0)
        self.assertEqual(cadence_from_mtimes([1_700_000_000.0]), 120.0)


class TestDecide(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 11, 25, 17, 10, 0)
        self.base = self.now.timestamp() - 400

    def test_running_waits_for_cadence(self) -> None:
        mtimes = [self.base, self.base + 100, self.base + 200]
        decision = decide_autoscan(
            job_status="RUNNING",
            task_status="READY",
            file_mtimes=mtimes,
            last_scanned_at=self.now - timedelta(seconds=30),
            now=self.now,
        )
        self.assertFalse(decision["due"])
        self.assertEqual(decision["reason"], "wait_cadence")
        self.assertTrue(decision["producing"])

    def test_running_due_on_cadence(self) -> None:
        mtimes = [self.base, self.base + 100, self.base + 200]
        decision = decide_autoscan(
            job_status="RUNNING",
            task_status="READY",
            file_mtimes=mtimes,
            last_scanned_at=self.now - timedelta(seconds=120),
            now=self.now,
        )
        self.assertTrue(decision["due"])
        self.assertEqual(decision["reason"], "cadence")

    def test_new_file_scans_after_settle(self) -> None:
        newest = self.now.timestamp() - 30
        mtimes = [newest - 200, newest - 100, newest]
        decision = decide_autoscan(
            job_status="RUNNING",
            task_status="READY",
            file_mtimes=mtimes,
            last_scanned_at=self.now - timedelta(seconds=80),
            now=self.now,
        )
        self.assertTrue(decision["due"])
        self.assertEqual(decision["reason"], "new_rollout")

    def test_fresh_file_waits_to_settle(self) -> None:
        newest = self.now.timestamp() - 3
        decision = decide_autoscan(
            job_status="RUNNING",
            task_status="READY",
            file_mtimes=[newest - 100, newest],
            last_scanned_at=self.now - timedelta(seconds=50),
            now=self.now,
        )
        self.assertFalse(decision["due"])
        self.assertEqual(decision["reason"], "file_settling")

    def test_finished_job_without_new_files_is_idle(self) -> None:
        mtimes = [self.base, self.base + 100]
        decision = decide_autoscan(
            job_status="SUCCEEDED",
            task_status="READY",
            file_mtimes=mtimes,
            last_scanned_at=datetime.fromtimestamp(self.base + 100),
            now=self.now,
        )
        self.assertFalse(decision["due"])
        self.assertEqual(decision["reason"], "not_running")
        self.assertFalse(decision["producing"])

    def test_stale_success_still_producing_follows_rhythm(self) -> None:
        newest = self.now.timestamp() - 30
        decision = decide_autoscan(
            job_status="SUCCEEDED",
            task_status="READY",
            file_mtimes=[newest - 100, newest],
            last_scanned_at=self.now - timedelta(seconds=90),
            now=self.now,
        )
        self.assertTrue(decision["producing"])
        self.assertTrue(decision["due"])
        self.assertEqual(decision["reason"], "new_rollout")

    def test_scanning_is_not_due(self) -> None:
        decision = decide_autoscan(
            job_status="RUNNING",
            task_status="SCANNING",
            file_mtimes=[self.base, self.base + 100],
            last_scanned_at=None,
            now=self.now,
        )
        self.assertFalse(decision["due"])
        self.assertEqual(decision["reason"], "scanning")


class TestTick(unittest.TestCase):
    def test_scans_only_the_due_running_task(self) -> None:
        now = datetime(2026, 11, 25, 17, 10, 0)
        newest = now.timestamp() - 40
        scanned = []
        candidates = [
            {
                "task_id": 1,
                "task_name": "done",
                "task_status": "READY",
                "job_status": "SUCCEEDED",
                "last_scanned_at": now - timedelta(hours=8),
            },
            {
                "task_id": 2,
                "task_name": "live",
                "task_status": "READY",
                "job_status": "RUNNING",
                "last_scanned_at": now - timedelta(seconds=80),
                "rollout_root": "/tmp/live",
                "recursive": False,
            },
        ]

        class _Path:
            def __init__(self, mtime: float) -> None:
                self._mtime = mtime

            def stat(self):
                return type("St", (), {"st_mtime": self._mtime})()

        def resolve_live(item):
            if item["task_id"] != 2:
                return []
            return [_Path(newest - 100), _Path(newest)]

        actions = run_autoscan_tick(
            now=now,
            candidates=candidates,
            resolve_paths=resolve_live,
            scan_one=lambda task_id: scanned.append(task_id) or {"busy": False, "scanned": 1},
        )
        self.assertEqual(scanned, [2])
        self.assertEqual(actions[0]["reason"], "new_rollout")
        self.assertFalse(worth_watching(candidates[0], now))
        self.assertTrue(worth_watching(candidates[1], now))


if __name__ == "__main__":
    unittest.main()
