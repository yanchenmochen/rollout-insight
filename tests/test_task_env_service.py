#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""service.task_env_service 单测（mock DB + env 加载）。"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("ROLLOUT_INSIGHT_CONFIG_PATH", "")

from service.task_env_service import (  # noqa: E402
    build_task_env_compare,
    build_task_env_view,
    load_task_env_snapshot,
)


class _FakeTask:
    def __init__(self, primary_id: int, name: str, root: str) -> None:
        self.primary_id = primary_id
        self.task_name = name
        self.rollout_root = root
        self.nanhu_job_id = ""


def _write_resolved(run_dir: Path, env: dict) -> None:
    import yaml

    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({"env": env, "launcher": "musa_run/run_demo.sh"}),
        encoding="utf-8",
    )


class TestTaskEnvService(unittest.TestCase):
    def test_snapshot_from_resolved_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            dumps = run_dir / "generated" / "rollout_dumps"
            dumps.mkdir(parents=True)
            _write_resolved(
                run_dir,
                {
                    "ROLLOUT_BATCH_SIZE": "16",
                    "SWE_AGENT_TIME_BUDGET_SEC": "240",
                    "DEBUG_ROLLOUT_ONLY": "0",
                    "TRAINING_SCRIPT": "musa_run/run_demo.sh",
                },
            )
            task = _FakeTask(7, "demo-task", str(dumps))
            with mock.patch(
                "db.db_task_service.DBTaskService.get_by_id", return_value=task
            ):
                snap = load_task_env_snapshot(7, fetch_nanhu_env=False)
            self.assertIsNotNone(snap)
            assert snap is not None
            self.assertEqual(snap["task_id"], 7)
            self.assertGreaterEqual(snap["highlight_count"], 3)
            keys = {r["catalog_key"] for r in snap["highlight"]}
            self.assertIn("ROLLOUT_BATCH_SIZE", keys)
            self.assertIn("SWE_AGENT_TIME_BUDGET_SEC", keys)
            self.assertEqual(snap["env_source"], "resolved_config.yaml")

    def test_view_missing_task_raises(self) -> None:
        from data.rollout_insight_exception import RolloutInsightException

        with mock.patch(
            "db.db_task_service.DBTaskService.get_by_id", return_value=None
        ):
            with self.assertRaises(RolloutInsightException):
                build_task_env_view(99999, fetch_nanhu_env=False)

    def test_compare_two_tasks_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            t1_run = root / "t1"
            t2_run = root / "t2"
            for run in (t1_run, t2_run):
                (run / "generated" / "rollout_dumps").mkdir(parents=True)
            _write_resolved(
                t1_run,
                {
                    "ROLLOUT_BATCH_SIZE": "16",
                    "SWE_AGENT_TIME_BUDGET_SEC": "300",
                    "DEBUG_ROLLOUT_ONLY": "0",
                    "TRAINING_SCRIPT": "musa_run/run_a.sh",
                },
            )
            _write_resolved(
                t2_run,
                {
                    "ROLLOUT_BATCH_SIZE": "32",
                    "SWE_AGENT_TIME_BUDGET_SEC": "300",
                    "DEBUG_ROLLOUT_ONLY": "0",
                    "TRAINING_SCRIPT": "musa_run/run_b.sh",
                    "SWE_DAPO_MIN_SURVIVING_ATTEMPTS": "4",
                },
            )
            tasks = {
                1: _FakeTask(1, "task-a", str(t1_run / "generated" / "rollout_dumps")),
                2: _FakeTask(2, "task-b", str(t2_run / "generated" / "rollout_dumps")),
            }

            def _get(tid: int):
                return tasks.get(int(tid))

            with mock.patch(
                "db.db_task_service.DBTaskService.get_by_id", side_effect=_get
            ):
                out = build_task_env_compare(
                    [1, 2],
                    baseline_task_id=1,
                    fetch_nanhu_env=False,
                    only_diff=True,
                )
            self.assertTrue(out["ok"])
            self.assertEqual(len(out["tasks"]), 2)
            diff_keys = {r["catalog_key"] for r in out["compare"]["diff_rows"]}
            self.assertIn("ROLLOUT_BATCH_SIZE", diff_keys)
            self.assertIn("TRAINING_SCRIPT", diff_keys)
            # 相同的预算不应出现在 only_diff rows
            row_keys = {r["catalog_key"] for r in out["compare"]["rows"]}
            self.assertNotIn("SWE_AGENT_TIME_BUDGET_SEC", row_keys)


if __name__ == "__main__":
    unittest.main()
