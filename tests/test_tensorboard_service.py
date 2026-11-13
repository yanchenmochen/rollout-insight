"""service.tensorboard_service：logdir_spec 与进程复用（mock Popen）。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from service import tensorboard_service as tbsvc  # noqa: E402
from service.tensorboard_service import (  # noqa: E402
    TensorBoardRun,
    build_logdir_spec,
    build_runs_for_tasks,
    public_tensorboard_url,
    start_tensorboard,
    stop_tensorboard,
    tensorboard_status,
)


class TestBuildRuns(unittest.TestCase):
    def test_skips_without_tbs_and_dedupes_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_a = Path(tmp) / "a"
            tbs_a = run_a / "tbs"
            tbs_a.mkdir(parents=True)
            (tbs_a / "events.out.tfevents.1").write_text("x")
            dumps_a = run_a / "generated" / "rollout_dumps"
            dumps_a.mkdir(parents=True)

            run_b = Path(tmp) / "b"
            dumps_b = run_b / "generated" / "rollout_dumps"
            dumps_b.mkdir(parents=True)

            tasks = [
                {
                    "primary_id": 1,
                    "task_name": "same name",
                    "rollout_root": str(dumps_a),
                },
                {
                    "primary_id": 2,
                    "task_name": "same name",
                    "rollout_root": str(dumps_a),
                },
                {
                    "primary_id": 3,
                    "task_name": "no_tbs",
                    "rollout_root": str(dumps_b),
                },
            ]
            runs, skipped = build_runs_for_tasks(tasks)
            self.assertEqual(len(runs), 2)
            self.assertEqual(runs[0].name, "same_name")
            self.assertEqual(runs[1].name, "same_name_2")
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]["task_id"], 3)

            spec = build_logdir_spec(runs)
            self.assertIn("same_name:", spec)
            self.assertIn("same_name_2:", spec)
            self.assertNotIn(",,", spec)


class TestPublicUrl(unittest.TestCase):
    def test_uses_request_hostname(self) -> None:
        self.assertEqual(
            public_tensorboard_url("nanhu.example.com:39998", 6006),
            "http://nanhu.example.com:6006/",
        )
        self.assertEqual(
            public_tensorboard_url(None, 6006),
            "http://127.0.0.1:6006/",
        )


class TestStartReuse(unittest.TestCase):
    def setUp(self) -> None:
        stop_tensorboard()

    def tearDown(self) -> None:
        stop_tensorboard()

    def test_reuse_same_spec(self) -> None:
        runs = [
            TensorBoardRun(task_id=1, name="r1", tbs_dir="/tmp/tbs1", run_dir="/tmp"),
        ]
        fake_proc = mock.Mock()
        fake_proc.poll.return_value = None
        fake_proc.pid = 4242

        with mock.patch.object(tbsvc.shutil, "which", return_value="/usr/bin/tensorboard"):
            with mock.patch.object(tbsvc.subprocess, "Popen", return_value=fake_proc) as popen:
                with mock.patch.object(tbsvc.time, "sleep", return_value=None):
                    first = start_tensorboard(runs, request_host="h:1", port=16006)
                    second = start_tensorboard(runs, request_host="h:1", port=16006)
        self.assertEqual(popen.call_count, 1)
        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])
        self.assertEqual(first["url"], "http://h:16006/")
        st = tensorboard_status()
        self.assertTrue(st["running"])
        self.assertEqual(st["pid"], 4242)


if __name__ == "__main__":
    unittest.main()
