"""utils.tb_run_paths：run 目录与 tbs/ 发现。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.tb_run_paths import discover_tbs_dir, resolve_run_dir  # noqa: E402


class TestResolveRunDir(unittest.TestCase):
    def test_from_generated_rollout_dumps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run_x"
            dumps = run / "generated" / "rollout_dumps"
            dumps.mkdir(parents=True)
            (run / "tbs").mkdir()
            resolved = resolve_run_dir(str(dumps))
            self.assertEqual(resolved, run)

    def test_from_rollout_dumps_without_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run_y"
            dumps = run / "rollout_dumps"
            dumps.mkdir(parents=True)
            (run / "logs").mkdir()
            resolved = resolve_run_dir(str(dumps))
            self.assertEqual(resolved, run)

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(resolve_run_dir(""))
        self.assertIsNone(resolve_run_dir(None))


class TestDiscoverTbsDir(unittest.TestCase):
    def test_requires_event_or_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            tbs = run / "tbs"
            tbs.mkdir(parents=True)
            self.assertIsNone(discover_tbs_dir(run))
            (tbs / "events.out.tfevents.1").write_text("x")
            self.assertEqual(discover_tbs_dir(run), tbs)

    def test_falls_back_to_tf_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            tf_logs = run / "tf_logs"
            tf_logs.mkdir(parents=True)
            (tf_logs / "events.out.tfevents.9").write_text("x")
            self.assertEqual(discover_tbs_dir(run), tf_logs)

    def test_prefers_tbs_over_tf_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            tbs = run / "tbs"
            tf_logs = run / "tf_logs"
            tbs.mkdir(parents=True)
            tf_logs.mkdir(parents=True)
            (tbs / "events.out.tfevents.1").write_text("x")
            (tf_logs / "events.out.tfevents.2").write_text("y")
            self.assertEqual(discover_tbs_dir(run), tbs)

    def test_resolve_run_dir_via_tf_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run_z"
            dumps = run / "generated" / "rollout_dumps"
            dumps.mkdir(parents=True)
            tf_logs = run / "tf_logs"
            tf_logs.mkdir()
            (tf_logs / "events.out.tfevents.1").write_text("x")
            self.assertEqual(resolve_run_dir(str(dumps)), run)


if __name__ == "__main__":
    unittest.main()
