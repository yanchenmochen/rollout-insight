"""utils.run_train_metrics：run 目录解析与 P0 指标抽取。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.run_train_metrics import (  # noqa: E402
    clear_p0_cache_for_tests,
    load_p0_from_log,
    load_p0_metrics,
    resolve_run_dir,
)


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


class TestLoadP0FromLog(unittest.TestCase):
    def test_parse_perf_and_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "train.log"
            log.write_text(
                "\n".join(
                    [
                        "(RolloutManager) [t] rollout.py:1421 - perf 0: "
                        "{'passrate/pass@1': 0.75, "
                        "'rollout/generated/group_reward/mixed_fraction': 0.5, "
                        "'rollout/removed_ratio': 0.5, "
                        "'rollout/funnel/loss_eligible_attempts': 64.0, "
                        "'rollout/generated/raw_reward': 0.75, "
                        "'rollout/removed/count_zero_std_1.0': 55.0}",
                        "(Actor) [t] data.py:194 - rollout/filtered 0: "
                        "{'rollout/filtered/raw_reward': 0.64, "
                        "'rollout/filtered/shaped_reward': 0.64}",
                        "(RolloutManager) [t] rollout.py:1421 - perf 1: "
                        "{'passrate/pass@1': 0.70, "
                        "'rollout/generated/group_reward/mixed_fraction': 0.4, "
                        "'rollout/removed_ratio': 0.6, "
                        "'rollout/funnel/loss_eligible_attempts': 50.0, "
                        "'rollout/generated/raw_reward': 0.70, "
                        "'rollout/removed/count_zero_std_1.0': 60.0}",
                        "(Actor) [t] data.py:194 - rollout/filtered 1: "
                        "{'rollout/filtered/raw_reward': 0.66, "
                        "'rollout/filtered/shaped_reward': 0.66}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            payload = load_p0_from_log(log)
            self.assertEqual(payload["n_steps"], 2)
            self.assertEqual(payload["passrate"]["pass@1"][0], 0.75)
            self.assertEqual(payload["passrate"]["pass@1"][1], 0.7)
            self.assertEqual(payload["reward"]["filtered_raw"][0], 0.64)
            self.assertEqual(len(payload["removed"]["by_reason"]), 1)
            self.assertEqual(payload["removed"]["by_reason"][0]["reason"], "zero_std_1.0")
            self.assertEqual(payload["removed"]["by_reason"][0]["total"], 115.0)


class TestLoadP0MetricsIntegration(unittest.TestCase):
    def setUp(self) -> None:
        clear_p0_cache_for_tests()

    def tearDown(self) -> None:
        clear_p0_cache_for_tests()

    def test_prefers_tbs_when_present(self) -> None:
        run = Path(
            "/mnt/si003126ndy1/default/shenyichong/code/slime/musa_run/"
            "research_loop/worktrees/c021-32b/musa_run/runs/"
            "20261107_002636_c021-e18-f1-ctx128k"
        )
        if not (run / "tbs").is_dir():
            self.skipTest("sample run tbs/ not available on this host")
        dumps = run / "generated" / "rollout_dumps"
        payload = load_p0_metrics(str(dumps), use_cache=False)
        self.assertEqual(payload["source"], "tbs")
        self.assertGreaterEqual(payload["n_steps"], 1)
        self.assertIn("pass@1", payload["passrate"])
        self.assertTrue(payload["summary"].get("pass_at_1_last") is not None)

    def test_memory_cache_hit_after_compute(self) -> None:
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run_cache"
            dumps = run / "generated" / "rollout_dumps"
            dumps.mkdir(parents=True)
            logs = run / "logs"
            logs.mkdir()
            log = logs / "train.log"
            log.write_text(
                "(RolloutManager) [t] rollout.py:1421 - perf 0: "
                "{'passrate/pass@1': 0.5, 'rollout/generated/raw_reward': 0.5}\n"
                "(RolloutManager) [t] rollout.py:1421 - perf 1: "
                "{'passrate/pass@1': 0.6, 'rollout/generated/raw_reward': 0.6}\n",
                encoding="utf-8",
            )
            with mock.patch(
                "utils.run_train_metrics._schedule_verify_and_refresh"
            ):
                first = load_p0_metrics(str(dumps), use_cache=True)
                self.assertEqual(first["source"], "log")
                self.assertEqual(first["n_steps"], 2)
                self.assertIn(first.get("cache"), {"computed", "sidecar", "memory"})
                second = load_p0_metrics(str(dumps), use_cache=True)
                self.assertEqual(second["cache"], "memory")
                self.assertEqual(second["n_steps"], 2)

    def test_sidecar_returns_before_fingerprint_scan(self) -> None:
        """有 sidecar 时请求路径先返回，指纹核对丢到后台。"""
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run_sidecar_fast"
            dumps = run / "generated" / "rollout_dumps"
            dumps.mkdir(parents=True)
            (run / "logs").mkdir()
            log = run / "logs" / "train.log"
            log.write_text(
                "(RolloutManager) [t] rollout.py:1421 - perf 0: "
                "{'passrate/pass@1': 0.4, 'rollout/generated/raw_reward': 0.4}\n",
                encoding="utf-8",
            )
            with mock.patch(
                "utils.run_train_metrics._schedule_verify_and_refresh"
            ):
                load_p0_metrics(str(dumps), use_cache=True)
            clear_p0_cache_for_tests()

            with mock.patch(
                "utils.run_train_metrics._schedule_verify_and_refresh"
            ) as sched:
                served = load_p0_metrics(str(dumps), use_cache=True, allow_stale=True)
            self.assertEqual(served["n_steps"], 1)
            self.assertIn(served.get("cache"), {"sidecar", "sidecar_stale"})
            self.assertTrue(served.get("refresh_pending"))
            sched.assert_called_once()


if __name__ == "__main__":
    unittest.main()
