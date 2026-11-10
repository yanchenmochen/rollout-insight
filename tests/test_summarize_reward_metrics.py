"""DBAttemptService.summarize_reward_metrics_* 单元逻辑（纯函数路径）。"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from db.db_attempt_service import DBAttemptService


def _session_patch(rows):
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = rows
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = False
    return patch("db.db_attempt_service.Session", return_value=mock_cm)


class TestSummarizeRewardMetricsByRollout(unittest.TestCase):
    def test_pass_abort_valid_groups(self):
        # rollout 0: 对齐 E18 step0 形态的缩微版
        # g0: 8×1 → flat pass（无效组）
        # g1: 7×0 + 1 abort → abort 后全0（无效组）
        # g2: 4×0 + 4×1 → 有效组
        # g13: 7×1 + 1 abort → abort 后全1（无效组）
        rows = []
        rid = 0
        for _ in range(8):
            rows.append((rid, 0, "completed", 1.0))
        for _ in range(7):
            rows.append((rid, 1, "completed", 0.0))
        rows.append((rid, 1, "aborted", 0.0))
        for _ in range(4):
            rows.append((rid, 2, "completed", 0.0))
        for _ in range(4):
            rows.append((rid, 2, "completed", 1.0))
        for _ in range(7):
            rows.append((rid, 13, "completed", 1.0))
        rows.append((rid, 13, "aborted", 0.0))

        with _session_patch(rows):
            out = DBAttemptService.summarize_reward_metrics_by_rollout(task_id=2)

        self.assertIn(0, out)
        m = out[0]
        self.assertEqual(m["n_passed"], 8 + 4 + 7)  # 19
        self.assertEqual(m["n_aborted"], 2)
        self.assertEqual(m["n_timeout"], 0)
        self.assertEqual(m["n_failed"], 7 + 4)  # 11
        self.assertEqual(m["n_valid_groups"], 1)  # 仅 g2
        self.assertEqual(m["n_attempts"], 8 + 8 + 8 + 8)

    def test_timeout_not_counted_as_fail(self):
        rows = [
            (1, 0, "truncated", 0.0),
            (1, 0, "truncated", 1.0),
            (1, 0, "completed", 0.0),
            (1, 0, "completed", 1.0),
            (1, 0, "aborted", 0.0),
        ]
        with _session_patch(rows):
            out = DBAttemptService.summarize_reward_metrics_by_rollout(task_id=2)
        m = out[1]
        self.assertEqual(m["n_timeout"], 2)
        self.assertEqual(m["n_failed"], 1)
        self.assertEqual(m["n_passed"], 1)
        self.assertEqual(m["n_aborted"], 1)
        # 奖励柱：超时可有奖；abort 计无测
        self.assertEqual(m["n_reward_1"], 2)  # pass + timeout:1
        self.assertEqual(m["n_reward_0"], 2)  # fail + timeout:0
        self.assertEqual(m["n_no_reward"], 1)  # abort
        # 超时边界内拆分（对齐结局柱超时段）
        self.assertEqual(m["n_timeout_rw1"], 1)
        self.assertEqual(m["n_timeout_rw0"], 1)
        self.assertEqual(m["n_timeout_norw"], 0)


class TestSummarizeRewardMetricsByGroup(unittest.TestCase):
    def test_group_metrics(self):
        # query returns (group_index, status, reward, instance_id)
        rows = [
            (10, "completed", 1.0, "inst-a"),
            (10, "completed", 0.0, "inst-a"),
            (10, "aborted", 0.0, "inst-a"),
            (10, "truncated", 0.0, "inst-a"),
            (11, "completed", 1.0, "inst-b"),
            (11, "completed", 1.0, "inst-b"),
        ]
        with _session_patch(rows):
            out = DBAttemptService.summarize_reward_metrics_by_group(
                task_id=2, rollout_id=9
            )
        g10 = out[10]
        self.assertEqual(g10["n_passed"], 1)
        self.assertEqual(g10["n_failed"], 1)
        self.assertEqual(g10["n_aborted"], 1)
        self.assertEqual(g10["n_timeout"], 1)
        self.assertTrue(g10["is_valid"])
        self.assertEqual(g10["instance_id"], "inst-a")
        self.assertAlmostEqual(g10["mean_reward"], 0.5)
        g11 = out[11]
        self.assertEqual(g11["n_passed"], 2)
        self.assertFalse(g11["is_valid"])


if __name__ == "__main__":
    unittest.main()