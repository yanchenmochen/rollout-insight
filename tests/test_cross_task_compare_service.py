"""跨任务对比：共同题目、epoch 对齐、变好/变坏判定、奖励指引。"""
from __future__ import annotations

import unittest
from unittest import mock

from service.cross_task_compare_service import (
    _build_instance_rows,
    _classify_pair_delta,
    _epoch_curve,
    _pooled_rate,
    _sort_instance_rows,
    _task_summary,
    build_cross_task_compare,
)
from service.problem_analysis_service import summarize_pass_visits


def _row(rid, gi, cat, instance_id):
    """构造一行 attempt 结局（字段名与 DBAttemptService 返回一致）。"""
    return {
        "attempt_pk": rid * 100 + gi,
        "rollout_id": rid,
        "group_index": gi,
        "attempt_id": 0,
        "instance_id": instance_id,
        "category": cat,
        "reward": 1.0 if cat == "pass" else 0.0,
        "tool_calls": 1,
        "agent_turns": 1,
        "diff_line_count": 1,
    }


class TestPooledRate(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(_pooled_rate([]))

    def test_pooled_uses_total_counts(self):
        visits = [
            {"n_pass": 1, "n_attempts": 2},
            {"n_pass": 2, "n_attempts": 2},
        ]
        self.assertAlmostEqual(_pooled_rate(visits), 3 / 4)


class TestClassifyPairDelta(unittest.TestCase):
    def test_signs(self):
        self.assertEqual(_classify_pair_delta(0.2), "improved")
        self.assertEqual(_classify_pair_delta(-0.2), "worsened")
        self.assertEqual(_classify_pair_delta(0.0), "unchanged")
        self.assertEqual(_classify_pair_delta(None), "unknown")


class TestEpochCurve(unittest.TestCase):
    def test_aligns_by_appearance_index(self):
        per_task = {
            1: {"a": [{"epoch": 1, "n_pass": 1, "n_attempts": 2, "rate": 0.5}]},
            2: {"a": [{"epoch": 1, "n_pass": 0, "n_attempts": 2, "rate": 0.0}]},
        }
        curve = _epoch_curve(per_task, [1, 2], ["a"])
        self.assertEqual(len(curve), 1)
        self.assertAlmostEqual(curve[0]["1"], 0.5)
        self.assertAlmostEqual(curve[0]["2"], 0.0)
        self.assertEqual(curve[0]["1_n_problems"], 1)

    def test_skips_epochs_not_yet_visited(self):
        per_task = {
            1: {
                "a": [
                    {"epoch": 1, "n_pass": 1, "n_attempts": 1, "rate": 1.0},
                    {"epoch": 2, "n_pass": 0, "n_attempts": 1, "rate": 0.0},
                ]
            },
            2: {"a": [{"epoch": 1, "n_pass": 1, "n_attempts": 1, "rate": 1.0}]},
        }
        curve = _epoch_curve(per_task, [1, 2], ["a"])
        self.assertEqual(len(curve), 2)
        # 第 2 个 epoch 任务 2 还没出现该题 → 不计入
        self.assertEqual(curve[1]["2_n_problems"], 0)
        self.assertIsNone(curve[1]["2"])


class TestTaskSummary(unittest.TestCase):
    def test_counts_improved_and_worsened(self):
        visits = {
            "good": [
                {"epoch": 1, "n_pass": 0, "n_attempts": 1, "rate": 0.0},
                {"epoch": 2, "n_pass": 1, "n_attempts": 1, "rate": 1.0},
            ],
            "bad": [
                {"epoch": 1, "n_pass": 1, "n_attempts": 1, "rate": 1.0},
                {"epoch": 2, "n_pass": 0, "n_attempts": 1, "rate": 0.0},
            ],
            "flat": [
                {"epoch": 1, "n_pass": 1, "n_attempts": 1, "rate": 1.0},
                {"epoch": 2, "n_pass": 1, "n_attempts": 1, "rate": 1.0},
            ],
        }
        summary = _task_summary(1, visits, ["good", "bad", "flat"])
        self.assertEqual(summary["n_common"], 3)
        self.assertEqual(summary["n_improved"], 1)
        self.assertEqual(summary["n_worsened"], 1)
        self.assertEqual(summary["n_unchanged"], 1)


class TestInstanceRows(unittest.TestCase):
    def test_vs_baseline_uses_last_rate(self):
        per_task = {
            1: {"a": [{"rate": 0.0, "n_pass": 0, "n_attempts": 1},
                      {"rate": 1.0, "n_pass": 1, "n_attempts": 1}]},
            2: {"a": [{"rate": 0.0, "n_pass": 0, "n_attempts": 1},
                      {"rate": 0.0, "n_pass": 0, "n_attempts": 1}]},
        }
        rows = _build_instance_rows(per_task, [1, 2], ["a"], 1)
        self.assertAlmostEqual(rows[0]["tasks"]["2"]["vs_baseline"], -1.0)

    def test_sort_puts_largest_gap_first(self):
        per_task = {
            1: {"a": [{"rate": 0.0, "n_pass": 0, "n_attempts": 1},
                      {"rate": 1.0, "n_pass": 1, "n_attempts": 1}],
                "b": [{"rate": 1.0, "n_pass": 1, "n_attempts": 1},
                      {"rate": 1.0, "n_pass": 1, "n_attempts": 1}]},
            2: {"a": [{"rate": 0.0, "n_pass": 0, "n_attempts": 1},
                      {"rate": 0.0, "n_pass": 0, "n_attempts": 1}],
                "b": [{"rate": 1.0, "n_pass": 1, "n_attempts": 1},
                      {"rate": 0.9, "n_pass": 0, "n_attempts": 1}]},
        }
        rows = _sort_instance_rows(_build_instance_rows(per_task, [1, 2], ["a", "b"], 1), [1, 2])
        self.assertEqual(rows[0]["instance_id"], "a")


class TestBuildCrossTaskCompare(unittest.TestCase):
    """用 mock 打桩 DB 读取，验证「共同题目 + 变好/变坏」主流程。"""

    def _patch_db(self, rows_by_task, tasks=(1, 2)):
        rows_payload = dict(rows_by_task)

        def fake_meta(task_id):
            return {
                "task_id": task_id,
                "task_name": f"task-{task_id}",
                "status": "running",
                "rollout_root": "/tmp/x",
                "nanhu_job_id": "",
            }

        def fake_visits(task_id):
            trend = summarize_pass_visits(rows_payload.get(task_id, []))
            visits = {}
            for item in trend.get("tasks") or []:
                visits[item["instance_id"]] = [
                    {
                        "epoch": idx + 1,
                        "rollout_id": v["rollout_id"],
                        "n_pass": v["n_pass"],
                        "n_attempts": v["n_attempts"],
                        "rate": v["rate"],
                    }
                    for idx, v in enumerate(item.get("visits") or [])
                ]
            return visits

        return (
            mock.patch("service.cross_task_compare_service._load_task_meta", fake_meta),
            mock.patch("service.cross_task_compare_service._load_visits", fake_visits),
            mock.patch(
                "service.cross_task_compare_service.list_compareable_tasks",
                return_value=[{"task_id": 1, "task_name": "task-1"}],
            ),
            mock.patch(
                "service.cross_task_compare_service.summarize_reward_guidance",
                return_value={"task_id": 1, "levers": []},
            ),
        )

    def test_requires_two_tasks(self):
        from data.rollout_insight_exception import RolloutInsightException

        with self.assertRaises(RolloutInsightException):
            build_cross_task_compare([1])

    def test_common_instances_and_trend_split(self):
        rows = {
            # 任务 1（基线）：题目 a 变好（fail→pass）、b 变坏（pass→fail）
            1: [
                _row(1, 0, "fail", "a"),
                _row(1, 1, "pass", "b"),
                _row(2, 0, "pass", "a"),
                _row(2, 1, "fail", "b"),
            ],
            # 任务 2：a 没变化（fail→fail）、b 变坏（pass→fail）
            2: [
                _row(1, 0, "fail", "a"),
                _row(1, 1, "pass", "b"),
                _row(2, 0, "fail", "a"),
                _row(2, 1, "fail", "b"),
            ],
        }
        patches = self._patch_db(rows)
        with patches[0], patches[1], patches[2], patches[3]:
            data = build_cross_task_compare([1, 2])

        self.assertEqual(data["n_common"], 2)
        self.assertIn("a", data["common_instances"])
        self.assertIn("b", data["common_instances"])
        # 自身训练轨迹：任务 1 的 a 变好、b 变坏
        self.assertEqual(data["buckets"]["1"]["improved"], 1)
        self.assertEqual(data["buckets"]["1"]["worsened"], 1)
        # 任务 2：a 没变化、b 变坏（自身轨迹）
        self.assertEqual(data["buckets"]["2"]["improved"], 0)
        self.assertEqual(data["buckets"]["2"]["worsened"], 1)
        self.assertEqual(data["buckets"]["2"]["unchanged"], 1)
        # 相对基线：任务 2 的 a 末次通过率 0 < 基线 1 → 变坏 1 题；b 与基线齐平
        self.assertEqual(data["buckets_vs_baseline"]["2"]["worsened"], 1)
        self.assertEqual(data["buckets_vs_baseline"]["2"]["unchanged"], 1)
        self.assertEqual(data["baseline_task_id"], 1)

    def test_focus_instance_filters_rows(self):
        rows = {
            1: [_row(1, 0, "fail", "a"), _row(2, 0, "pass", "a")],
            2: [_row(1, 0, "fail", "a"), _row(2, 0, "pass", "a")],
        }
        patches = self._patch_db(rows)
        with patches[0], patches[1], patches[2], patches[3]:
            data = build_cross_task_compare([1, 2], instance_id="a")
        self.assertEqual(data["focus_instance"], "a")
        self.assertEqual(len(data["focus_rows"]), 1)
        self.assertEqual(data["focus_rows"][0]["instance_id"], "a")

    def test_explicit_baseline_is_used(self):
        """用户主动指定的基线任务要生效。"""
        rows = {
            1: [_row(1, 0, "fail", "a"), _row(2, 0, "pass", "a")],
            2: [_row(1, 0, "pass", "a"), _row(2, 0, "fail", "a")],
        }
        patches = self._patch_db(rows)
        with patches[0], patches[1], patches[2], patches[3]:
            data = build_cross_task_compare([1, 2], baseline_task_id=2)
        self.assertEqual(data["baseline_task_id"], 2)
        self.assertEqual(data["baseline_task_name"], "task-2")
        # 基线是任务 2（末次通过率 0），任务 1 末次通过率 1 → 相对基线 +1
        row = data["instances"][0]
        self.assertAlmostEqual(row["tasks"]["1"]["vs_baseline"], 1.0)
        self.assertAlmostEqual(row["tasks"]["2"]["vs_baseline"], 0.0)

    def test_invalid_baseline_falls_back_to_first(self):
        """非法基线 id 回退到第一个任务。"""
        rows = {
            1: [_row(1, 0, "fail", "a"), _row(2, 0, "pass", "a")],
            2: [_row(1, 0, "fail", "a"), _row(2, 0, "pass", "a")],
        }
        patches = self._patch_db(rows)
        with patches[0], patches[1], patches[2], patches[3]:
            data = build_cross_task_compare([1, 2], baseline_task_id=999)
        self.assertEqual(data["baseline_task_id"], 1)

    def test_baseline_outside_task_list_falls_back_to_first(self):
        """基线不在参与对比的任务里时回退到第一个任务。"""
        rows = {
            1: [_row(1, 0, "fail", "a"), _row(2, 0, "pass", "a")],
            2: [_row(1, 0, "fail", "a"), _row(2, 0, "pass", "a")],
        }
        patches = self._patch_db(rows)
        with patches[0], patches[1], patches[2], patches[3]:
            data = build_cross_task_compare([1, 2], baseline_task_id=7)
        self.assertEqual(data["baseline_task_id"], 1)

    def test_instance_rows_include_pass_counts(self):
        """逐次出现的通过数量（n_pass / n_attempts）要带给前端。"""
        rows = {
            1: [
                _row(1, 0, "pass", "a"),
                _row(1, 1, "fail", "a"),
                _row(2, 0, "pass", "a"),
            ],
            2: [_row(1, 0, "fail", "a"), _row(2, 0, "pass", "a")],
        }
        patches = self._patch_db(rows)
        with patches[0], patches[1], patches[2], patches[3]:
            data = build_cross_task_compare([1, 2])
        row = data["instances"][0]
        visits = row["tasks"]["1"]["visits"]
        self.assertEqual(len(visits), 2)
        # 第一次出现：1 次通过 / 2 次尝试
        self.assertEqual(visits[0]["n_pass"], 1)
        self.assertEqual(visits[0]["n_attempts"], 2)
        self.assertEqual(visits[0]["epoch"], 1)
        # 第二次出现：1 次通过 / 1 次尝试
        self.assertEqual(visits[1]["n_pass"], 1)
        self.assertEqual(visits[1]["n_attempts"], 1)
        self.assertEqual(visits[1]["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
