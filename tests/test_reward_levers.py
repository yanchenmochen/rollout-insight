"""奖励上升杠杆：有效梯度组 / 可测通过率 / timeout / abort / 原始奖励。"""
from __future__ import annotations

import unittest

from service.reward_diagnosis_service import build_reward_levers


def _row(rid, gi, cat, reward=None, instance_id="alpha"):
    if reward is None:
        if cat == "pass":
            reward = 1.0
        elif cat in {"fail", "timeout"}:
            reward = 0.0
        else:
            reward = None
    return {
        "attempt_pk": rid * 100 + gi,
        "rollout_id": rid,
        "group_index": gi,
        "attempt_id": 0,
        "instance_id": instance_id,
        "category": cat,
        "reward": reward,
        "tool_calls": 1,
        "agent_turns": 1,
        "diff_line_count": 1,
    }


class TestBuildRewardLevers(unittest.TestCase):
    def test_always_returns_all_five_levers(self):
        result = build_reward_levers([])
        keys = [item["key"] for item in result["levers"]]
        self.assertEqual(
            set(keys),
            {
                "learnable_ratio",
                "measurable_pass_rate",
                "timeout_ratio",
                "abort_ratio",
                "raw_mean_reward",
            },
        )

    def test_lever_has_phenomenon_and_action(self):
        result = build_reward_levers([_row(1, 0, "pass")])
        for lever in result["levers"]:
            self.assertTrue(lever["title"])
            self.assertTrue(lever["phenomenon"])
            self.assertTrue(lever["action"])

    def test_learnable_ratio_and_mixed_count(self):
        rows = [
            # 组 0：mixed（有对有挂）；组 1：all_pass
            _row(1, 0, "pass"),
            _row(1, 0, "fail"),
            _row(1, 1, "pass"),
            _row(1, 1, "pass"),
        ]
        result = build_reward_levers(rows)
        values = result["values"]
        # 2 个组：一个 mixed、一个 all_pass → learnable_ratio = 0.5
        self.assertAlmostEqual(values["learnable_ratio"], 0.5)
        self.assertEqual(values["n_mixed"], 1)
        self.assertEqual(values["n_groups"], 2)

    def test_low_learnable_ratio_is_bad(self):
        # 只有 all_pass / all_fail 组，没有混合组 → 无学习空间
        rows = [_row(1, 0, "pass"), _row(1, 1, "pass"), _row(1, 2, "fail")]
        result = build_reward_levers(rows)
        lever = next(x for x in result["levers"] if x["key"] == "learnable_ratio")
        self.assertEqual(lever["status"], "bad")

    def test_timeout_ratio_high_is_bad(self):
        rows = [_row(1, 0, "timeout"), _row(1, 1, "timeout"), _row(1, 2, "pass")]
        result = build_reward_levers(rows)
        lever = next(x for x in result["levers"] if x["key"] == "timeout_ratio")
        self.assertEqual(lever["status"], "bad")
        self.assertAlmostEqual(result["values"]["timeout_ratio"], 2 / 3)

    def test_abort_ratio_high_is_bad(self):
        rows = [_row(1, 0, "abort"), _row(1, 1, "abort"), _row(1, 2, "pass")]
        result = build_reward_levers(rows)
        lever = next(x for x in result["levers"] if x["key"] == "abort_ratio")
        self.assertEqual(lever["status"], "bad")

    def test_measurable_pass_rate_excludes_abort_and_timeout(self):
        rows = [
            _row(1, 0, "pass"),
            _row(1, 1, "fail"),
            _row(1, 2, "abort"),
            _row(1, 3, "timeout"),
        ]
        result = build_reward_levers(rows)
        # 分母去掉 abort / timeout → 1 / 2
        self.assertAlmostEqual(result["values"]["measurable_pass_rate"], 0.5)

    def test_raw_mean_reward_below_measurable_is_warn(self):
        # 可测通过率 1.0，但 abort 的 reward 记成 0 把原始平均奖励拉低
        rows = [
            _row(1, 0, "pass", reward=1.0),
            _row(1, 1, "abort", reward=0.0),
        ]
        result = build_reward_levers(rows)
        self.assertAlmostEqual(result["values"]["measurable_pass_rate"], 1.0)
        self.assertAlmostEqual(result["values"]["raw_mean_reward"], 0.5)
        lever = next(x for x in result["levers"] if x["key"] == "raw_mean_reward")
        self.assertEqual(lever["status"], "warn")

    def test_levers_sorted_by_status(self):
        rows = [_row(1, 0, "timeout"), _row(1, 1, "timeout"), _row(1, 2, "pass")]
        result = build_reward_levers(rows)
        order = {"bad": 0, "warn": 1, "unknown": 2, "ok": 3}
        statuses = [order.get(item["status"], 9) for item in result["levers"]]
        self.assertEqual(statuses, sorted(statuses))

    def test_empty_rows_gives_unknown_status(self):
        result = build_reward_levers([])
        for lever in result["levers"]:
            self.assertEqual(lever["status"], "unknown")
            self.assertIsNone(lever["value"])


if __name__ == "__main__":
    unittest.main()
