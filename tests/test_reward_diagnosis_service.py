"""奖励诊断：有效梯度组、奖励分解、行为卡片。"""
from __future__ import annotations

import unittest

from service.reward_diagnosis_service import (
    aggregate_attempt_turn_flags,
    classify_behavior_primary,
    classify_group_kind,
    summarize_behavior_cards,
    summarize_gradient_groups,
    summarize_reward_split,
)


def _row(
    rid,
    gi,
    cat,
    *,
    attempt_pk=1,
    reward=None,
    tool_calls=4,
    agent_turns=4,
    diff_line_count=20,
    instance_id="alpha",
):
    if reward is None:
        if cat == "pass":
            reward = 1.0
        elif cat in {"fail", "timeout"}:
            reward = 0.0
        else:
            reward = None
    return {
        "attempt_pk": attempt_pk,
        "rollout_id": rid,
        "group_index": gi,
        "attempt_id": 0,
        "instance_id": instance_id,
        "category": cat,
        "reward": reward,
        "tool_calls": tool_calls,
        "agent_turns": agent_turns,
        "diff_line_count": diff_line_count,
    }


class TestClassifyGroupKind(unittest.TestCase):
    def test_four_kinds(self):
        self.assertEqual(classify_group_kind(["pass", "fail"]), "mixed")
        self.assertEqual(classify_group_kind(["pass", "pass", "abort"]), "all_pass")
        self.assertEqual(classify_group_kind(["fail", "timeout"]), "all_fail")
        self.assertEqual(classify_group_kind(["abort", "timeout"]), "infra_only")


class TestGradientGroups(unittest.TestCase):
    def test_series_and_dip(self):
        rows = []
        # rollout 0: 1 mixed + 1 all_pass → learnable 0.5
        rows += [_row(0, 0, "pass", attempt_pk=1), _row(0, 0, "fail", attempt_pk=2)]
        rows += [_row(0, 1, "pass", attempt_pk=3), _row(0, 1, "pass", attempt_pk=4)]
        # rollout 1: 2 all_fail → learnable 0.0，骤降
        rows += [_row(1, 0, "fail", attempt_pk=5), _row(1, 0, "fail", attempt_pk=6)]
        rows += [_row(1, 1, "abort", attempt_pk=7), _row(1, 1, "timeout", attempt_pk=8)]

        out = summarize_gradient_groups(rows)
        self.assertEqual(out["n_rollouts"], 2)
        self.assertEqual(out["series"][0]["n_mixed"], 1)
        self.assertEqual(out["series"][0]["n_all_pass"], 1)
        self.assertAlmostEqual(out["series"][0]["learnable_ratio"], 0.5)
        self.assertEqual(out["series"][1]["n_all_fail"], 1)
        self.assertEqual(out["series"][1]["n_infra_only"], 1)
        self.assertAlmostEqual(out["series"][1]["learnable_ratio"], 0.0)
        self.assertEqual(len(out["dips"]), 1)
        self.assertEqual(out["dips"][0]["rollout_id"], 1)
        self.assertAlmostEqual(out["dips"][0]["drop"], 0.5)


class TestRewardSplit(unittest.TestCase):
    def test_raw_vs_measurable(self):
        rows = [
            _row(0, 0, "pass", attempt_pk=1, reward=1.0),
            _row(0, 0, "fail", attempt_pk=2, reward=0.0),
            _row(0, 0, "timeout", attempt_pk=3, reward=0.0),
            _row(0, 0, "abort", attempt_pk=4, reward=None),
        ]
        out = summarize_reward_split(rows)
        point = out["series"][0]
        self.assertEqual(point["n_attempts"], 4)
        self.assertAlmostEqual(point["raw_mean_reward"], (1.0 + 0.0 + 0.0) / 3)
        self.assertAlmostEqual(point["measurable_pass_rate"], 0.5)
        self.assertAlmostEqual(point["timeout_ratio"], 0.25)
        self.assertAlmostEqual(point["abort_ratio"], 0.25)


class TestBehaviorCards(unittest.TestCase):
    def test_priority_and_examples(self):
        rows = [
            _row(0, 0, "abort", attempt_pk=1),
            _row(0, 0, "timeout", attempt_pk=2),
            _row(0, 0, "fail", attempt_pk=3, tool_calls=1, diff_line_count=0),
            _row(0, 0, "fail", attempt_pk=4, tool_calls=5, diff_line_count=30),
            _row(0, 0, "fail", attempt_pk=5, tool_calls=1, agent_turns=1, diff_line_count=0),
            _row(0, 0, "pass", attempt_pk=6),
        ]
        flags = [
            {
                "attempt_pk": 3,
                "rollout_id": 0,
                "segment_index": 0,
                "turn_index": 0,
                "has_tool_call": False,
                "tool_call_names": [],
            },
            {
                "attempt_pk": 4,
                "rollout_id": 0,
                "segment_index": 0,
                "turn_index": 0,
                "has_tool_call": True,
                "tool_call_names": ["execute_bash"],
            },
            {
                "attempt_pk": 5,
                "rollout_id": 0,
                "segment_index": 0,
                "turn_index": 0,
                "has_tool_call": True,
                "tool_call_names": ["finish"],
            },
        ]
        split = summarize_reward_split(rows)
        out = summarize_behavior_cards(rows, flags, reward_split_series=split["series"])
        by_key = {card["key"]: card for card in out["cards"]}
        self.assertEqual(out["n_failed"], 5)
        self.assertEqual(by_key["abort"]["n_attempts"], 1)
        self.assertEqual(by_key["timeout"]["n_attempts"], 1)
        self.assertEqual(by_key["no_tool_call"]["n_attempts"], 1)
        self.assertEqual(by_key["last_not_finish"]["n_attempts"], 1)
        self.assertEqual(by_key["early_finish"]["n_attempts"], 1)
        self.assertEqual(
            by_key["abort"]["examples"][0]["detail_url"],
            "/api/failures/attempts/1",
        )
        self.assertIn("lever", by_key["timeout"])

    def test_classify_primary_pass_is_none(self):
        self.assertIsNone(classify_behavior_primary(_row(0, 0, "pass", attempt_pk=1)))

    def test_aggregate_turn_flags(self):
        flags = [
            {
                "attempt_pk": 9,
                "rollout_id": 1,
                "segment_index": 0,
                "turn_index": 0,
                "has_tool_call": True,
                "tool_call_names": ["think"],
            },
            {
                "attempt_pk": 9,
                "rollout_id": 1,
                "segment_index": 0,
                "turn_index": 1,
                "has_tool_call": True,
                "tool_call_names": ["finish"],
            },
        ]
        out = aggregate_attempt_turn_flags(flags)
        self.assertTrue(out[9]["last_is_finish"])
        self.assertFalse(out[9]["any_without_tool_call"])


if __name__ == "__main__":
    unittest.main()
