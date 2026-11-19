"""问题分析：跨 epoch 通过率与 assistant tool_call 计数。"""
from __future__ import annotations

import unittest

from service.problem_analysis_service import (
    _instance_trend,
    _match_cache,
    group_instance_appearances,
    merge_response_stats,
    public_response_stats,
    summarize_pass_visits,
    summarize_response_attempts,
)
from utils.rollout_pt_parser import split_response_to_turns


def _rows(rid, gi, iid, n_pass, n_fail, n_abort=0):
    rows = []
    for _ in range(n_pass):
        rows.append(
            {"rollout_id": rid, "group_index": gi, "instance_id": iid, "category": "pass"}
        )
    for _ in range(n_fail):
        rows.append(
            {"rollout_id": rid, "group_index": gi, "instance_id": iid, "category": "fail"}
        )
    for _ in range(n_abort):
        rows.append(
            {"rollout_id": rid, "group_index": gi, "instance_id": iid, "category": "abort"}
        )
    return rows


class TestPassTrend(unittest.TestCase):
    def test_first_last_delta_and_histogram(self):
        rows = []
        rows += _rows(0, 0, "alpha", 1, 7)
        rows += _rows(1, 0, "alpha", 4, 4)
        rows += _rows(0, 1, "beta", 6, 2)
        rows += _rows(1, 1, "beta", 1, 7)
        rows += _rows(0, 2, "gamma", 0, 8)
        rows += _rows(0, 3, "delta", 2, 7)
        rows += _rows(1, 3, "delta", 2, 7)
        rows += _rows(0, 4, "epsilon", 1, 0, n_abort=1)
        rows += _rows(0, 5, "", 3, 0)

        trend = summarize_pass_visits(rows, batch_size=2, batch_size_source="env")
        by_id = {item["instance_id"]: item for item in trend["tasks"]}

        self.assertEqual(trend["summary"], {"improved": 1, "unchanged": 1, "worsened": 1, "once": 2})
        self.assertEqual(trend["n_attempts_without_instance"], 3)
        self.assertEqual(by_id["alpha"]["trend"], "improved")
        self.assertAlmostEqual(by_id["alpha"]["delta"], 4 / 8 - 1 / 8)
        self.assertEqual(by_id["alpha"]["visits"][0]["n_pass"], 1)
        self.assertEqual(by_id["alpha"]["visits"][0]["n_attempts"], 8)
        self.assertEqual(by_id["beta"]["trend"], "worsened")
        self.assertAlmostEqual(by_id["beta"]["delta"], 1 / 8 - 6 / 8)
        self.assertEqual(by_id["delta"]["trend"], "unchanged")
        self.assertEqual(by_id["delta"]["visits"][0]["n_attempts"], 9)
        self.assertEqual(by_id["gamma"]["trend"], "once")
        self.assertIsNone(by_id["gamma"]["delta"])
        self.assertAlmostEqual(by_id["epsilon"]["first_rate"], 0.5)
        self.assertEqual(trend["histogram"][11]["count"], 1)
        self.assertEqual(trend["histogram"][3]["count"], 1)
        self.assertEqual(trend["histogram"][8]["count"], 1)
        self.assertEqual(sum(item["count"] for item in trend["histogram"]), 3)
        self.assertFalse(trend["steps_per_epoch_exact"])
        self.assertFalse(trend["alignment_checked"])
        series = trend["epoch_series"]
        self.assertEqual([item["epoch"] for item in series], [1, 2])
        self.assertEqual(series[0]["n_problems"], 5)
        self.assertEqual(series[0]["n_pass"], 10)
        self.assertEqual(series[0]["n_attempts"], 35)
        self.assertAlmostEqual(series[0]["rate"], 10 / 35)
        self.assertEqual(series[1]["n_problems"], 3)
        self.assertEqual(series[1]["n_pass"], 7)
        self.assertEqual(series[1]["n_attempts"], 25)
        self.assertAlmostEqual(series[1]["rate"], 7 / 25)

    def test_epoch_window_detects_repeat(self):
        rows = []
        rows += _rows(0, 0, "alpha", 1, 0)
        rows += _rows(1, 0, "alpha", 1, 0)
        rows += _rows(2, 0, "beta", 0, 1)
        rows += _rows(3, 0, "beta", 0, 1)
        trend = summarize_pass_visits(rows, batch_size=1, batch_size_source="env")
        self.assertEqual(trend["steps_per_epoch"], 2)
        self.assertTrue(trend["alignment_checked"])
        self.assertFalse(trend["epoch_aligned"])
        self.assertTrue(any(item["instance_id"] == "alpha" for item in trend["alignment_violations"]))

    def test_epoch_window_aligned_when_each_problem_once(self):
        rows = []
        rows += _rows(0, 0, "alpha", 1, 7)
        rows += _rows(0, 1, "beta", 2, 6)
        rows += _rows(1, 0, "alpha", 3, 5)
        rows += _rows(1, 1, "beta", 4, 4)
        trend = summarize_pass_visits(rows, batch_size=2, batch_size_source="group_mode")
        self.assertEqual(trend["steps_per_epoch"], 1)
        self.assertTrue(trend["epoch_aligned"])
        self.assertEqual(trend["n_epochs_observed"], 2)
        self.assertEqual(trend["n_epochs_from_steps"], 2)
        self.assertEqual(trend["batch_size_source"], "group_mode")

    def test_batch_size_falls_back_to_group_mode(self):
        rows = _rows(0, 0, "alpha", 1, 0) + _rows(0, 1, "beta", 0, 1)
        rows += _rows(1, 0, "alpha", 1, 0) + _rows(1, 1, "beta", 0, 1)
        trend = summarize_pass_visits(rows)
        self.assertEqual(trend["batch_size"], 2)
        self.assertEqual(trend["batch_size_source"], "group_mode")


class TestInstanceAppearances(unittest.TestCase):
    def test_groups_visits_and_keeps_trajectory_ids(self):
        rows = [
            {"rollout_id": 3, "group_index": 1, "instance_id": "repo_pr1", "category": "pass",
             "attempt_pk": 10, "attempt_id": 2, "reward": 1.0},
            {"rollout_id": 3, "group_index": 1, "instance_id": "repo_pr1", "category": "fail",
             "attempt_pk": 11, "attempt_id": 1, "reward": 0.0},
            {"rollout_id": 0, "group_index": 4, "instance_id": "repo_pr1", "category": "pass",
             "attempt_pk": 1, "attempt_id": 7, "reward": 1.0},
            {"rollout_id": 0, "group_index": 0, "instance_id": "other", "category": "fail",
             "attempt_pk": 2, "attempt_id": 1, "reward": 0.0},
        ]
        view = group_instance_appearances(
            "repo_pr1",
            rows,
            rollout_pks={0: 100, 3: 103},
            segment_ids={10: 500},
        )
        self.assertEqual(view["n_visits"], 2)
        self.assertEqual(view["visits"][0]["rollout_id"], 0)
        self.assertEqual(view["visits"][0]["rollout_pk"], 100)
        self.assertEqual(view["visits"][0]["epoch"], 1)
        self.assertEqual(view["visits"][1]["epoch"], 2)
        self.assertEqual(view["visits"][1]["n_pass"], 1)
        self.assertEqual(view["visits"][1]["n_attempts"], 2)
        self.assertEqual(view["visits"][1]["attempts"][0]["attempt_id"], 1)
        self.assertEqual(view["visits"][1]["attempts"][1]["segment_id"], 500)
        self.assertIsNone(view["visits"][0]["attempts"][0]["segment_id"])

    def test_attempt_carries_length_and_turns(self):
        """attempt 行带上响应长度与轮数，供表格列与趋势图使用。"""
        rows = [
            {"rollout_id": 0, "group_index": 0, "instance_id": "repo_pr1", "category": "pass",
             "attempt_pk": 1, "attempt_id": 0, "reward": 1.0,
             "response_length": 1200, "agent_turns": 12, "tool_calls": 10, "diff_line_count": 5},
        ]
        view = group_instance_appearances("repo_pr1", rows)
        attempt = view["visits"][0]["attempts"][0]
        self.assertEqual(attempt["response_length"], 1200)
        self.assertEqual(attempt["agent_turns"], 12)
        self.assertEqual(attempt["tool_calls"], 10)

    def test_visit_metrics_averages(self):
        rows = [
            {"rollout_id": 0, "group_index": 0, "instance_id": "repo_pr1", "category": "pass",
             "attempt_pk": 1, "attempt_id": 0, "reward": 1.0,
             "response_length": 100, "agent_turns": 10, "tool_calls": 8},
            {"rollout_id": 0, "group_index": 0, "instance_id": "repo_pr1", "category": "fail",
             "attempt_pk": 2, "attempt_id": 1, "reward": 0.0,
             "response_length": 300, "agent_turns": 20, "tool_calls": 12},
        ]
        view = group_instance_appearances("repo_pr1", rows)
        metrics = view["visits"][0]["metrics"]
        self.assertEqual(metrics["response_length_mean"], 200.0)
        self.assertEqual(metrics["response_length_total"], 400)
        self.assertEqual(metrics["agent_turns_mean"], 15.0)
        self.assertEqual(metrics["tool_calls_mean"], 10.0)


class TestInstanceTrend(unittest.TestCase):
    def test_trend_points_follow_visits(self):
        rows = [
            {"rollout_id": 0, "group_index": 0, "instance_id": "repo_pr1", "category": "pass",
             "attempt_pk": 1, "attempt_id": 0, "reward": 1.0,
             "response_length": 100, "agent_turns": 10},
            {"rollout_id": 0, "group_index": 0, "instance_id": "repo_pr1", "category": "fail",
             "attempt_pk": 2, "attempt_id": 1, "reward": 0.0,
             "response_length": 300, "agent_turns": 20},
            {"rollout_id": 5, "group_index": 0, "instance_id": "repo_pr1", "category": "pass",
             "attempt_pk": 3, "attempt_id": 0, "reward": 1.0,
             "response_length": 200, "agent_turns": 15},
        ]
        view = group_instance_appearances("repo_pr1", rows)
        trend = _instance_trend(view["visits"])
        self.assertEqual(trend["n_visits"], 2)
        self.assertTrue(trend["has_rate"])
        self.assertTrue(trend["has_response_length"])
        self.assertTrue(trend["has_agent_turns"])
        first, second = trend["points"]
        self.assertEqual(first["epoch"], 1)
        self.assertEqual(first["n_pass"], 1)
        self.assertEqual(first["n_attempts"], 2)
        self.assertAlmostEqual(first["rate"], 0.5)
        self.assertEqual(first["response_length"], 200.0)
        self.assertEqual(second["epoch"], 2)
        self.assertAlmostEqual(second["rate"], 1.0)

    def test_trend_empty(self):
        trend = _instance_trend([])
        self.assertEqual(trend["n_visits"], 0)
        self.assertFalse(trend["has_rate"])
        self.assertFalse(trend["has_response_length"])


class TestResponseStats(unittest.TestCase):
    def test_two_dimensions(self):
        raw = summarize_response_attempts(
            [
                {"assistant_turns": [{"has_tool_call": False, "tool_call_names": []}]},
                {
                    "assistant_turns": [
                        {"has_tool_call": True, "tool_call_names": ["execute_bash"]},
                        {"has_tool_call": True, "tool_call_names": ["finish"]},
                    ]
                },
                {"assistant_turns": [{"has_tool_call": True, "tool_call_names": ["execute_bush"]}]},
                {
                    "assistant_turns": [
                        {"has_tool_call": True, "tool_call_names": ["finish"]},
                        {"has_tool_call": True, "tool_call_names": ["execute_bash"]},
                    ]
                },
                {"assistant_turns": []},
            ]
        )
        self.assertEqual(raw["n_assistant"], 6)
        self.assertEqual(raw["n_with_tool_call"], 5)
        self.assertEqual(raw["n_without_tool_call"], 1)
        self.assertEqual(raw["n_finish_calls"], 2)
        self.assertEqual(raw["n_invalid_responses"], 1)
        self.assertEqual(raw["invalid_names"].get("execute_bush"), 1)
        self.assertEqual(raw["n_attempts"], 5)
        self.assertEqual(raw["n_attempts_with_assistant"], 4)
        self.assertEqual(raw["n_attempts_no_tool_call"], 1)
        self.assertEqual(raw["n_attempts_last_is_finish"], 1)
        self.assertEqual(raw["n_attempts_last_not_finish"], 3)
        self.assertEqual(raw["n_attempts_incomplete"], 3)

        merged = merge_response_stats([raw, raw])
        self.assertEqual(merged["n_without_tool_call"], 2)
        self.assertEqual(merged["n_finish_calls"], 4)
        view = public_response_stats(merged)
        self.assertEqual(view["tool_names"][0]["name"] in {"execute_bash", "finish", "execute_bush"}, True)
        self.assertAlmostEqual(view["ratios"]["without_tool_call"], 2 / 12)

    def test_split_response_last_tool_is_finish(self):
        text = (
            "<|im_start|>assistant\n"
            "<tool_call>execute_bash\n</tool_call><|im_end|>\n"
            "<|im_start|>assistant\n"
            "只写了总结，没有工具。<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<tool_call>finish\n</tool_call><|im_end|>"
        )
        turns = [
            {
                "has_tool_call": item["has_tool_call"],
                "tool_call_names": item["tool_call_names"],
            }
            for item in split_response_to_turns(text)
            if item["role_label"] == "assistant"
        ]
        raw = summarize_response_attempts([{"assistant_turns": turns}])
        self.assertEqual(raw["n_assistant"], 3)
        self.assertEqual(raw["n_without_tool_call"], 1)
        self.assertEqual(raw["n_finish_calls"], 1)
        self.assertEqual(raw["n_attempts_last_is_finish"], 1)
        self.assertEqual(raw["n_attempts_no_tool_call"], 1)
        self.assertEqual(raw["n_attempts_incomplete"], 1)


class TestResponseCache(unittest.TestCase):
    def test_fingerprint_mismatch_stays_pending(self):
        rollouts = [
            {"rollout_id": 0, "pt_path": "/tmp/a.pt", "pt_size_bytes": 10, "pt_mtime": 1.0},
            {"rollout_id": 1, "pt_path": "/tmp/b.pt", "pt_size_bytes": 11, "pt_mtime": 2.0},
        ]
        cache = {
            "version": 1,
            "rollouts": {
                "0": {
                    "pt_path": "/tmp/a.pt",
                    "pt_size_bytes": 10,
                    "pt_mtime": 1.0,
                    "stats": {"n_assistant": 3},
                },
                "1": {
                    "pt_path": "/tmp/b.pt",
                    "pt_size_bytes": 9,
                    "pt_mtime": 2.0,
                    "stats": {"n_assistant": 1},
                },
            },
        }
        matched, pending = _match_cache(rollouts, cache)
        self.assertEqual([item["rollout_id"] for item in matched], [0])
        self.assertEqual([item["rollout_id"] for item in pending], [1])

    def test_mtime_json_roundtrip_still_matches(self):
        rollouts = [
            {"rollout_id": 0, "pt_path": "/tmp/a.pt", "pt_size_bytes": 10, "pt_mtime": "1718000000.123457"},
        ]
        cache = {
            "version": 1,
            "rollouts": {
                "0": {
                    "pt_path": "/tmp/a.pt",
                    "pt_size_bytes": 10,
                    "pt_mtime": 1718000000.1234568,
                    "stats": {"n_assistant": 1},
                }
            },
        }
        matched, pending = _match_cache(rollouts, cache)
        self.assertEqual(len(matched), 1)
        self.assertEqual(pending, [])


if __name__ == "__main__":
    unittest.main()
