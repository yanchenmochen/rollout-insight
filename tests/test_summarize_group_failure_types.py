"""rollout_controller：组内失败类型汇总（组结果列的超链接数据源）。"""
from __future__ import annotations

import unittest

from controller.rollout_controller import summarize_group_failure_types


def _attempt(pk, gi, aid, status="completed", reward=0.0):
    return {
        "primary_id": pk,
        "group_index": gi,
        "attempt_id": aid,
        "status": status,
        "reward": reward,
    }


class TestSummarizeGroupFailureTypes(unittest.TestCase):
    def test_uses_annotation_outcomes_when_available(self):
        attempts = [
            _attempt(1, 0, 2),
            _attempt(2, 0, 7, status="truncated"),
            _attempt(3, 1, 9, status="aborted", reward=None),
        ]
        index = {
            (0, 2): {"outcome_category": "fail", "outcome_reason": "evaluator:tests_failed"},
            (0, 7): {
                "outcome_category": "timeout",
                "outcome_reason": "slime_context_limit_reached",
            },
            (1, 9): {
                "outcome_category": "abort",
                "abort_reason": "openhands_infra_exit_2",
            },
        }
        out = summarize_group_failure_types(attempts, index)
        self.assertEqual(sorted(out.keys()), [0, 1])

        g0 = out[0]
        self.assertEqual(g0["count"], 2)
        reasons = {t["reason"]: t for t in g0["types"]}
        self.assertEqual(reasons["evaluator:tests_failed"]["attempts"], [1])
        self.assertEqual(reasons["slime_context_limit_reached"]["attempts"], [2])
        self.assertEqual(reasons["slime_context_limit_reached"]["category"], "timeout")

        g1 = out[1]
        self.assertEqual(g1["types"][0]["category"], "abort")
        self.assertEqual(g1["types"][0]["attempts"], [3])

    def test_falls_back_to_status_and_reward(self):
        attempts = [
            _attempt(1, 0, 2, status="completed", reward=0.0),
            _attempt(2, 0, 3, status="truncated", reward=0.0),
            _attempt(3, 0, 4, status="aborted", reward=None),
            _attempt(4, 0, 5, status="completed", reward=1.0),
        ]
        out = summarize_group_failure_types(attempts, {})
        cats = {t["category"] for t in out[0]["types"]}
        self.assertEqual(cats, {"fail", "timeout", "abort"})
        # 通过的 attempt 不应出现在失败类型里
        pks = [pk for t in out[0]["types"] for pk in t["attempts"]]
        self.assertNotIn(4, pks)

    def test_empty(self):
        self.assertEqual(summarize_group_failure_types([], {}), {})


if __name__ == "__main__":
    unittest.main()
