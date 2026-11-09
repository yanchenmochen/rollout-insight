"""classify_sample_outcome / cluster_attempt_outcomes。"""
from __future__ import annotations

import unittest

from utils.rollout_pt_parser import (
    classify_sample_outcome,
    cluster_abort_reasons,
    cluster_attempt_outcomes,
)


class TestClusterAttemptOutcomes(unittest.TestCase):
    def test_classify_priority(self):
        self.assertEqual(
            classify_sample_outcome(
                {
                    "status": "aborted",
                    "metadata": {"abort_reason": "exception:SandboxInternalException"},
                    "reward": 0.0,
                }
            )["category"],
            "abort",
        )
        self.assertEqual(
            classify_sample_outcome(
                {
                    "status": "truncated",
                    "metadata": {
                        "conversation_error_code": "slime_context_limit_reached",
                        "raw_reward": 0.0,
                        "diagnosis": "evaluator:tests_failed",
                    },
                    "reward": 0.0,
                }
            )["category"],
            "timeout",
        )
        self.assertEqual(
            classify_sample_outcome(
                {
                    "status": "completed",
                    "metadata": {
                        "raw_reward": 0.0,
                        "diagnosis": "evaluator:tests_failed",
                    },
                    "reward": 0.0,
                }
            ),
            {
                "category": "fail",
                "reason": "evaluator:tests_failed",
                "label": "fail:evaluator:tests_failed",
            },
        )
        self.assertEqual(
            classify_sample_outcome(
                {
                    "status": "completed",
                    "metadata": {"raw_reward": 1.0},
                    "reward": 1.0,
                }
            )["category"],
            "pass",
        )

    def test_cluster_cross_and_group(self):
        dump = {
            "samples": [
                {
                    "group_index": 1,
                    "status": "aborted",
                    "metadata": {"abort_reason": "exception:A"},
                },
                {
                    "group_index": 1,
                    "status": "completed",
                    "metadata": {"raw_reward": 1.0},
                    "reward": 1.0,
                },
                {
                    "group_index": 2,
                    "status": "truncated",
                    "metadata": {
                        "conversation_error_code": "slime_context_limit_reached",
                        "raw_reward": 0.0,
                    },
                    "reward": 0.0,
                },
                {
                    "group_index": 2,
                    "status": "completed",
                    "metadata": {
                        "raw_reward": 0.0,
                        "diagnosis": "evaluator:tests_failed",
                    },
                    "reward": 0.0,
                },
                {
                    "group_index": 2,
                    "status": "aborted",
                    "metadata": {"abort_reason": "exception:A"},
                },
            ]
        }
        all_c = cluster_attempt_outcomes(dump)
        self.assertEqual(all_c["n_attempts"], 5)
        self.assertEqual(all_c["n_abort"], 2)
        self.assertEqual(all_c["n_timeout"], 1)
        self.assertEqual(all_c["n_fail"], 1)
        self.assertEqual(all_c["n_pass"], 1)
        self.assertEqual(all_c["n_failure"], 4)
        self.assertEqual(all_c["n_groups_with_failure"], 2)
        cats = {r["category"]: r["count"] for r in all_c["by_category"]}
        self.assertEqual(cats["abort"], 2)
        self.assertEqual(cats["timeout"], 1)
        labels = {r["label"]: r["count"] for r in all_c["by_reason"]}
        self.assertEqual(labels["abort:exception:A"], 2)
        self.assertEqual(labels["timeout:slime_context_limit_reached"], 1)
        self.assertEqual(labels["fail:evaluator:tests_failed"], 1)
        self.assertEqual(len(all_c["group_trend"]), 2)
        g2 = cluster_attempt_outcomes(dump, group_index=2)
        self.assertEqual(g2["n_attempts"], 3)
        self.assertEqual(g2["n_abort"], 1)
        self.assertEqual(g2["n_timeout"], 1)
        self.assertEqual(g2["n_fail"], 1)
        self.assertEqual(g2["n_pass"], 0)

        # 兼容旧 abort 聚类
        abort = cluster_abort_reasons(dump)
        self.assertEqual(abort["total"], 2)
        self.assertEqual(abort["by_reason"][0]["reason"], "exception:A")
        self.assertEqual(abort["by_group"][2]["total"], 1)


if __name__ == "__main__":
    unittest.main()
