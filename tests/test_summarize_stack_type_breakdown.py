"""summarize_stack_type_breakdown：类型柱拆分（失败 / 超时有奖无奖 / abort）。"""
from __future__ import annotations

import unittest

from utils.rollout_pt_parser import summarize_stack_type_breakdown


class TestSummarizeStackTypeBreakdown(unittest.TestCase):
    def test_fail_timeout_reward_abort_split(self):
        dump = {
            "samples": [
                {
                    "group_index": 0,
                    "status": "completed",
                    "metadata": {"raw_reward": 1.0},
                    "reward": 1.0,
                },
                {
                    "group_index": 0,
                    "status": "completed",
                    "metadata": {
                        "raw_reward": 0.0,
                        "diagnosis": "evaluator:tests_failed",
                    },
                    "reward": 0.0,
                },
                {
                    "group_index": 1,
                    "status": "completed",
                    "metadata": {
                        "raw_reward": 0.0,
                        "diagnosis": "agent:no_diff",
                    },
                    "reward": 0.0,
                },
                {
                    "group_index": 1,
                    "status": "truncated",
                    "metadata": {
                        "conversation_error_code": "slime_context_limit_reached",
                        "raw_reward": 1.0,
                    },
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
                    "status": "aborted",
                    "metadata": {"abort_reason": "exception:SandboxInternalException"},
                    "reward": 0.0,
                },
            ]
        }
        out = summarize_stack_type_breakdown(dump)
        self.assertEqual(out["n_attempts"], 6)
        self.assertEqual(out["n_pass"], 1)
        self.assertEqual(out["n_fail"], 2)
        self.assertEqual(out["n_timeout"], 2)
        self.assertEqual(out["n_abort"], 1)
        self.assertEqual(out["n_timeout_rw1"], 1)
        self.assertEqual(out["n_timeout_rw0"], 1)

        by_key = {s["key"]: s for s in out["segments"]}
        self.assertEqual(by_key["pass"]["count"], 1)
        self.assertEqual(by_key["fail:evaluator:tests_failed"]["count"], 1)
        self.assertEqual(by_key["fail:agent:no_diff"]["count"], 1)
        self.assertEqual(
            by_key["timeout:slime_context_limit_reached:rw1"]["count"], 1
        )
        self.assertEqual(
            by_key["timeout:slime_context_limit_reached:rw0"]["count"], 1
        )
        self.assertEqual(
            by_key["abort:exception:SandboxInternalException"]["count"], 1
        )
        self.assertIn("有奖", by_key["timeout:slime_context_limit_reached:rw1"]["label"])
        self.assertIn("无奖", by_key["timeout:slime_context_limit_reached:rw0"]["label"])

    def test_empty_dump(self):
        out = summarize_stack_type_breakdown({})
        self.assertEqual(out["n_attempts"], 0)
        self.assertEqual(out["segments"], [])


if __name__ == "__main__":
    unittest.main()
