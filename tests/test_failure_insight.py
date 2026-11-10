"""failure_insight：正常失败 / 超时 / abort 深挖提取与聚类。"""
from __future__ import annotations

import unittest

from utils.failure_insight import (
    cluster_failure_insights,
    empty_failure_insights,
    extract_sample_failure_detail,
)


class TestFailureInsight(unittest.TestCase):
    def test_extract_fail_from_stdout_tail(self):
        sample = {
            "group_index": 2,
            "rollout_id": 7,
            "status": "completed",
            "reward": 0.0,
            "metadata": {
                "raw_reward": 0.0,
                "diagnosis": "evaluator:tests_failed",
                "instance_id": "repo_pr1",
                "sandbox_id": "sbx-1",
                "evaluation": {
                    "grader": "scale_swe",
                    "exit_code": 1,
                    "stdout_tail": (
                        "=================================== FAILURES ===================================\n"
                        "___________________ test_call_is_namedtuple_with_annotations ___________________\n"
                        "test_fail_to_pass.py:42: in test_call_is_namedtuple_with_annotations\n"
                        '    assert "request" in annot\n'
                        "E   AssertionError: assert 'request' in {}\n"
                    ),
                    "stderr_tail": "CommandExecError: 1",
                },
            },
        }
        detail = extract_sample_failure_detail(sample, sample_index=0)
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(detail["category"], "fail")
        self.assertEqual(
            detail["failed_test"], "test_call_is_namedtuple_with_annotations"
        )
        self.assertIn("AssertionError", detail["error_summary"] or "")
        paths = {s["path"] for s in detail["sources"]}
        self.assertIn("sample.metadata.evaluation.stdout_tail", paths)
        self.assertIn("sample.metadata.diagnosis", paths)

    def test_extract_timeout_driver_error(self):
        sample = {
            "group_index": 1,
            "rollout_id": 3,
            "status": "truncated",
            "metadata": {
                "conversation_error_code": "slime_context_limit_reached",
                "driver_error": (
                    "BadRequestError: litellm.BadRequestError: OpenAIException - "
                    "adapter context limit reached: prompt_tokens=131499 "
                    "context_limit_tokens=131071"
                ),
                "context_prompt_tokens": 131499,
                "context_limit_tokens": 131071,
                "attempt_completed_turns": 99,
                "raw_reward": 0.0,
            },
        }
        detail = extract_sample_failure_detail(sample)
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(detail["category"], "timeout")
        self.assertIn("prompt_tokens=N", detail["driver_error_norm"] or "")
        self.assertEqual(detail["context_prompt_tokens"], 131499)
        paths = {s["path"] for s in detail["sources"]}
        self.assertIn("sample.metadata.driver_error", paths)

    def test_extract_abort_notes_no_traceback(self):
        sample = {
            "group_index": 0,
            "status": "aborted",
            "metadata": {"abort_reason": "exception:SandboxApiException"},
        }
        detail = extract_sample_failure_detail(sample)
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(detail["category"], "abort")
        self.assertIn("traceback", (detail["note"] or "").lower())

    def test_cluster_groups_fail_and_timeout(self):
        dump = {
            "samples": [
                {
                    "group_index": 0,
                    "rollout_id": 1,
                    "status": "completed",
                    "reward": 0.0,
                    "metadata": {
                        "raw_reward": 0.0,
                        "diagnosis": "evaluator:tests_failed",
                        "instance_id": "a",
                        "evaluation": {
                            "stdout_tail": (
                                "=================================== FAILURES ===================================\n"
                                "___________________ SeekableTest.test_past_end ___________________\n"
                                "E   StopIteration\n"
                            ),
                        },
                    },
                },
                {
                    "group_index": 0,
                    "rollout_id": 2,
                    "status": "completed",
                    "reward": 0.0,
                    "metadata": {
                        "raw_reward": 0.0,
                        "diagnosis": "evaluator:tests_failed",
                        "instance_id": "b",
                        "evaluation": {
                            "stdout_tail": (
                                "=================================== FAILURES ===================================\n"
                                "___________________ SeekableTest.test_past_end ___________________\n"
                                "E   StopIteration\n"
                            ),
                        },
                    },
                },
                {
                    "group_index": 1,
                    "rollout_id": 3,
                    "status": "truncated",
                    "metadata": {
                        "conversation_error_code": "slime_context_limit_reached",
                        "driver_error": "adapter context limit reached: prompt_tokens=100 context_limit_tokens=90",
                        "context_prompt_tokens": 100,
                        "context_limit_tokens": 90,
                    },
                },
                {
                    "group_index": 1,
                    "rollout_id": 4,
                    "status": "completed",
                    "reward": 1.0,
                    "metadata": {"raw_reward": 1.0},
                },
            ]
        }
        out = cluster_failure_insights(dump)
        self.assertEqual(out["n_fail"], 2)
        self.assertEqual(out["n_timeout"], 1)
        self.assertEqual(len(out["fail"]["clusters"]), 1)
        self.assertEqual(out["fail"]["clusters"][0]["count"], 2)
        self.assertEqual(
            out["fail"]["clusters"][0]["failed_test"], "SeekableTest.test_past_end"
        )
        self.assertEqual(out["timeout"]["clusters"][0]["count"], 1)
        self.assertIn(0, out["by_group"])
        self.assertEqual(out["by_group"][0]["n_fail"], 2)

        g1 = cluster_failure_insights(dump, group_index=1)
        self.assertEqual(g1["n_fail"], 0)
        self.assertEqual(g1["n_timeout"], 1)

    def test_empty(self):
        empty = empty_failure_insights()
        self.assertEqual(empty["n_fail"], 0)
        self.assertEqual(cluster_failure_insights({})["n_fail"], 0)


if __name__ == "__main__":
    unittest.main()
