"""utils.failure_reason_taxonomy：失败原因归一化与 agent 结束态标签。"""
from __future__ import annotations

import unittest

from utils.failure_reason_taxonomy import (
    AGENT_OUTCOME_LABELS,
    CATEGORY_LABELS,
    agent_outcome_label,
    agent_outcome_reason,
    classify_from_fields,
    is_agent_error_outcome,
    normalize_reason,
    split_reason_prefix,
)


class TestNormalizeReason(unittest.TestCase):
    def test_abort_reason_wins(self):
        category, reason = normalize_reason(
            status="aborted", abort_reason="exception:SandboxApiException"
        )
        self.assertEqual(category, "abort")
        self.assertEqual(reason, "exception:SandboxApiException")

    def test_abort_by_status(self):
        category, reason = normalize_reason(status="aborted")
        self.assertEqual(category, "abort")
        self.assertEqual(reason, "status:aborted")

    def test_timeout_by_status(self):
        category, reason = normalize_reason(status="truncated", reward=0.0)
        self.assertEqual(category, "timeout")
        self.assertEqual(reason, "status:truncated")

    def test_timeout_uses_error_code_first(self):
        category, reason = normalize_reason(
            status="truncated",
            error_code="slime_context_limit_reached",
            agent_outcome="context_exhausted",
        )
        self.assertEqual(category, "timeout")
        self.assertEqual(reason, "slime_context_limit_reached")

    def test_timeout_falls_back_to_agent_outcome_code(self):
        """没有 error_code 时用 agent 结束态映射出的原因码（与 .pt 侧口径一致）。"""
        category, reason = normalize_reason(
            status="truncated", agent_outcome="context_exhausted"
        )
        self.assertEqual(category, "timeout")
        self.assertEqual(reason, "slime_context_limit_reached")

    def test_pass_and_fail_by_reward(self):
        self.assertEqual(normalize_reason(reward=1.0, diagnosis="resolved")[0], "pass")
        category, reason = normalize_reason(reward=0.0, diagnosis="evaluator:tests_failed")
        self.assertEqual(category, "fail")
        self.assertEqual(reason, "evaluator:tests_failed")

    def test_sdk_error_is_abort(self):
        category, reason = normalize_reason(agent_outcome="sdk_error")
        self.assertEqual(category, "abort")
        self.assertEqual(reason, "agent:sdk_error")

    def test_other_without_reward(self):
        category, reason = normalize_reason(status="completed", diagnosis="")
        self.assertEqual(category, "other")


class TestClassifyFromFields(unittest.TestCase):
    def test_from_summary_fields(self):
        category, reason = classify_from_fields(
            {
                "status": "completed",
                "reward": 0.0,
                "diagnosis": "evaluator:invalid_evaluator_payload",
            }
        )
        self.assertEqual((category, reason), ("fail", "evaluator:invalid_evaluator_payload"))

    def test_agent_outcome_used_when_no_diagnosis(self):
        """status=aborted 时优先按 status 归类（abort_reason 缺失）。"""
        category, reason = classify_from_fields(
            {"status": "aborted", "reward": None, "agent_outcome": "time_budget_exceeded"}
        )
        self.assertEqual(category, "abort")
        self.assertEqual(reason, "status:aborted")

    def test_agent_outcome_reason_when_status_is_generic(self):
        """status 无信息时用 agent 结束态推出原因码。"""
        category, reason = classify_from_fields(
            {"status": "completed", "reward": None, "agent_outcome": "time_budget_exceeded"}
        )
        self.assertEqual(category, "other")
        self.assertEqual(reason, "agent_time_budget_exceeded")


class TestAgentOutcome(unittest.TestCase):
    def test_label_known_and_unknown(self):
        self.assertEqual(agent_outcome_label("context_exhausted"), "上下文打满")
        self.assertEqual(agent_outcome_label("weird_state"), "weird_state")
        self.assertEqual(agent_outcome_label(""), "")

    def test_is_error_outcome(self):
        self.assertFalse(is_agent_error_outcome("finished"))
        self.assertFalse(is_agent_error_outcome(""))
        self.assertTrue(is_agent_error_outcome("context_exhausted"))
        self.assertTrue(is_agent_error_outcome("sdk_error"))

    def test_reason_codes_present_for_known_outcomes(self):
        self.assertEqual(
            agent_outcome_reason("context_exhausted"), "slime_context_limit_reached"
        )
        self.assertEqual(agent_outcome_reason("time_budget_exceeded"), "agent_time_budget_exceeded")

    def test_labels_cover_known_outcomes(self):
        for outcome in ("finished", "context_exhausted", "time_budget_exceeded", "sdk_error"):
            self.assertIn(outcome, AGENT_OUTCOME_LABELS)

    def test_category_labels_cover_categories(self):
        for cat in ("pass", "fail", "timeout", "abort", "other"):
            self.assertIn(cat, CATEGORY_LABELS)


class TestSplitReasonPrefix(unittest.TestCase):
    def test_split(self):
        self.assertEqual(split_reason_prefix("evaluator:tests_failed"), ("evaluator", "tests_failed"))

    def test_no_prefix(self):
        self.assertEqual(split_reason_prefix("resolved"), ("", "resolved"))


if __name__ == "__main__":
    unittest.main()
