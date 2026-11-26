"""Agentic 问题卡片拼装：给定计数输出 count / ratio / status。"""
from __future__ import annotations

import unittest

from data.agentic_problem_catalog import AGENTIC_PROBLEMS
from service.agentic_problem_service import assemble_problem_cards, problem_status


class TestProblemStatus(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(problem_status(None, warn_ratio=0.1, bad_ratio=0.2), "unknown")
        self.assertEqual(problem_status(0.05, warn_ratio=0.1, bad_ratio=0.2), "ok")
        self.assertEqual(problem_status(0.1, warn_ratio=0.1, bad_ratio=0.2), "warn")
        self.assertEqual(problem_status(0.25, warn_ratio=0.1, bad_ratio=0.2), "bad")


class TestAssembleProblemCards(unittest.TestCase):
    def test_agent_outcome_and_behavior(self):
        cards = assemble_problem_cards(
            n_attempts=100,
            totals={
                "n_attempts": 100,
                "n_abort": 10,
                "n_fail": 40,
                "n_timeout": 5,
                "n_other": 0,
                "n_failure": 55,
            },
            agent_outcomes=[
                {"agent_outcome": "max_fake_responses", "count": 8},
                {"agent_outcome": "context_exhausted", "count": 20},
                {"agent_outcome": "sdk_error", "count": 3},
            ],
            clusters=[
                {"category": "fail", "reason": "evaluator:tests_failed", "count": 30},
                {"category": "fail", "reason": "evaluator:invalid_evaluator_payload", "count": 4},
                {"category": "fail", "reason": "agent:no_diff", "count": 2},
            ],
            behavior={
                "n_failed": 55,
                "cards": [
                    {"key": "no_tool_call", "n_attempts": 12, "n_failed": 55},
                    {"key": "last_not_finish", "n_attempts": 5, "n_failed": 55},
                    {"key": "early_finish", "n_attempts": 1, "n_failed": 55},
                ],
            },
            gradient_latest={
                "n_groups": 16,
                "n_all_pass": 4,
                "n_all_fail": 8,
                "n_mixed": 4,
            },
            pool_health={
                "n_problems": 64,
                "n_saturated": 20,
                "n_hard": 10,
                "saturated_pool_ratio": 20 / 64,
                "hard_pool_ratio": 10 / 64,
            },
            task_id=7,
        )
        by_id = {c["id"]: c for c in cards}
        self.assertEqual(len(cards), len(AGENTIC_PROBLEMS))

        fake = by_id["max_fake_responses"]
        self.assertEqual(fake["count"], 8)
        self.assertAlmostEqual(fake["ratio"], 0.08)
        self.assertEqual(fake["status"], "bad")
        self.assertIn("agent=max_fake_responses", fake["failure_url"] or "")

        ctx = by_id["context_exhausted"]
        self.assertEqual(ctx["count"], 20)
        self.assertEqual(ctx["status"], "bad")

        no_tool = by_id["no_tool_call"]
        self.assertEqual(no_tool["count"], 12)
        self.assertAlmostEqual(no_tool["ratio"], 12 / 55)
        self.assertEqual(no_tool["denominator_label"], "failed")

        zero = by_id["zero_variance_groups"]
        self.assertEqual(zero["count"], 12)
        self.assertEqual(zero["denominator"], 16)
        self.assertAlmostEqual(zero["ratio"], 12 / 16)
        self.assertEqual(zero["status"], "bad")

        sat = by_id["pool_saturated"]
        self.assertEqual(sat["count"], 20)
        self.assertAlmostEqual(sat["ratio"], 20 / 64)

        eval_fault = by_id["evaluator_fault"]
        self.assertEqual(eval_fault["count"], 4)

        tests = by_id["tests_failed"]
        self.assertEqual(tests["count"], 30)

        # 严重度：bad 应排在 ok 前面
        statuses = [c["status"] for c in cards]
        first_ok = next((i for i, s in enumerate(statuses) if s == "ok"), len(statuses))
        last_bad = max((i for i, s in enumerate(statuses) if s == "bad"), default=-1)
        self.assertLess(last_bad, first_ok)

    def test_missing_fields_are_zero(self):
        cards = assemble_problem_cards(n_attempts=0)
        self.assertEqual(len(cards), len(AGENTIC_PROBLEMS))
        for card in cards:
            self.assertEqual(card["count"], 0)
            self.assertIn(card["status"], {"ok", "unknown"})


if __name__ == "__main__":
    unittest.main()
