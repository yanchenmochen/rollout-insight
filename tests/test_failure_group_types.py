"""failure_analysis_service：组内失败类型汇总 + agent 结束态聚类。

覆盖「组结果」列渲染（超时 / abort 变超链接）与失败分析页的 agent 异常维度。
"""
from __future__ import annotations

import unittest

from service.failure_analysis_service import (
    build_agent_outcome_clusters,
    build_group_failure_types,
)


def _item(pk, gi, category, reason, rollout_id=0, instance_id="repo_pr1", agent_outcome=""):
    return {
        "attempt_pk": pk,
        "group_index": gi,
        "category": category,
        "reason": reason,
        "rollout_id": rollout_id,
        "instance_id": instance_id,
        "agent_outcome": agent_outcome,
    }


class TestBuildGroupFailureTypes(unittest.TestCase):
    def test_groups_by_group_index_with_attempt_links(self):
        items = [
            _item(11, 0, "fail", "evaluator:tests_failed"),
            _item(12, 0, "fail", "evaluator:tests_failed"),
            _item(13, 0, "timeout", "slime_context_limit_reached"),
            _item(21, 1, "abort", "openhands_infra_exit_2"),
        ]
        rows = build_group_failure_types(items)
        self.assertEqual([r["group_index"] for r in rows], [0, 1])

        g0 = rows[0]
        self.assertEqual(g0["count"], 3)
        types = {t["reason"]: t for t in g0["types"]}
        self.assertEqual(types["evaluator:tests_failed"]["count"], 2)
        self.assertEqual(types["evaluator:tests_failed"]["attempts"], [11, 12])
        self.assertEqual(types["slime_context_limit_reached"]["attempts"], [13])

        g1 = rows[1]
        self.assertEqual(g1["count"], 1)
        self.assertEqual(g1["types"][0]["attempts"], [21])

    def test_types_sorted_by_count_desc(self):
        items = [_item(i, 0, "fail", f"reason-{i % 3}") for i in range(9)]
        rows = build_group_failure_types(items)
        counts = [t["count"] for t in rows[0]["types"]]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_skips_missing_group_index(self):
        item = _item(1, 0, "fail", "r")
        item["group_index"] = None
        self.assertEqual(build_group_failure_types([item]), [])

    def test_empty(self):
        self.assertEqual(build_group_failure_types([]), [])


class TestBuildAgentOutcomeClusters(unittest.TestCase):
    def test_clusters_by_agent_outcome(self):
        items = [
            _item(1, 0, "fail", "evaluator:tests_failed", agent_outcome="finished"),
            _item(2, 0, "timeout", "slime_context_limit_reached", agent_outcome="context_exhausted"),
            _item(3, 1, "timeout", "slime_context_limit_reached", agent_outcome="context_exhausted"),
            _item(4, 1, "abort", "openhands_infra_exit_2", agent_outcome="sdk_error"),
        ]
        rows = build_agent_outcome_clusters(items)
        by_outcome = {r["agent_outcome"]: r for r in rows}
        self.assertEqual(by_outcome["context_exhausted"]["count"], 2)
        self.assertTrue(by_outcome["context_exhausted"]["is_error"])
        self.assertEqual(by_outcome["context_exhausted"]["label"], "上下文打满")
        self.assertFalse(by_outcome["finished"]["is_error"])
        self.assertTrue(by_outcome["sdk_error"]["is_error"])

    def test_items_without_agent_outcome_are_skipped(self):
        items = [_item(1, 0, "fail", "evaluator:tests_failed")]
        self.assertEqual(build_agent_outcome_clusters(items), [])

    def test_empty(self):
        self.assertEqual(build_agent_outcome_clusters([]), [])


if __name__ == "__main__":
    unittest.main()
