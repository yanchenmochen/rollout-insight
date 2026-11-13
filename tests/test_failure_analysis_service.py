"""failure_analysis_service：DB attempt ↔ artifacts 关联与失败聚类。"""
from __future__ import annotations

import unittest

from service.failure_analysis_service import build_failure_clusters, link_attempts_to_entries


def _row(pk, gi, aid, inst="repo_pr1", category="fail"):
    return {"attempt_pk": pk, "group_index": gi, "attempt_id": aid, "instance_id": inst, "category": category}


def _entry(stem, gi, rid, inst="repo_pr1", sample_index=0):
    return {"stem": stem, "group_index": gi, "rollout_id": rid, "instance_id": inst, "sample_index": sample_index}


class TestLinkAttempts(unittest.TestCase):
    def test_exact_match_on_group_and_rollout_id(self):
        links = link_attempts_to_entries([_row(1, 3, 24)], [_entry("a", 3, 24), _entry("b", 3, 25)])
        self.assertEqual(links[1][0]["stem"], "a")
        self.assertEqual(links[1][1], "exact")

    def test_abort_matched_by_group_instance(self):
        rows = [_row(1, 3, 24), _row(2, 3, 5, category="abort")]
        entries = [_entry("a", 3, 24, sample_index=24), _entry("ab", 3, None, sample_index=29)]
        links = link_attempts_to_entries(rows, entries)
        self.assertEqual(links[2], (entries[1], "group_unique"))

    def test_multiple_aborts_paired_in_order(self):
        rows = [_row(10, 7, 40, category="abort"), _row(11, 7, 3, category="abort")]
        entries = [_entry("late", 7, None, sample_index=90), _entry("early", 7, None, sample_index=60)]
        links = link_attempts_to_entries(rows, entries)
        self.assertEqual(links[11][0]["stem"], "early")
        self.assertEqual(links[10][0]["stem"], "late")
        self.assertEqual(links[10][1], "group_order")

    def test_no_candidate_left_unlinked(self):
        links = link_attempts_to_entries([_row(1, 3, 24, inst="other")], [_entry("a", 3, None)])
        self.assertEqual(links, {})


class TestBuildClusters(unittest.TestCase):
    def test_group_by_category_reason_with_sub_keys(self):
        items = [
            {"category": "fail", "reason": "evaluator:tests_failed", "rollout_id": 0, "instance_id": "p1", "failed_test": "t1"},
            {"category": "fail", "reason": "evaluator:tests_failed", "rollout_id": 1, "instance_id": "p2", "failed_test": "t1"},
            {"category": "abort", "reason": "exception:SandboxApiException", "rollout_id": 1, "instance_id": "p1"},
        ]
        clusters = build_failure_clusters(items)
        top = clusters[0]
        self.assertEqual((top["category"], top["reason"], top["count"]), ("fail", "evaluator:tests_failed", 2))
        self.assertEqual(top["n_instances"], 2)
        self.assertEqual(top["sub"], [{"key": "t1", "count": 2}])
        self.assertEqual(top["by_rollout"], [{"rollout_id": 0, "count": 1}, {"rollout_id": 1, "count": 1}])
        self.assertAlmostEqual(clusters[1]["pct"], 33.3)


if __name__ == "__main__":
    unittest.main()
