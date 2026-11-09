"""summarize_groups：rollout 详情页 Group 概览聚合。"""
from __future__ import annotations

import unittest

from controller.rollout_controller import summarize_groups


class TestSummarizeGroups(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(summarize_groups([]), [])
        self.assertEqual(summarize_groups(None), [])  # type: ignore[arg-type]

    def test_buckets_and_segments(self):
        attempts = [
            {
                "group_index": 10,
                "instance_id": "repo_pr1",
                "reward_bucket": "all_one",
                "segment_count": 2,
            },
            {
                "group_index": 10,
                "reward_bucket": "all_zero",
                "segment_count": 1,
            },
            {
                "group_index": 10,
                "reward_bucket": "mixed",
                "segments": [{"primary_id": 1}, {"primary_id": 2}, {"primary_id": 3}],
            },
            {
                "group_index": 7,
                "reward_bucket": "none",
                "segment_count": 4,
            },
            {
                "group_index": 7,
                "reward_bucket": "ALL_ONE",  # case-insensitive
                "segment_count": 1,
            },
        ]
        rows = summarize_groups(attempts, {7: {"title": "from-brief"}})
        self.assertEqual([r["group_index"] for r in rows], [7, 10])

        g7 = rows[0]
        self.assertEqual(g7["title"], "from-brief")
        self.assertEqual(g7["n_attempts"], 2)
        self.assertEqual(g7["n_segments"], 5)
        self.assertEqual(g7["all_zero"], 0)
        self.assertEqual(g7["all_one"], 1)
        self.assertEqual(g7["mixed"], 0)
        self.assertEqual(g7["none"], 1)
        self.assertEqual(
            g7["all_zero"] + g7["all_one"] + g7["mixed"] + g7["none"],
            g7["n_attempts"],
        )
        self.assertEqual(g7["outcome"], "other")  # 全1 + 无，无跨 attempt 分化

        g10 = rows[1]
        self.assertEqual(g10["title"], "repo_pr1")
        self.assertEqual(g10["n_attempts"], 3)
        # 2 + 1 + len(segments)=3
        self.assertEqual(g10["n_segments"], 6)
        self.assertEqual(g10["all_zero"], 1)
        self.assertEqual(g10["all_one"], 1)
        self.assertEqual(g10["mixed"], 1)
        self.assertEqual(g10["none"], 0)
        self.assertEqual(
            g10["all_zero"] + g10["all_one"] + g10["mixed"] + g10["none"],
            g10["n_attempts"],
        )
        self.assertEqual(g10["outcome"], "split")  # 既有全0又有全1

    def test_group_187_style_split_not_seg_mixed(self):
        """全0×6 + 全1×2 → 组结果=分化；段内混合仍为 0。"""
        attempts = (
            [{"group_index": 187, "reward_bucket": "all_zero", "segment_count": 1}] * 6
            + [{"group_index": 187, "reward_bucket": "all_one", "segment_count": 1}] * 2
        )
        # 给其中一个多一个 segment，模拟 segments=9
        attempts[0] = {
            "group_index": 187,
            "reward_bucket": "all_zero",
            "segment_count": 2,
        }
        rows = summarize_groups(attempts)
        self.assertEqual(len(rows), 1)
        g = rows[0]
        self.assertEqual(g["n_attempts"], 8)
        self.assertEqual(g["n_segments"], 9)
        self.assertEqual(g["all_zero"], 6)
        self.assertEqual(g["all_one"], 2)
        self.assertEqual(g["mixed"], 0)
        self.assertEqual(g["none"], 0)
        self.assertEqual(g["outcome"], "split")
        self.assertEqual(
            g["all_zero"] + g["all_one"] + g["mixed"] + g["none"], g["n_attempts"]
        )

    def test_skips_missing_group_index(self):
        rows = summarize_groups(
            [
                {"group_index": None, "reward_bucket": "all_one", "segment_count": 9},
                {"reward_bucket": "all_one", "segment_count": 9},
                {"group_index": 1, "reward_bucket": "all_one", "segment_count": 1},
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_index"], 1)
        self.assertEqual(rows[0]["n_segments"], 1)


if __name__ == "__main__":
    unittest.main()
