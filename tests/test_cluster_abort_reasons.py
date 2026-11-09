"""cluster_abort_reasons / sample_abort_reason。"""
from __future__ import annotations

import unittest

from utils.rollout_pt_parser import cluster_abort_reasons, sample_abort_reason


class TestClusterAbortReasons(unittest.TestCase):
    def test_sample_abort_reason(self):
        self.assertEqual(
            sample_abort_reason(
                {
                    "status": "aborted",
                    "metadata": {"abort_reason": "exception:SandboxInternalException"},
                }
            ),
            "exception:SandboxInternalException",
        )
        self.assertIsNone(
            sample_abort_reason({"status": "completed", "metadata": {}, "reward": 0.0})
        )
        # train_drop_reason 不是 abort
        self.assertIsNone(
            sample_abort_reason(
                {
                    "status": "completed",
                    "reward": 1.0,
                    "metadata": {
                        "raw_reward": 1.0,
                        "train_drop_reason": "zero_std_1.0",
                    },
                }
            )
        )

    def test_cluster_rollout_and_group(self):
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
                    "status": "aborted",
                    "metadata": {"abort_reason": "exception:A"},
                },
                {
                    "group_index": 2,
                    "status": "aborted",
                    "metadata": {"abort_reason": "exception:B"},
                },
            ]
        }
        all_c = cluster_abort_reasons(dump)
        self.assertEqual(all_c["total"], 3)
        self.assertEqual(all_c["n_groups_with_abort"], 2)
        self.assertEqual(all_c["by_reason"][0]["reason"], "exception:A")
        self.assertEqual(all_c["by_reason"][0]["count"], 2)

        g2 = cluster_abort_reasons(dump, group_index=2)
        self.assertEqual(g2["total"], 2)
        reasons = {r["reason"]: r["count"] for r in g2["by_reason"]}
        self.assertEqual(reasons, {"exception:A": 1, "exception:B": 1})
        self.assertEqual(all_c["by_group"][2]["total"], 2)
        self.assertEqual(len(all_c["by_group"][2]["by_reason"]), 2)


if __name__ == "__main__":
    unittest.main()
