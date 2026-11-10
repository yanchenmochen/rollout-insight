"""summarize_train_sample_stats：进入训练样本统计。"""
from __future__ import annotations

import unittest

from utils.rollout_pt_parser import summarize_train_sample_stats


class TestSummarizeTrainSampleStats(unittest.TestCase):
    def test_kept_vs_dropped(self):
        dump = {
            "samples": [
                {"metadata": {"raw_reward": 1.0}},
                {"metadata": {"raw_reward": 0.0, "train_drop_reason": "zero_std_1.0"}},
                {"metadata": {"raw_reward": 0.0, "train_drop_reason": "zero_std_1.0"}},
                {"metadata": {"train_drop_reason": "ungradable"}},
                {"metadata": {}},
            ]
        }
        out = summarize_train_sample_stats(dump)
        self.assertEqual(out["n_samples"], 5)
        self.assertEqual(out["n_train_kept"], 2)
        self.assertEqual(out["n_train_dropped"], 3)
        self.assertEqual(out["keep_pct"], 40.0)
        by_reason = {r["reason"]: r["count"] for r in out["by_drop_reason"]}
        self.assertEqual(by_reason["zero_std_1.0"], 2)
        self.assertEqual(by_reason["ungradable"], 1)

    def test_empty(self):
        out = summarize_train_sample_stats({})
        self.assertEqual(out["n_samples"], 0)
        self.assertEqual(out["n_train_kept"], 0)


if __name__ == "__main__":
    unittest.main()
