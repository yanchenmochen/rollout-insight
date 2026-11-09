"""resolve_sample_reward：abort 不得把 sample.reward=0.0 当成评测失败。"""
from __future__ import annotations

import unittest

from utils.rollout_pt_parser import resolve_sample_reward


class TestResolveSampleReward(unittest.TestCase):
    def test_abort_status_returns_none_even_if_reward_zero(self):
        sample = {
            "status": "aborted",
            "reward": 0.0,
            "metadata": {"abort_reason": "exception:SandboxApiException"},
        }
        self.assertIsNone(resolve_sample_reward(sample))

    def test_raw_reward_key_missing_with_abort_reason(self):
        sample = {
            "status": "completed",
            "reward": 0.0,
            "metadata": {"abort_reason": "exception:SandboxApiException"},
        }
        self.assertIsNone(resolve_sample_reward(sample))

    def test_prefers_raw_reward_when_present(self):
        sample = {
            "status": "completed",
            "reward": 0.0,
            "metadata": {"raw_reward": 1.0},
        }
        self.assertEqual(resolve_sample_reward(sample), 1.0)

    def test_raw_reward_none_is_unmeasurable(self):
        sample = {
            "status": "completed",
            "reward": 0.0,
            "metadata": {"raw_reward": None},
        }
        self.assertIsNone(resolve_sample_reward(sample))

    def test_fallback_sample_reward(self):
        sample = {"status": "completed", "reward": 0.0, "metadata": {}}
        self.assertEqual(resolve_sample_reward(sample), 0.0)


if __name__ == "__main__":
    unittest.main()
