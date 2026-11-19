"""训练曲线 epoch 聚合：epoch 边界计算 + epoch 内取平均。"""
from __future__ import annotations

import unittest

from utils.train_curve_epoch import (
    aggregate_by_epoch,
    epoch_view,
    resolve_steps_per_epoch,
)


def _points(values, start_step=0):
    """构造逐 step 的点序列。"""
    return [
        {"step": start_step + i, "value": float(v)}
        for i, v in enumerate(values)
    ]


class TestResolveStepsPerEpoch(unittest.TestCase):
    def test_exact_division(self):
        # 64 道题 / 每轮抽 16 道 = 4 个 rollout 一个 epoch
        self.assertEqual(resolve_steps_per_epoch(64, 16), 4)

    def test_non_divisible_returns_none(self):
        self.assertIsNone(resolve_steps_per_epoch(64, 15))

    def test_missing_inputs_return_none(self):
        self.assertIsNone(resolve_steps_per_epoch(None, 16))
        self.assertIsNone(resolve_steps_per_epoch(64, None))
        self.assertIsNone(resolve_steps_per_epoch(0, 16))
        self.assertIsNone(resolve_steps_per_epoch(64, 0))


class TestAggregateByEpoch(unittest.TestCase):
    def test_averages_within_epoch(self):
        points = _points([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = aggregate_by_epoch(points, 3)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["epoch"], 1)
        self.assertAlmostEqual(out[0]["value"], 2.0)  # (1+2+3)/3
        self.assertEqual(out[0]["n_steps"], 3)
        self.assertEqual(out[0]["step"], 2)
        self.assertEqual(out[1]["epoch"], 2)
        self.assertAlmostEqual(out[1]["value"], 5.0)  # (4+5+6)/3
        self.assertEqual(out[1]["step"], 5)

    def test_last_partial_epoch_is_kept(self):
        points = _points([1.0, 2.0, 3.0, 4.0])
        out = aggregate_by_epoch(points, 3)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1]["n_steps"], 1)
        self.assertAlmostEqual(out[1]["value"], 4.0)

    def test_unsorted_input_is_sorted(self):
        points = [
            {"step": 2, "value": 3.0},
            {"step": 0, "value": 1.0},
            {"step": 1, "value": 2.0},
        ]
        out = aggregate_by_epoch(points, 3)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["value"], 2.0)

    def test_no_span_returns_empty(self):
        self.assertEqual(aggregate_by_epoch(_points([1.0]), 0), [])
        self.assertEqual(aggregate_by_epoch(_points([1.0]), None), [])

    def test_empty_points(self):
        self.assertEqual(aggregate_by_epoch([], 3), [])

    def test_max_epochs_keeps_tail(self):
        points = _points(list(range(20)))
        out = aggregate_by_epoch(points, 2, max_epochs=3)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[-1]["epoch"], 10)


class TestEpochView(unittest.TestCase):
    def test_available_when_span_known(self):
        view = epoch_view(_points([1.0, 2.0, 3.0, 4.0]), 2)
        self.assertTrue(view["available"])
        self.assertEqual(view["steps_per_epoch"], 2)
        self.assertEqual(len(view["points"]), 2)

    def test_unavailable_when_span_unknown(self):
        view = epoch_view(_points([1.0, 2.0]), None)
        self.assertFalse(view["available"])
        self.assertEqual(view["points"], [])
        self.assertIn("epoch", view["note"])

    def test_note_mentions_span(self):
        view = epoch_view(_points([1.0, 2.0, 3.0, 4.0]), 2)
        self.assertIn("2", view["note"])


if __name__ == "__main__":
    unittest.main()