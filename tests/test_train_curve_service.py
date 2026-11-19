"""训练曲线服务：多子曲线（奖励三合一）。"""
from __future__ import annotations

import unittest

from service.train_curve_service import _metric_view, _series_view

TAGS = [
    "rollout/generated/raw_reward",
    "rollout/generated/paired_raw_reward",
    "rollout/filtered/raw_reward",
    "rollout/response_len/mean",
]


def _series(tag, values):
    return {
        tag: [
            {"step": i, "value": float(v)}
            for i, v in enumerate(values)
        ]
    }


def _all_series():
    out = {}
    for tag in TAGS:
        out.update(_series(tag, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    return out


class TestMetricView(unittest.TestCase):
    def test_reward_has_three_sub_series(self):
        view = _metric_view("raw_reward", TAGS, _all_series())
        self.assertEqual(view["label"], "奖励")
        self.assertEqual(view["n_series"], 3)
        self.assertEqual(
            [s["tag"] for s in view["sub_series"]],
            [
                "rollout/generated/raw_reward",
                "rollout/generated/paired_raw_reward",
                "rollout/filtered/raw_reward",
            ],
        )

    def test_single_tag_metric_has_one_series(self):
        view = _metric_view("response_length", TAGS, _all_series())
        self.assertEqual(view["n_series"], 1)
        self.assertEqual(view["sub_series"][0]["tag"], "rollout/response_len/mean")

    def test_missing_metric_has_no_data(self):
        view = _metric_view("rollout_time", TAGS, _all_series())
        self.assertFalse(view["has_data"])
        self.assertEqual(view["n_series"], 0)

    def test_series_label_strips_common_prefix(self):
        view = _metric_view("response_length", TAGS, _all_series())
        self.assertEqual(view["sub_series"][0]["label"], "response_len/mean")


class TestSeriesView(unittest.TestCase):
    def test_step_points_preserved(self):
        points = _series("x", [1.0, 2.0, 3.0])["x"]
        view = _series_view("x", points)
        self.assertEqual([p["step"] for p in view["points"]], [0, 1, 2])
        self.assertEqual(view["n_points"], 3)


if __name__ == "__main__":
    unittest.main()
