"""训练曲线服务：多子曲线（奖励三合一）+ epoch 聚合视图。"""
from __future__ import annotations

import unittest

from service.train_curve_service import _build_eval_series, _metric_view, _series_view

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
        view = _metric_view("raw_reward", TAGS, _all_series(), 3)
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
        view = _metric_view("response_length", TAGS, _all_series(), 3)
        self.assertEqual(view["n_series"], 1)
        self.assertEqual(view["sub_series"][0]["tag"], "rollout/response_len/mean")

    def test_missing_metric_has_no_data(self):
        view = _metric_view("rollout_time", TAGS, _all_series(), 3)
        self.assertFalse(view["has_data"])
        self.assertEqual(view["n_series"], 0)

    def test_series_label_strips_common_prefix(self):
        view = _metric_view("response_length", TAGS, _all_series(), 3)
        self.assertEqual(view["sub_series"][0]["label"], "response_len/mean")

    def test_epoch_points_attached_to_each_series(self):
        view = _metric_view("raw_reward", TAGS, _all_series(), 3)
        for series in view["sub_series"]:
            self.assertEqual(series["n_epoch_points"], 2)
            # epoch 内取平均：(1+2+3)/3=2，(4+5+6)/3=5
            self.assertAlmostEqual(series["epoch_points"][0]["value"], 2.0)
            self.assertAlmostEqual(series["epoch_points"][1]["value"], 5.0)


class TestSeriesView(unittest.TestCase):
    def test_epoch_points_empty_without_span(self):
        points = _series("x", [1.0, 2.0])["x"]
        view = _series_view("x", points, None)
        self.assertEqual(view["n_epoch_points"], 0)
        self.assertEqual(view["n_points"], 2)

    def test_step_points_preserved(self):
        points = _series("x", [1.0, 2.0, 3.0])["x"]
        view = _series_view("x", points, 2)
        self.assertEqual([p["step"] for p in view["points"]], [0, 1, 2])


class TestBuildEvalSeries(unittest.TestCase):
    """外部评测：一个评测集可同时给出多个题池的分数，必须拆成多条曲线。"""

    def _round(self, step, scores, status="finished", round_name="first run"):
        return {
            "step": step,
            "round": round_name,
            "job_status": status,
            "scores": scores,
        }

    def test_weighted_and_holdout_split_into_two_series(self):
        rounds = [
            self._round(10, {"weighted": 0.660, "weighted_tasks": 429,
                             "holdout": 0.400, "holdout_tasks": 85,
                             "holdout_tasks_total": 87}),
            self._round(20, {"weighted": 0.599, "weighted_tasks": 429,
                             "holdout": 0.427, "holdout_tasks": 85,
                             "holdout_tasks_total": 87}),
        ]
        series = _build_eval_series(rounds)
        self.assertEqual([s["kind"] for s in series], ["weighted", "holdout"])
        self.assertEqual(series[0]["n_points"], 2)
        self.assertEqual(series[1]["n_points"], 2)
        # 加权分曲线用加权分，holdout 曲线用 holdout 分，互不串味
        self.assertAlmostEqual(series[0]["first_value"], 0.660)
        self.assertAlmostEqual(series[1]["first_value"], 0.400)
        self.assertAlmostEqual(series[1]["last_value"], 0.427)

    def test_series_carries_task_pool_info(self):
        rounds = [
            self._round(10, {"weighted": 0.660, "weighted_tasks": 429,
                             "holdout": 0.400, "holdout_tasks": 85,
                             "holdout_tasks_total": 87}),
        ]
        series = _build_eval_series(rounds)
        weighted = series[0]
        holdout = series[1]
        self.assertEqual(weighted["tasks"], 429)
        self.assertEqual(holdout["tasks"], 87)
        self.assertEqual(holdout["scored_tasks"], 85)

    def test_merged_form_single_series(self):
        rounds = [
            self._round(50, {"merged": 0.596, "merged_pass": 298, "merged_total": 500}),
        ]
        series = _build_eval_series(rounds)
        self.assertEqual([s["kind"] for s in series], ["merged"])
        self.assertEqual(series[0]["tasks"], 500)
        self.assertEqual(series[0]["scored_tasks"], 298)

    def test_records_without_score_are_skipped(self):
        rounds = [
            self._round(30, {}, status="error", round_name="failed start"),
            self._round(40, {"weighted": 0.588, "weighted_tasks": 429}),
        ]
        series = _build_eval_series(rounds)
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0]["n_points"], 1)
        self.assertEqual(series[0]["first_step"], 40)

    def test_points_sorted_by_step(self):
        rounds = [
            self._round(40, {"weighted": 0.588}),
            self._round(10, {"weighted": 0.660}),
            self._round(20, {"weighted": 0.599}),
        ]
        series = _build_eval_series(rounds)
        self.assertEqual([p["step"] for p in series[0]["points"]], [10, 20, 40])

    def test_delta_is_last_minus_first(self):
        rounds = [
            self._round(10, {"weighted": 0.660}),
            self._round(70, {"weighted": 0.583}),
        ]
        series = _build_eval_series(rounds)
        self.assertAlmostEqual(series[0]["delta"], -0.077, places=4)

    def test_empty_rounds(self):
        self.assertEqual(_build_eval_series([]), [])


if __name__ == "__main__":
    unittest.main()