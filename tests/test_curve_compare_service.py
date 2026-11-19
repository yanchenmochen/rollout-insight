"""多任务曲线对比服务。"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from service.curve_compare_service import (
    _series_label,
    _task_metrics,
    build_curve_compare,
    list_comparable_tasks,
    list_curve_compare_brief,
)


def _fake_series(tag_values):
    """构造 ``{tag: [{"step":..,"value":..}]}``。"""
    return {
        tag: [{"step": i, "value": float(v)} for i, v in enumerate(vals)]
        for tag, vals in tag_values.items()
    }


class TestSeriesLabel(unittest.TestCase):
    def test_strips_common_prefix(self):
        self.assertEqual(_series_label("rollout/generated/raw_reward"), "generated/raw_reward")
        self.assertEqual(_series_label("train/grad_norm"), "grad_norm")
        self.assertEqual(_series_label("plain_tag"), "plain_tag")


class TestTaskMetrics(unittest.TestCase):
    def test_one_metric_multiple_sub_series(self):
        """「奖励」命中三个 tag，应产出三条子曲线。"""
        series = _fake_series(
            {
                "rollout/generated/raw_reward": [0.1, 0.2],
                "rollout/generated/paired_raw_reward": [0.2, 0.3],
                "rollout/filtered/raw_reward": [0.05, 0.15],
            }
        )
        task = {"task_id": 1, "task_name": "t1"}
        metrics = _task_metrics(task, ["raw_reward"], series, None, list(series.keys()))
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["key"], "raw_reward")
        self.assertEqual(len(metrics[0]["sub_series"]), 3)

    def test_metric_without_match_skipped(self):
        task = {"task_id": 1, "task_name": "t1"}
        metrics = _task_metrics(task, ["grad_norm"], {}, None, [])
        self.assertEqual(metrics, [])

    def test_custom_metric_exact_tag(self):
        series = _fake_series({"my/custom_tag": [1.0, 2.0]})
        task = {"task_id": 1, "task_name": "t1"}
        metrics = _task_metrics(
            task, ["custom:my/custom_tag"], series, None, list(series.keys())
        )
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["label"], "my/custom_tag")


class TestBuildCurveCompare(unittest.TestCase):
    def _patch_sources(self, tags_by_task):
        """打桩：让 ``list_comparable_tasks`` 与事件读取返回固定数据。"""
        tasks = [
            {"task_id": tid, "task_name": f"task-{tid}", "event_dir": f"/ev/{tid}"}
            for tid in tags_by_task
        ]
        return (
            patch(
                "service.curve_compare_service.list_comparable_tasks",
                return_value=tasks,
            ),
            patch(
                "service.curve_compare_service._resolve_steps_per_epoch",
                return_value=None,
            ),
            patch(
                "utils.tb_series_reader.read_scalar_series",
                side_effect=lambda event_dir: _fake_series(tags_by_task[int(event_dir.split("/")[-1])]),
            ),
        )

    def test_multi_task_overlay(self):
        tags_by_task = {
            1: {"rollout/generated/raw_reward": [0.1, 0.2]},
            2: {"rollout/generated/raw_reward": [0.3, 0.4]},
        }
        p1, p2, p3 = self._patch_sources(tags_by_task)
        with p1, p2, p3:
            data = build_curve_compare([1, 2], ["raw_reward"], "step")
        self.assertEqual(len(data["tasks"]), 2)
        self.assertEqual(len(data["metrics"]), 1)
        series = data["metrics"][0]["series"]
        self.assertEqual(len(series), 2)
        self.assertEqual({s["task_id"] for s in series}, {1, 2})
        self.assertEqual(data["view"], "step")

    def test_filters_by_task_ids(self):
        tags_by_task = {
            1: {"rollout/generated/raw_reward": [0.1]},
            2: {"rollout/generated/raw_reward": [0.3]},
        }
        p1, p2, p3 = self._patch_sources(tags_by_task)
        with p1, p2, p3:
            data = build_curve_compare([2], ["raw_reward"], "step")
        self.assertEqual(len(data["tasks"]), 1)
        self.assertEqual(data["tasks"][0]["task_id"], 2)
        self.assertEqual(len(data["metrics"][0]["series"]), 1)

    def test_metric_without_data_dropped(self):
        tags_by_task = {1: {"rollout/generated/raw_reward": [0.1]}}
        p1, p2, p3 = self._patch_sources(tags_by_task)
        with p1, p2, p3:
            data = build_curve_compare([1], ["raw_reward", "grad_norm"], "step")
        self.assertEqual([m["key"] for m in data["metrics"]], ["raw_reward"])


class TestListComparableTasks(unittest.TestCase):
    def test_skips_tasks_without_event_dir(self):
        from unittest.mock import MagicMock

        task_ok = MagicMock()
        task_ok.primary_id = 7
        task_ok.task_name = "with-tb"
        task_ok.rollout_root = "/tmp/x"

        task_bad = MagicMock()
        task_bad.primary_id = 8
        task_bad.task_name = "no-tb"
        task_bad.rollout_root = ""

        with patch(
            "db.db_task_service.DBTaskService.list_tasks",
            return_value=[task_ok, task_bad],
        ), patch(
            "service.train_curve_service.resolve_event_dir",
            side_effect=lambda root: "/ev/x" if root else None,
        ):
            tasks = list_comparable_tasks()
        self.assertEqual([t["task_id"] for t in tasks], [7])

    def test_brief_payload_shape(self):
        with patch(
            "service.curve_compare_service.list_comparable_tasks",
            return_value=[{"task_id": 1, "task_name": "t", "event_dir": "/ev"}],
        ):
            data = list_curve_compare_brief()
        self.assertIn("tasks", data)
        self.assertIn("available_metrics", data)
        self.assertIn("default_keys", data)
        self.assertEqual(len(data["tasks"]), 1)


if __name__ == "__main__":
    unittest.main()
