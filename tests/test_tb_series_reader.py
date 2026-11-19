"""utils.tb_series_reader：读取 TB 事件目录的完整 scalar 序列。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.tb_series_reader import (
    _downsample,
    list_scalar_tags,
    read_scalar_series,
    series_points,
)

try:  # tensorboard 可选依赖
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    _HAS_TB = True
except Exception:  # noqa: BLE001
    event_pb2 = None
    summary_pb2 = None
    EventFileWriter = None
    _HAS_TB = False


def _write_event_dir(tmpdir: Path, tag: str, values):
    """写一个 TB 事件文件，返回事件目录。"""
    event_dir = tmpdir / "tbs"
    event_dir.mkdir(parents=True, exist_ok=True)
    writer = EventFileWriter(str(event_dir))
    for step, value in enumerate(values, start=1):
        summary = event_pb2.Event(
            summary=summary_pb2.Summary(
                value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(value))]
            ),
            step=step,
            wall_time=1700000000.0 + step,
        )
        writer.add_event(summary)
    writer.flush()
    writer.close()
    return event_dir


class TestDownsample(unittest.TestCase):
    def test_keeps_all_when_within_limit(self):
        points = [{"step": i, "value": i} for i in range(5)]
        self.assertEqual(_downsample(points, 10), points)

    def test_keeps_first_and_last(self):
        points = [{"step": i, "value": i} for i in range(100)]
        picked = _downsample(points, 10)
        self.assertEqual(len(picked), 10)
        self.assertEqual(picked[0]["step"], 0)
        self.assertEqual(picked[-1]["step"], 99)

    def test_zero_limit_keeps_all(self):
        points = [{"step": i, "value": i} for i in range(50)]
        self.assertEqual(_downsample(points, 0), points)


class TestSeriesPoints(unittest.TestCase):
    def test_missing_tag_returns_empty(self):
        self.assertEqual(series_points({}, "nope"), [])

    def test_returns_points(self):
        points = [{"step": 1, "value": 0.5}]
        self.assertEqual(series_points({"a": points}, "a"), points)


class TestReadScalarSeries(unittest.TestCase):
    def test_missing_dir_returns_empty(self):
        self.assertEqual(read_scalar_series("/nonexistent/path"), {})
        self.assertEqual(list_scalar_tags("/nonexistent/path"), [])

    def test_none_returns_empty(self):
        self.assertEqual(read_scalar_series(None), {})

    @unittest.skipUnless(_HAS_TB, "tensorboard 未安装")
    def test_reads_full_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = _write_event_dir(Path(tmp), "rollout/mixed_reward", [0.1, 0.2, 0.3])
            series = read_scalar_series(event_dir)
            self.assertIn("rollout/mixed_reward", series)
            points = series["rollout/mixed_reward"]
            self.assertEqual(len(points), 3)
            self.assertEqual(points[0]["step"], 1)
            self.assertAlmostEqual(points[2]["value"], 0.3)

    @unittest.skipUnless(_HAS_TB, "tensorboard 未安装")
    def test_list_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = _write_event_dir(Path(tmp), "actor/grad_norm", [1.0, 2.0])
            tags = list_scalar_tags(event_dir)
            self.assertIn("actor/grad_norm", tags)

    @unittest.skipUnless(_HAS_TB, "tensorboard 未安装")
    def test_filter_by_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = _write_event_dir(Path(tmp), "a", [1.0])
            series = read_scalar_series(event_dir, tags=["a", "missing"])
            self.assertEqual(list(series.keys()), ["a"])

    @unittest.skipUnless(_HAS_TB, "tensorboard 未安装")
    def test_max_points_downsamples(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = _write_event_dir(Path(tmp), "x", list(range(50)))
            series = read_scalar_series(event_dir, max_points=10)
            points = series["x"]
            self.assertLessEqual(len(points), 11)
            self.assertEqual(points[0]["step"], 1)
            self.assertEqual(points[-1]["step"], 50)

    @unittest.skipUnless(_HAS_TB, "tensorboard 未安装")
    def test_no_random_point_loss_when_within_reservoir(self):
        """TB 的 Reservoir 是随机抽样；点数不超过容量时必须完整保留。"""
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = _write_event_dir(Path(tmp), "x", list(range(300)))
            series = read_scalar_series(event_dir, max_points=0)
            steps = [p["step"] for p in series["x"]]
            self.assertEqual(steps, list(range(1, 301)))


if __name__ == "__main__":
    unittest.main()
