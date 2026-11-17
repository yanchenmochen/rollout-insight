"""指标说明只命中已知评测 tag；未知 tag 不编造含义。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.tb_deep_link import (  # noqa: E402
    append_tag_filter,
    tag_filter_supported,
    tensorboard_tag_filter,
)
from utils.tb_metric_catalog import (  # noqa: E402
    DESC_ATTEMPT_COUNT,
    DESC_PRIMARY,
    DESC_TRUNCATED,
    DESC_UNKNOWN,
    describe_tb_tag,
)
from service.insight_page_service import read_latest_scalars  # noqa: E402


class TestDescribeTbTag(unittest.TestCase):
    def test_primary_eval_score(self) -> None:
        self.assertEqual(describe_tb_tag("eval/gsm8k"), DESC_PRIMARY)
        self.assertEqual(describe_tb_tag("eval/livecodebench"), DESC_PRIMARY)
        self.assertEqual(describe_tb_tag("  eval/aime  "), DESC_PRIMARY)

    def test_truncated_ratio_is_not_primary(self) -> None:
        self.assertEqual(
            describe_tb_tag("eval/gsm8k-truncated_ratio"),
            DESC_TRUNCATED,
        )
        self.assertEqual(
            describe_tb_tag("eval/foo/bar-truncated_ratio"),
            DESC_TRUNCATED,
        )

    def test_attempt_count(self) -> None:
        self.assertEqual(describe_tb_tag("eval/gsm8k/attempt_count"), DESC_ATTEMPT_COUNT)
        # 主分数正则也能匹配这一条，但仍按尝试次数说明。
        self.assertEqual(describe_tb_tag("eval/attempt_count"), DESC_ATTEMPT_COUNT)

    def test_unknown_tags_are_not_invented(self) -> None:
        for tag in (
            "train/loss",
            "train/kl",
            "actor/kl_loss",
            "loss",
            "eval/foo/bar",
            "eval/foo-bar",
            "train/gsm8k-truncated_ratio",
            "train/attempt_count",
            "",
        ):
            self.assertEqual(describe_tb_tag(tag), DESC_UNKNOWN, tag)


class TestReadLatestScalars(unittest.TestCase):
    def test_missing_dir_returns_message(self) -> None:
        rows, err = read_latest_scalars("/tmp/rollout-insight-no-such-tb-dir")
        self.assertEqual(rows, [])
        self.assertTrue(err)

    def test_empty_dir_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows, err = read_latest_scalars(tmp)
        self.assertEqual(rows, [])
        self.assertTrue(err is None or isinstance(err, str))


class TestTensorboardTagLink(unittest.TestCase):
    def test_checked_version_uses_query_not_hash(self) -> None:
        self.assertTrue(tag_filter_supported("2.20.0"))
        self.assertFalse(tag_filter_supported("2.11.2"))
        self.assertFalse(tag_filter_supported(""))
        filt = tensorboard_tag_filter("eval/gsm8k")
        self.assertEqual(filt, "^eval/gsm8k$")
        self.assertNotIn("#timeseries", filt)
        url = append_tag_filter("http://10.0.0.1:6006/#timeseries", "train/loss")
        parts = urlsplit(url)
        self.assertEqual(parts.fragment, "")
        self.assertNotIn("timeseries", parts.fragment)
        self.assertEqual(parse_qs(parts.query)["tagFilter"], ["^train/loss$"])


if __name__ == "__main__":
    unittest.main()
