"""rollout_pt_parser / parse_attempt_token_stats 在无 torch 环境下可用。"""
from __future__ import annotations

import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

LIVE_DUMP_PATH = Path(
    "/mnt/seed-program-nas/001688/songquanheng/output/slime/runs/"
    "20261104_113329_c021-f1base-rollout/generated/rollout_dumps/rollout_0.pt"
)


class TestRolloutPtParserImportNoTorch(unittest.TestCase):
    """模块 import 与纯文本切分不得触发 torch。"""

    def test_import_and_split_without_torch(self) -> None:
        import sys

        # 即使 torch 不可用，本模块与切分函数仍应可加载
        broken = mock.Mock()
        broken.__spec__ = None
        with mock.patch.dict(sys.modules, {"torch": broken}):
            for name in (
                "utils.rollout_pt_parser",
                "service.rollout_parse_service",
            ):
                sys.modules.pop(name, None)
            from utils.rollout_pt_parser import (  # noqa: WPS433
                split_markdown_headed_turns,
                split_response_to_turns,
            )

            text = (
                "<|im_start|>user\nhello<|im_end|>\n"
                "<|im_start|>assistant\nhi<|im_end|>"
            )
            turns = split_response_to_turns(text)
            self.assertGreaterEqual(len(turns), 2)
            md = "### turn 1 · user\nA\n### turn 2 · assistant\nB\n"
            self.assertGreaterEqual(len(split_markdown_headed_turns(md)), 2)


class TestLoadPtZipPickle(unittest.TestCase):
    def test_load_pt_synthetic_zip(self) -> None:
        from utils.rollout_pt_parser import load_pt

        dump = {
            "rollout_id": 9,
            "samples": [
                {
                    "group_index": 0,
                    "rollout_id": 1,
                    "reward": 1.0,
                    "response": "<|im_start|>assistant\nok<|im_end|>",
                    "metadata": {"segment_index": 0, "segment_count": 1},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            pt_path = Path(tmp) / "rollout_9.pt"
            with zipfile.ZipFile(pt_path, "w") as zf:
                zf.writestr("rollout_9/data.pkl", pickle.dumps(dump))
                zf.writestr("rollout_9/byteorder", "little")
            loaded = load_pt(str(pt_path))
        self.assertEqual(loaded["rollout_id"], 9)
        self.assertEqual(len(loaded["samples"]), 1)


@unittest.skipUnless(LIVE_DUMP_PATH.is_file(), f"live dump 不存在：{LIVE_DUMP_PATH}")
class TestLoadPtLiveNoTorch(unittest.TestCase):
    def test_load_live_dump(self) -> None:
        from utils.rollout_pt_parser import load_pt, split_response_to_turns

        dump = load_pt(str(LIVE_DUMP_PATH))
        samples = dump.get("samples") or []
        self.assertGreater(len(samples), 0)
        resp = samples[0].get("response") or ""
        if isinstance(resp, str) and "<|im_start|>" in resp:
            self.assertGreaterEqual(len(split_response_to_turns(resp)), 1)


class TestParseAttemptTokenStatsImport(unittest.TestCase):
    def test_service_imports_without_torch(self) -> None:
        import sys

        for name in list(sys.modules):
            if name in (
                "service.rollout_parse_service",
                "utils.rollout_pt_parser",
            ) or name.startswith("utils.rollout_pt_parser."):
                sys.modules.pop(name, None)
        from service.rollout_parse_service import parse_attempt_token_stats

        self.assertTrue(callable(parse_attempt_token_stats))


if __name__ == "__main__":
    unittest.main()
