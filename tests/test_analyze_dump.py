"""utils.analyze_dump 的测试用例。"""
import os
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from utils.analyze_dump import (
    load_dump,
    load_dump_meta,
    split_turns,
    summarize_dump,
    render_transcript,
)


DUMP_PATH = Path(
    "/mnt/zj-data/data/songquanheng/output/slime/"
    "021_32b_agentic_scale_swe_1000_optim_20261001_195820/rollout_dumps/rollout_0.pt"
)

LIVE_DUMP_PATH = Path(
    "/mnt/seed-program-nas/001688/songquanheng/output/slime/runs/"
    "20261104_113329_c021-f1base-rollout/generated/rollout_dumps/rollout_0.pt"
)


@unittest.skipUnless(DUMP_PATH.is_file(), f"rollout dump 不存在：{DUMP_PATH}")
class TestAnalyzeDump(unittest.TestCase):
    """基于真实 rollout_0.pt 的解析测试。"""

    def setUp(self) -> None:
        # 确保 tokenizer 库在多线程下静默运行
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def test_load_dump_meta_keys(self) -> None:
        meta: Dict[str, Any] = load_dump_meta(DUMP_PATH)
        expected = {"rollout_id", "n_samples", "n_attempts", "mean_reward", "pt_path"}
        self.assertTrue(
            expected.issubset(meta.keys()),
            f"missing keys: {expected - set(meta.keys())}",
        )
        self.assertEqual(meta["pt_path"], str(DUMP_PATH))
        self.assertIsInstance(meta["rollout_id"], int)
        self.assertGreater(meta["n_samples"], 0)
        self.assertGreater(meta["n_attempts"], 0)
        self.assertIsInstance(meta["mean_reward"], float)

    def test_summarize_dump_shape(self) -> None:
        rows: List[Dict[str, Any]] = summarize_dump(DUMP_PATH)
        self.assertGreater(len(rows), 0)
        first = rows[0]
        expected = {
            "idx", "group_index", "rollout_id", "segment_index", "segment_count",
            "instance_id", "reward", "status", "agent_turns", "tool_calls",
            "diff_line_count", "response_length", "mask_sum", "sandbox_id",
        }
        self.assertTrue(
            expected.issubset(first.keys()),
            f"missing keys: {expected - set(first.keys())}",
        )
        # idx 必须与行号一致
        for i, row in enumerate(rows):
            self.assertEqual(row["idx"], i)

    def test_split_turns_smoke(self) -> None:
        text = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nhi<|im_end|>"
        turns = split_turns(text)
        self.assertGreaterEqual(len(turns), 1)

    def test_render_transcript_idx_zero(self) -> None:
        # 不传 tokenizer，回落到 legacy 渲染：要求不抛错
        out = render_transcript(DUMP_PATH, 0, include_prompt=False)
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)


@unittest.skipUnless(LIVE_DUMP_PATH.is_file(), f"live dump 不存在：{LIVE_DUMP_PATH}")
class TestAnalyzeDumpLiveNoTorchSkill(unittest.TestCase):
    """不依赖 Cursor skill / torch 也能解析当前训练落盘。"""

    def test_load_and_summarize_without_external_skill(self) -> None:
        meta = load_dump_meta(LIVE_DUMP_PATH)
        rows = summarize_dump(LIVE_DUMP_PATH)
        self.assertGreater(meta["n_samples"], 0)
        self.assertEqual(len(rows), meta["n_samples"])
        self.assertIsNotNone(rows[0]["segment_index"])


class TestAnalyzeDumpSyntheticZip(unittest.TestCase):
    """用最小 zip/.pt 验证无 torch 路径与 None segment_index 兜底。"""

    def test_zip_pickle_and_null_segment_index(self) -> None:
        dump = {
            "rollout_id": 7,
            "samples": [
                {
                    "group_index": 0,
                    "rollout_id": 11,
                    "reward": 1.0,
                    "status": "Status.FINISHED",
                    "response_length": 3,
                    "loss_mask": [1, 0, 1],
                    "response": "<|im_start|>assistant\nok<|im_end|>",
                    "metadata": {
                        "segment_index": None,
                        "segment_count": None,
                        "instance_id": "i0",
                        "sandbox_id": "s0",
                        "agent_turns": 1,
                        "tool_calls": 0,
                        "diff_line_count": 0,
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            pt_path = Path(tmp) / "rollout_7.pt"
            with zipfile.ZipFile(pt_path, "w") as zf:
                zf.writestr("rollout_7/data.pkl", pickle.dumps(dump))
                zf.writestr("rollout_7/byteorder", "little")
            meta = load_dump_meta(pt_path)
            rows = summarize_dump(pt_path)
        self.assertEqual(meta["rollout_id"], 7)
        self.assertEqual(meta["n_samples"], 1)
        self.assertEqual(rows[0]["segment_index"], 0)
        self.assertEqual(rows[0]["segment_count"], 1)
        self.assertEqual(rows[0]["mask_sum"], 2)


class TestImportOnly(unittest.TestCase):
    """即使 .pt 不存在，也要保证 import 路径无副作用。"""

    def test_import(self) -> None:
        from utils import analyze_dump  # noqa: F401

        self.assertTrue(hasattr(analyze_dump, "load_dump_meta"))
        self.assertTrue(hasattr(analyze_dump, "load_dump"))
        self.assertTrue(callable(load_dump))


if __name__ == "__main__":
    unittest.main()
