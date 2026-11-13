"""``utils.slime_job_type`` / ``eval_tb_scores`` 单测。"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("ROLLOUT_INSIGHT_CONFIG_PATH", "")

from utils.eval_tb_scores import is_primary_eval_tag  # noqa: E402
from utils.slime_job_type import (  # noqa: E402
    classify_slime_job_type,
    load_env_from_run_dir,
)


class TestClassifySlimeJobType(unittest.TestCase):
    def test_eval_by_training_script(self) -> None:
        out = classify_slime_job_type(
            {
                "TRAINING_SCRIPT": (
                    "/mnt/x/nanhu-slime/musa_run/eval/run_021_32b_eval.sh"
                ),
                "DEBUG_ROLLOUT_ONLY": "1",
            }
        )
        self.assertEqual(out["job_type"], "eval")
        self.assertEqual(out["job_type_label"], "评测")

    def test_eval_by_launcher(self) -> None:
        out = classify_slime_job_type(
            {"DEBUG_ROLLOUT_ONLY": "1", "COLOCATE": "0"},
            launcher="musa_run/eval/run_021_32b_eval.sh",
        )
        self.assertEqual(out["job_type"], "eval")

    def test_eval_by_num_rollout_zero(self) -> None:
        out = classify_slime_job_type(
            {"NUM_ROLLOUT": "0", "DISABLE_EVAL": "0", "DEBUG_ROLLOUT_ONLY": "1"}
        )
        self.assertEqual(out["job_type"], "eval")

    def test_eval_by_slime_eval_config_combo(self) -> None:
        out = classify_slime_job_type(
            {
                "SLIME_EVAL_CONFIG": "/path/to/panel.yaml",
                "DEBUG_ROLLOUT_ONLY": "1",
                "COLOCATE": "0",
            }
        )
        self.assertEqual(out["job_type"], "eval")

    def test_rollout_only(self) -> None:
        out = classify_slime_job_type(
            {
                "TRAINING_SCRIPT": "musa_run/run_021_32b_scale_swe.sh",
                "DEBUG_ROLLOUT_ONLY": "1",
                "DISABLE_EVAL": "1",
                "ROLLOUT_SHUFFLE": "0",
            }
        )
        self.assertEqual(out["job_type"], "rollout_only")

    def test_train_debug_rollout_only_zero(self) -> None:
        out = classify_slime_job_type(
            {
                "TRAINING_SCRIPT": "musa_run/run_021_32b_scale_swe.sh",
                "DEBUG_ROLLOUT_ONLY": "0",
                "DISABLE_EVAL": "1",
                "COLOCATE": "1",
            }
        )
        self.assertEqual(out["job_type"], "train")
        self.assertEqual(out["job_type_label"], "真实训练")
        self.assertTrue(any("DEBUG_ROLLOUT_ONLY=0" in s for s in out["signals"]))
        self.assertTrue(any("TRAINING_SCRIPT=" in s for s in out["signals"]))

    def test_train_debug_rollout_only_unset(self) -> None:
        out = classify_slime_job_type(
            {
                "TRAINING_SCRIPT": "musa_run/run_021_32b_scale_swe.sh",
                "DISABLE_EVAL": "1",
            }
        )
        self.assertEqual(out["job_type"], "train")

    def test_shuffle_alone_not_train(self) -> None:
        """不以 ROLLOUT_SHUFFLE=1 单独判定为真实训练。"""
        out = classify_slime_job_type({"ROLLOUT_SHUFFLE": "1", "DISABLE_EVAL": "1"})
        self.assertEqual(out["job_type"], "unknown")

    def test_unknown_empty(self) -> None:
        out = classify_slime_job_type({})
        self.assertEqual(out["job_type"], "unknown")


class TestLoadEnvFromRunDir(unittest.TestCase):
    def test_reads_resolved_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "resolved_config.yaml").write_text(
                "\n".join(
                    [
                        "launcher: musa_run/eval/run_021_32b_eval.sh",
                        "env:",
                        "  DEBUG_ROLLOUT_ONLY: '1'",
                        "  COLOCATE: '0'",
                        "  SLIME_EVAL_CONFIG: /tmp/panel.yaml",
                    ]
                ),
                encoding="utf-8",
            )
            env, launcher, source = load_env_from_run_dir(root)
            self.assertEqual(launcher, "musa_run/eval/run_021_32b_eval.sh")
            self.assertEqual(env.get("DEBUG_ROLLOUT_ONLY"), "1")
            self.assertEqual(env.get("SLIME_EVAL_CONFIG"), "/tmp/panel.yaml")
            self.assertIn("resolved_config", source)
            typed = classify_slime_job_type(env, launcher=launcher)
            self.assertEqual(typed["job_type"], "eval")


class TestPrimaryEvalTag(unittest.TestCase):
    def test_primary_yes(self) -> None:
        self.assertTrue(is_primary_eval_tag("eval/swebench_verified_panel150"))
        self.assertTrue(is_primary_eval_tag("eval/scale_swe_holdout87"))

    def test_primary_no(self) -> None:
        self.assertFalse(is_primary_eval_tag("eval/panel/attempt_count"))
        self.assertFalse(
            is_primary_eval_tag("eval/swebench_verified_panel150-truncated_ratio")
        )
        self.assertFalse(is_primary_eval_tag("train/loss"))


if __name__ == "__main__":
    unittest.main()
