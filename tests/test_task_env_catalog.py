#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""utils.task_env_catalog 单测。"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("ROLLOUT_INSIGHT_CONFIG_PATH", "")

from utils.task_env_catalog import (  # noqa: E402
    compare_env_maps,
    diff_env_maps,
    is_sensitive_key,
    pick_highlight_env,
    redact_env_value,
    summarize_env_map,
)


class TestRedact(unittest.TestCase):
    def test_sensitive_key(self) -> None:
        self.assertTrue(is_sensitive_key("NANHU_TOKEN"))
        self.assertTrue(is_sensitive_key("api_password"))
        self.assertFalse(is_sensitive_key("ROLLOUT_BATCH_SIZE"))

    def test_redact_masks(self) -> None:
        out = redact_env_value("API_TOKEN", "abcdefgh")
        self.assertIn("****", out)
        self.assertNotEqual(out, "abcdefgh")


class TestHighlight(unittest.TestCase):
    def test_pick_known_keys(self) -> None:
        env = {
            "ROLLOUT_BATCH_SIZE": "16",
            "SWE_AGENT_TIME_BUDGET_SEC": "300",
            "UNKNOWN_FOO": "1",
        }
        rows = pick_highlight_env(env)
        present = {r["catalog_key"]: r for r in rows if r["present"]}
        self.assertIn("ROLLOUT_BATCH_SIZE", present)
        self.assertEqual(present["ROLLOUT_BATCH_SIZE"]["value"], "16")
        self.assertIn("SWE_AGENT_TIME_BUDGET_SEC", present)
        # 未知键默认不进 highlight
        self.assertTrue(all(r["catalog_key"] != "UNKNOWN_FOO" for r in rows if r["present"]))

    def test_case_insensitive(self) -> None:
        env = {"rollout_batch_size": "8"}
        rows = pick_highlight_env(env)
        hit = [r for r in rows if r["present"] and r["catalog_key"] == "ROLLOUT_BATCH_SIZE"]
        self.assertEqual(len(hit), 1)
        self.assertEqual(hit[0]["value"], "8")

    def test_summarize_counts(self) -> None:
        summary = summarize_env_map(
            {
                "DEBUG_ROLLOUT_ONLY": "0",
                "TRAINING_SCRIPT": "musa_run/run_x.sh",
                "EXTRA": "1",
            }
        )
        self.assertEqual(summary["env_count"], 3)
        self.assertGreaterEqual(summary["highlight_count"], 2)


class TestDiff(unittest.TestCase):
    def test_changed_and_only(self) -> None:
        left = {
            "ROLLOUT_BATCH_SIZE": "16",
            "SWE_AGENT_TIME_BUDGET_SEC": "300",
            "LR": "1e-6",
        }
        right = {
            "ROLLOUT_BATCH_SIZE": "32",
            "SWE_AGENT_TIME_BUDGET_SEC": "300",
            "KL_COEF": "0.01",
        }
        diff = diff_env_maps(left, right, left_label="A", right_label="B")
        changed_keys = {r["catalog_key"] for r in diff["changed"]}
        self.assertIn("ROLLOUT_BATCH_SIZE", changed_keys)
        only_left = {r["catalog_key"] for r in diff["only_left"]}
        self.assertIn("LR", only_left)
        only_right = {r["catalog_key"] for r in diff["only_right"]}
        self.assertIn("KL_COEF", only_right)
        same_keys = {r["catalog_key"] for r in diff["same"]}
        self.assertIn("SWE_AGENT_TIME_BUDGET_SEC", same_keys)

    def test_compare_matrix(self) -> None:
        envs = [
            {"ROLLOUT_BATCH_SIZE": "16", "DEBUG_ROLLOUT_ONLY": "0"},
            {"ROLLOUT_BATCH_SIZE": "32", "DEBUG_ROLLOUT_ONLY": "0"},
            {"ROLLOUT_BATCH_SIZE": "16", "DEBUG_ROLLOUT_ONLY": "1"},
        ]
        out = compare_env_maps(envs, labels=["t1", "t2", "t3"], baseline_index=0)
        self.assertEqual(out["summary"]["task_count"], 3)
        self.assertGreaterEqual(out["summary"]["diff_key_count"], 2)
        batch = next(r for r in out["rows"] if r["catalog_key"] == "ROLLOUT_BATCH_SIZE")
        self.assertFalse(batch["all_same"])
        self.assertEqual(batch["values"][0], "16")
        self.assertEqual(batch["values"][1], "32")
        self.assertEqual(batch["statuses"][1], "changed")

    def test_empty_compare(self) -> None:
        out = compare_env_maps([])
        self.assertEqual(out["summary"]["task_count"], 0)
        self.assertEqual(out["rows"], [])


if __name__ == "__main__":
    unittest.main()
