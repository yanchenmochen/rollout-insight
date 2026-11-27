#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""utils.agent_budget_profile 单测：锁死 task8 / task9 式判定。"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("ROLLOUT_INSIGHT_CONFIG_PATH", "")

from utils.agent_budget_profile import (  # noqa: E402
    build_agent_budget_profile,
    classify_degradation,
    normalize_attempt_row,
    percentile,
)


def _row(
    rid: int,
    *,
    turns: float,
    per_turn_s: float,
    tokens_per_turn: float,
    time_budget: bool = False,
    agent_run: Optional[float] = None,
    peak_ctx: float = 70000,
) -> dict:
    turns_i = max(1, int(turns))
    model_gen = per_turn_s * turns_i
    run = agent_run if agent_run is not None else model_gen * 1.1
    return {
        "rollout_id": rid,
        "agent_turns": turns_i,
        "agent_outcome": "time_budget_exceeded" if time_budget else "finished",
        "phase_times": {
            "phase_time/agent_run": run,
            "phase_time/model_generation": model_gen,
            "phase_time/attempt_total": run + 300,
            "phase_time/sandbox_boot_queue": 160,
        },
        "attempt_generated_response_tokens": tokens_per_turn * turns_i,
        "attempt_peak_context_tokens": peak_ctx,
        "context_limit_tokens": 131072,
    }


def _series_like_task8() -> list:
    """模拟 task8：单轮变慢 + token 涨 + 轮数降 + 撞墙升高。"""
    rows = []
    # 10 个 rollout，前段健康，后段退化
    specs = [
        (0, 62, 18.0, 350, 0.0),
        (1, 60, 20.0, 380, 0.0),
        (2, 58, 25.0, 420, 0.05),
        (3, 55, 35.0, 520, 0.12),
        (4, 50, 45.0, 650, 0.20),
        (5, 45, 60.0, 780, 0.35),
        (6, 40, 75.0, 870, 0.50),
        (7, 38, 82.0, 920, 0.55),
        (8, 36, 86.0, 960, 0.60),
        (9, 35, 80.0, 940, 0.45),
    ]
    for rid, turns, pts, tpt, tb_ratio in specs:
        n = 20
        n_tb = int(round(tb_ratio * n))
        for i in range(n):
            rows.append(
                _row(
                    rid,
                    turns=turns,
                    per_turn_s=pts,
                    tokens_per_turn=tpt,
                    time_budget=(i < n_tb),
                    peak_ctx=88000 - rid * 1500,
                )
            )
    return rows


def _series_like_task9() -> list:
    """模拟 task9：轻度变慢但不够阈值，撞墙很低。"""
    rows = []
    specs = [
        (0, 59, 23.5, 384, 0.0),
        (1, 59, 25.0, 420, 0.0),
        (2, 60, 28.0, 480, 0.01),
        (3, 60, 30.0, 520, 0.02),
        (4, 60, 32.0, 560, 0.02),
        (5, 60, 34.0, 580, 0.03),
        (6, 60, 35.0, 600, 0.03),
        (7, 60, 35.5, 602, 0.04),
        (8, 60, 35.7, 602, 0.05),
        (9, 60, 35.7, 602, 0.05),
    ]
    for rid, turns, pts, tpt, tb_ratio in specs:
        n = 20
        n_tb = int(round(tb_ratio * n))
        for i in range(n):
            rows.append(
                _row(
                    rid,
                    turns=turns,
                    per_turn_s=pts,
                    tokens_per_turn=tpt,
                    time_budget=(i < n_tb),
                    peak_ctx=80000,
                )
            )
    return rows


class TestNormalize(unittest.TestCase):
    def test_phase_times_prefix(self) -> None:
        row = normalize_attempt_row(
            {
                "rollout_id": 1,
                "agent_turns": 10,
                "phase_times": {
                    "phase_time/agent_run": 1000,
                    "phase_time/model_generation": 800,
                },
            }
        )
        self.assertEqual(row["agent_run_s"], 1000)
        self.assertAlmostEqual(row["per_turn_seconds"], 80.0)
        self.assertEqual(row["per_turn_source"], "model_generation")

    def test_nested_phase_time(self) -> None:
        row = normalize_attempt_row(
            {
                "rollout_id": 1,
                "turns": 5,
                "metadata": {"phase_time": {"agent_run": 500, "model_generation": 400}},
            }
        )
        self.assertEqual(row["agent_run_s"], 500)
        self.assertAlmostEqual(row["per_turn_seconds"], 80.0)


class TestPercentile(unittest.TestCase):
    def test_median(self) -> None:
        self.assertEqual(percentile([1, 2, 3], 0.5), 2)


class TestDegradationVerdict(unittest.TestCase):
    def test_task8_like_per_turn_slowdown(self) -> None:
        profile = build_agent_budget_profile(_series_like_task8(), min_samples=3)
        deg = profile["degradation"]
        self.assertEqual(deg["verdict"], "per_turn_slowdown")
        self.assertGreaterEqual(deg["per_turn_growth"], 0.50)
        self.assertGreaterEqual(deg["tokens_per_turn_growth"], 0.30)
        self.assertLessEqual(deg["turns_growth"], 0.10)
        self.assertEqual(deg["slowdown_attribution"], "model_output_growth")

    def test_task9_like_healthy(self) -> None:
        profile = build_agent_budget_profile(_series_like_task9(), min_samples=3)
        deg = profile["degradation"]
        # +52% 左右，低于 50% 阈值 → healthy
        self.assertEqual(deg["verdict"], "healthy")
        self.assertLess(deg["per_turn_growth"], 0.50)

    def test_turn_blowup(self) -> None:
        rows = []
        for rid, turns in [(0, 40), (1, 42), (2, 45), (3, 50), (4, 55), (5, 60)]:
            for _ in range(10):
                rows.append(
                    _row(rid, turns=turns, per_turn_s=20.0, tokens_per_turn=300)
                )
        profile = build_agent_budget_profile(rows, min_samples=3)
        self.assertEqual(profile["degradation"]["verdict"], "turn_count_blowup")

    def test_insufficient(self) -> None:
        out = classify_degradation([])
        self.assertEqual(out["verdict"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
