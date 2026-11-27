#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Agent 时间预算退化画像（纯函数）。

按 rollout 聚合 attempt 级 ``phase_time`` / 轮数 / token，判定：
``per_turn_slowdown`` / ``turn_count_blowup`` / ``context_exhaustion`` / ``healthy``。

数据行由调用方构造（artifacts summary / .pt metadata）；本模块不读文件、不查 DB。
口径对齐 ``docs/PLAN_reward_uplift_specs.md`` §A。
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "normalize_attempt_row",
    "percentile",
    "window_size",
    "aggregate_rollout_point",
    "classify_degradation",
    "build_agent_budget_profile",
    "infer_budget_seconds",
]

_DEFAULT_BUDGET = 3600
_DEFAULT_CONTEXT_LIMIT = 131072
_VERDICTS = (
    "per_turn_slowdown",
    "turn_count_blowup",
    "context_exhaustion",
    "healthy",
    "insufficient_data",
)


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _phase_lookup(raw: Mapping[str, Any], *keys: str) -> Optional[float]:
    """兼容扁平 / 嵌套 / phase_times 前缀三种写法。"""
    # 1) 扁平 key
    for key in keys:
        if key in raw:
            hit = _as_float(raw.get(key))
            if hit is not None:
                return hit
        prefixed = f"phase_time/{key}" if not str(key).startswith("phase_time/") else key
        if prefixed in raw:
            hit = _as_float(raw.get(prefixed))
            if hit is not None:
                return hit

    # 2) nested phase_time dict
    nested = raw.get("phase_time")
    if isinstance(nested, Mapping):
        for key in keys:
            short = str(key).split("/")[-1]
            hit = _as_float(nested.get(short) or nested.get(key))
            if hit is not None:
                return hit

    # 3) phase_times dict（artifacts summary）
    phase_times = raw.get("phase_times")
    if isinstance(phase_times, Mapping):
        for key in keys:
            short = str(key).split("/")[-1]
            for candidate in (
                key,
                short,
                f"phase_time/{short}",
                f"phase_time/{key}",
            ):
                if candidate in phase_times:
                    hit = _as_float(phase_times.get(candidate))
                    if hit is not None:
                        return hit
    return None


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """线性插值分位数；空序列返回 None。"""
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    nums.sort()
    if len(nums) == 1:
        return nums[0]
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(nums) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return nums[lo]
    frac = pos - lo
    return nums[lo] * (1.0 - frac) + nums[hi] * frac


def window_size(n_rollouts: int) -> int:
    """首/末窗口宽度：clamp(round(0.2n), 3, 10)。"""
    n = max(0, int(n_rollouts or 0))
    if n <= 0:
        return 0
    return max(1, min(10, max(3, int(round(0.2 * n))))) if n >= 3 else n


def normalize_attempt_row(
    raw: Mapping[str, Any],
    *,
    default_context_limit: int = _DEFAULT_CONTEXT_LIMIT,
) -> Dict[str, Any]:
    """把任意来源的 attempt 行规整成画像输入行。"""
    meta = raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else {}
    merged: Dict[str, Any] = {}
    merged.update(meta)
    merged.update(raw)

    turns = (
        _as_int(merged.get("turns"))
        or _as_int(merged.get("agent_turns"))
        or _as_int(merged.get("attempt_completed_turns"))
    )
    agent_run = _as_float(merged.get("agent_run_s")) or _phase_lookup(
        merged, "agent_run", "phase_time/agent_run"
    )
    model_gen = _as_float(merged.get("model_generation_s")) or _phase_lookup(
        merged, "model_generation", "phase_time/model_generation"
    )
    attempt_total = _as_float(merged.get("attempt_total_s")) or _phase_lookup(
        merged, "attempt_total", "phase_time/attempt_total"
    )
    sandbox_boot_queue = _as_float(merged.get("sandbox_boot_queue_s")) or _phase_lookup(
        merged, "sandbox_boot_queue", "phase_time/sandbox_boot_queue"
    )
    sandbox_boot = _as_float(merged.get("sandbox_boot_s")) or _phase_lookup(
        merged, "sandbox_boot", "phase_time/sandbox_boot"
    )
    cli_install = _as_float(merged.get("cli_install_s")) or _phase_lookup(
        merged, "cli_install", "phase_time/cli_install"
    )
    evaluator_total = _as_float(merged.get("evaluator_total_s")) or _phase_lookup(
        merged, "evaluator_total", "phase_time/evaluator_total"
    )
    tool_execution = _as_float(merged.get("tool_execution_s")) or _phase_lookup(
        merged, "tool_execution", "phase_time/tool_execution"
    )
    group_sibling_wait = _as_float(merged.get("group_sibling_wait_s")) or _phase_lookup(
        merged, "group_sibling_wait", "phase_time/group_sibling_wait"
    )
    rollout_batch_wait = _as_float(merged.get("rollout_batch_wait_s")) or _phase_lookup(
        merged, "rollout_batch_wait", "phase_time/rollout_batch_wait"
    )

    gen_tokens = _as_float(
        merged.get("generated_response_tokens")
        or merged.get("attempt_generated_response_tokens")
    )
    peak_ctx = _as_float(
        merged.get("peak_context_tokens") or merged.get("attempt_peak_context_tokens")
    )
    ctx_limit = _as_int(merged.get("context_limit_tokens")) or int(default_context_limit)

    per_turn_seconds = None
    per_turn_source = "missing"
    if turns and turns > 0:
        if model_gen is not None:
            per_turn_seconds = model_gen / float(turns)
            per_turn_source = "model_generation"
        elif agent_run is not None:
            per_turn_seconds = agent_run / float(turns)
            per_turn_source = "agent_run"

    tokens_per_turn = None
    if turns and turns > 0 and gen_tokens is not None:
        tokens_per_turn = gen_tokens / float(turns)

    outcome = str(merged.get("agent_outcome") or "").strip()
    return {
        "rollout_id": _as_int(merged.get("rollout_id")),
        "group_index": _as_int(merged.get("group_index")),
        "attempt_id": _as_int(merged.get("attempt_id")),
        "instance_id": str(merged.get("instance_id") or ""),
        "status": str(merged.get("status") or ""),
        "agent_outcome": outcome,
        "category": str(merged.get("category") or ""),
        "reward": _as_float(merged.get("reward")),
        "attempt_total_s": attempt_total,
        "agent_run_s": agent_run,
        "model_generation_s": model_gen,
        "tool_execution_s": tool_execution,
        "cli_install_s": cli_install,
        "sandbox_boot_s": sandbox_boot,
        "evaluator_total_s": evaluator_total,
        "sandbox_boot_queue_s": sandbox_boot_queue,
        "group_sibling_wait_s": group_sibling_wait,
        "rollout_batch_wait_s": rollout_batch_wait,
        "turns": turns,
        "generated_response_tokens": gen_tokens,
        "peak_context_tokens": peak_ctx,
        "context_limit_tokens": ctx_limit,
        "per_turn_seconds": per_turn_seconds,
        "per_turn_source": per_turn_source,
        "tokens_per_turn": tokens_per_turn,
    }


def aggregate_rollout_point(
    rows: Sequence[Mapping[str, Any]],
    *,
    budget_seconds: float = _DEFAULT_BUDGET,
    context_limit_tokens: int = _DEFAULT_CONTEXT_LIMIT,
    min_samples: int = 3,
) -> Dict[str, Any]:
    """把同一 rollout 的规范行聚成一个 RolloutPoint。"""
    norms: List[Dict[str, Any]] = []
    for r in rows:
        # 已是规范行（含 agent_run_s）则直接用，避免二次 normalize 丢掉字段
        if "agent_run_s" in r or "per_turn_seconds" in r:
            norms.append(dict(r))
        else:
            norms.append(
                normalize_attempt_row(r, default_context_limit=context_limit_tokens)
            )
    rollout_ids = [n["rollout_id"] for n in norms if n["rollout_id"] is not None]
    rollout_id = rollout_ids[0] if rollout_ids else None

    n_attempts = len(norms)
    n_time_budget = sum(
        1
        for n in norms
        if "time_budget" in str(n.get("agent_outcome") or "").lower()
    )
    n_context = sum(
        1
        for n in norms
        if "context_exhaust" in str(n.get("agent_outcome") or "").lower()
        or str(n.get("agent_outcome") or "") == "slime_context_limit_reached"
    )

    agent_runs = [n["agent_run_s"] for n in norms if n["agent_run_s"] is not None]
    model_gens = [
        n["model_generation_s"] for n in norms if n["model_generation_s"] is not None
    ]
    turns_vals = [float(n["turns"]) for n in norms if n["turns"] and n["turns"] > 0]
    per_turns = [n["per_turn_seconds"] for n in norms if n["per_turn_seconds"] is not None]
    tok_turns = [n["tokens_per_turn"] for n in norms if n["tokens_per_turn"] is not None]
    peaks = [
        n["peak_context_tokens"] for n in norms if n["peak_context_tokens"] is not None
    ]
    attempt_totals = [
        n["attempt_total_s"] for n in norms if n["attempt_total_s"] is not None
    ]
    queues = [
        n["sandbox_boot_queue_s"]
        for n in norms
        if n["sandbox_boot_queue_s"] is not None
    ]

    overheads: List[float] = []
    for n in norms:
        if n["attempt_total_s"] is not None and n["agent_run_s"] is not None:
            overheads.append(n["attempt_total_s"] - n["agent_run_s"])

    n_valid = sum(
        1
        for n in norms
        if n["agent_run_s"] is not None and n["turns"] and n["turns"] > 0
    )
    agent_run_p50 = percentile(agent_runs, 0.5)
    attempt_total_p50 = percentile(attempt_totals, 0.5)
    sandbox_boot_queue_p50 = percentile(queues, 0.5)
    peak_p50 = percentile(peaks, 0.5)

    budget_hit = 0
    if budget_seconds and agent_runs:
        thr = float(budget_seconds) * 0.98
        budget_hit = sum(1 for v in agent_runs if v >= thr)

    return {
        "rollout_id": rollout_id,
        "n_attempts": n_attempts,
        "n_valid": n_valid,
        "n_time_budget": n_time_budget,
        "n_context_exhausted": n_context,
        "time_budget_ratio": (n_time_budget / n_attempts) if n_attempts else None,
        "context_exhausted_ratio": (n_context / n_attempts) if n_attempts else None,
        "agent_run_p50": agent_run_p50,
        "agent_run_p90": percentile(agent_runs, 0.9),
        "model_generation_p50": percentile(model_gens, 0.5),
        "per_turn_seconds_p50": percentile(per_turns, 0.5),
        "tokens_per_turn_p50": percentile(tok_turns, 0.5),
        "turns_p50": percentile(turns_vals, 0.5),
        "peak_context_p50": peak_p50,
        "peak_context_ratio_p50": (
            (peak_p50 / float(context_limit_tokens))
            if peak_p50 is not None and context_limit_tokens
            else None
        ),
        "overhead_seconds_p50": percentile(overheads, 0.5),
        "sandbox_boot_queue_p50": sandbox_boot_queue_p50,
        "queue_wait_ratio_p50": (
            (sandbox_boot_queue_p50 / attempt_total_p50)
            if sandbox_boot_queue_p50 is not None
            and attempt_total_p50
            and attempt_total_p50 > 0
            else None
        ),
        "budget_hit_ratio": (budget_hit / len(agent_runs)) if agent_runs else None,
        "low_confidence": n_valid < int(min_samples),
    }


def _growth(first: Optional[float], last: Optional[float]) -> Optional[float]:
    if first is None or last is None:
        return None
    if abs(float(first)) < 1e-12:
        return None
    return (float(last) - float(first)) / float(first)


def _window_median(series: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    vals = [
        _as_float(p.get(key))
        for p in series
        if _as_float(p.get(key)) is not None
    ]
    return percentile([v for v in vals if v is not None], 0.5)


def classify_degradation(
    series: Sequence[Mapping[str, Any]],
    *,
    min_rollouts: int = 3,
) -> Dict[str, Any]:
    """按 A.6 规则判定退化模式（命中即停）。"""
    usable = [
        p
        for p in series
        if p.get("rollout_id") is not None and not p.get("low_confidence")
    ]
    if len(usable) < int(min_rollouts):
        # 置信度不足时仍用全部点，但标 low
        usable = [p for p in series if p.get("rollout_id") is not None]
    if len(usable) < 2:
        return {
            "verdict": "insufficient_data",
            "verdict_reasons": ["有效 rollout 不足，无法判定退化"],
            "confidence": "low",
            "slowdown_attribution": "not_applicable",
            "secondary_signals": [],
        }

    usable = sorted(usable, key=lambda p: int(p["rollout_id"]))
    w = window_size(len(usable))
    first_win = usable[:w]
    last_win = usable[-w:]

    def g(key: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        first = _window_median(first_win, key)
        last = _window_median(last_win, key)
        return first, last, _growth(first, last)

    per_first, per_last, per_growth = g("per_turn_seconds_p50")
    tok_first, tok_last, tok_growth = g("tokens_per_turn_p50")
    turns_first, turns_last, turns_growth = g("turns_p50")
    peak_ratio_first, peak_ratio_last, _ = g("peak_context_ratio_p50")
    tb_first, tb_last, _ = g("time_budget_ratio")
    ctx_ratio_first, ctx_ratio_last, _ = g("context_exhausted_ratio")
    queue_first, queue_last, queue_growth = g("sandbox_boot_queue_p50")
    qratio_first, qratio_last, _ = g("queue_wait_ratio_p50")

    secondary: List[str] = []
    if turns_growth is not None and turns_growth > 0.05:
        secondary.append("turns_also_growing")
    if peak_ratio_last is not None and peak_ratio_last >= 0.85:
        secondary.append("peak_context_approaching_limit")
    if qratio_last is not None and qratio_last >= 0.10:
        secondary.append("queue_wait_high")
    if tb_last is not None and tb_last >= 0.05:
        secondary.append("time_budget_hits_rising")

    reasons: List[str] = []
    verdict = "healthy"
    confidence = "medium"

    # 优先级：per_turn（含撞墙）→ turn_blowup → context → healthy
    # 撞墙率阈值对齐 PLAN §3.0.7：避免轻度变慢但几乎不撞墙的任务被误判
    if (
        per_growth is not None
        and per_growth >= 0.50
        and tok_growth is not None
        and tok_growth >= 0.30
        and (turns_growth is None or turns_growth <= 0.10)
        and tb_last is not None
        and tb_last >= 0.05
    ):
        verdict = "per_turn_slowdown"
        reasons.append(
            f"单轮耗时 {per_first:.1f}s → {per_last:.1f}s（{per_growth * 100:.0f}%）"
        )
        reasons.append(
            f"单轮 token {tok_first:.0f} → {tok_last:.0f}（{tok_growth * 100:.0f}%）"
        )
        if turns_growth is not None:
            reasons.append(
                f"轮数 {turns_first:.0f} → {turns_last:.0f}（{turns_growth * 100:.0f}%）未上升"
            )
        reasons.append(f"末段撞墙率 {tb_last * 100:.1f}%")
        confidence = "high"
    elif turns_growth is not None and turns_growth >= 0.15:
        verdict = "turn_count_blowup"
        reasons.append(
            f"轮数 {turns_first:.0f} → {turns_last:.0f}（+{turns_growth * 100:.0f}%）"
        )
        confidence = "high"
    elif (
        peak_ratio_last is not None
        and peak_ratio_last >= 0.90
        and ctx_ratio_last is not None
        and ctx_ratio_last >= 0.10
    ):
        verdict = "context_exhaustion"
        reasons.append(
            f"末段 peak_context 达上限 {peak_ratio_last * 100:.0f}%，"
            f"context_exhausted 占比 {ctx_ratio_last * 100:.1f}%"
        )
        confidence = "high"
    else:
        reasons.append("未命中单轮变慢 / 轮数爆炸 / 上下文打满阈值")
        if per_growth is not None:
            reasons.append(f"单轮耗时增幅 {per_growth * 100:.0f}%（阈值 50%）")
        if tb_last is not None:
            reasons.append(f"末段撞墙率 {tb_last * 100:.1f}%")

    # 归因（仅 per_turn_slowdown）
    attribution = "not_applicable"
    if verdict == "per_turn_slowdown":
        c1 = tok_growth is not None and tok_growth >= 0.30
        c3 = queue_growth is not None and queue_growth >= 0.50
        if c1 and not c3:
            attribution = "model_output_growth"
        elif c3 and not c1:
            attribution = "engine_or_infra"
        elif c1 and c3:
            attribution = "mixed"
        else:
            attribution = "unresolved"

    return {
        "verdict": verdict,
        "verdict_reasons": reasons,
        "confidence": confidence,
        "slowdown_attribution": attribution,
        "secondary_signals": secondary,
        "first_rollout_id": usable[0]["rollout_id"],
        "last_rollout_id": usable[-1]["rollout_id"],
        "first_window_rollout_ids": [p["rollout_id"] for p in first_win],
        "last_window_rollout_ids": [p["rollout_id"] for p in last_win],
        "per_turn_seconds_first": per_first,
        "per_turn_seconds_last": per_last,
        "per_turn_growth": per_growth,
        "tokens_per_turn_first": tok_first,
        "tokens_per_turn_last": tok_last,
        "tokens_per_turn_growth": tok_growth,
        "turns_first": turns_first,
        "turns_last": turns_last,
        "turns_growth": turns_growth,
        "peak_context_ratio_first": peak_ratio_first,
        "peak_context_ratio_last": peak_ratio_last,
        "time_budget_ratio_first": tb_first,
        "time_budget_ratio_last": tb_last,
        "context_exhausted_ratio_first": ctx_ratio_first,
        "context_exhausted_ratio_last": ctx_ratio_last,
        "sandbox_boot_queue_first": queue_first,
        "sandbox_boot_queue_last": queue_last,
        "queue_wait_ratio_first": qratio_first,
        "queue_wait_ratio_last": qratio_last,
    }


def infer_budget_seconds(rows: Sequence[Mapping[str, Any]]) -> int:
    """从撞墙样本的 agent_run 推断预算（p1 向下取整到百位）；不足则 3600。"""
    hits: List[float] = []
    for raw in rows:
        n = normalize_attempt_row(raw)
        outcome = str(n.get("agent_outcome") or "").lower()
        if "time_budget" not in outcome:
            continue
        if n.get("agent_run_s") is not None:
            hits.append(float(n["agent_run_s"]))
    if len(hits) < 5:
        return _DEFAULT_BUDGET
    p1 = percentile(hits, 0.01)
    if p1 is None:
        return _DEFAULT_BUDGET
    rounded = int(math.floor(p1 / 100.0) * 100)
    return rounded if rounded >= 600 else _DEFAULT_BUDGET


def build_agent_budget_profile(
    rows: Sequence[Mapping[str, Any]],
    *,
    budget_seconds: Optional[int] = None,
    context_limit_tokens: int = _DEFAULT_CONTEXT_LIMIT,
    min_samples: int = 3,
) -> Dict[str, Any]:
    """从 attempt 行构建完整预算画像。"""
    norms = [
        normalize_attempt_row(r, default_context_limit=context_limit_tokens) for r in rows
    ]
    norms = [n for n in norms if n.get("rollout_id") is not None]
    budget = int(budget_seconds) if budget_seconds else infer_budget_seconds(norms)

    by_rollout: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for n in norms:
        by_rollout[int(n["rollout_id"])].append(n)

    series: List[Dict[str, Any]] = []
    for rid in sorted(by_rollout.keys()):
        series.append(
            aggregate_rollout_point(
                by_rollout[rid],
                budget_seconds=budget,
                context_limit_tokens=context_limit_tokens,
                min_samples=min_samples,
            )
        )

    degradation = classify_degradation(series)
    sources = {n.get("per_turn_source") for n in norms if n.get("per_turn_source")}
    sources.discard("missing")
    if not sources:
        per_turn_source = "missing"
    elif sources == {"model_generation"}:
        per_turn_source = "model_generation"
    elif sources == {"agent_run"}:
        per_turn_source = "agent_run"
    else:
        per_turn_source = "mixed"

    missing_phase = [
        p["rollout_id"]
        for p in series
        if p.get("agent_run_p50") is None
    ]
    low_conf = [p["rollout_id"] for p in series if p.get("low_confidence")]

    note = (
        "预算墙钟计在 agent_run 上；单轮耗时优先 model_generation/turns。"
        "判定：单轮变慢（+50% 且 token+30% 且轮数未暴涨）优先于轮数爆炸 / 上下文打满。"
    )
    return {
        "budget_seconds": budget,
        "budget_clock": "agent_run",
        "context_limit_tokens": int(context_limit_tokens),
        "n_attempts": len(norms),
        "n_rollouts": len(series),
        "n_rollouts_with_data": sum(1 for p in series if not p.get("low_confidence")),
        "series": series,
        "degradation": degradation,
        "data_quality": {
            "min_samples": int(min_samples),
            "low_confidence_rollouts": low_conf,
            "missing_phase_time_rollouts": missing_phase,
            "per_turn_source": per_turn_source,
            "notes": [],
        },
        "note": note,
        "available": len(series) > 0 and any(
            p.get("per_turn_seconds_p50") is not None or p.get("time_budget_ratio")
            for p in series
        ),
    }
