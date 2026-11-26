#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""任务级 Agentic 问题分析：把失败聚类 / 行为卡片 / 组统计拼成问题卡片。

纯函数 ``assemble_problem_cards`` 可单测；``build_task_problem_report`` 负责读库拼装。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from data.agentic_problem_catalog import AGENTIC_PROBLEMS, list_agentic_problems

__all__ = [
    "assemble_problem_cards",
    "build_task_problem_report",
    "problem_status",
]

_RATE_EPS = 1e-9


def problem_status(
    ratio: Optional[float],
    *,
    warn_ratio: float,
    bad_ratio: float,
) -> str:
    """按占比阈值给出 ok / warn / bad / unknown。"""
    if ratio is None:
        return "unknown"
    value = float(ratio)
    if value >= float(bad_ratio) - _RATE_EPS:
        return "bad"
    if value >= float(warn_ratio) - _RATE_EPS:
        return "warn"
    return "ok"


def _ratio(count: int, denom: Optional[int]) -> Optional[float]:
    if denom is None or int(denom) <= 0:
        return None
    return float(count) / float(denom)


def _outcome_count(agent_outcomes: Sequence[Dict[str, Any]], key: str) -> int:
    for row in agent_outcomes:
        if str(row.get("agent_outcome") or "") == key:
            return int(row.get("count") or 0)
    return 0


def _outcome_count_any(
    agent_outcomes: Sequence[Dict[str, Any]], keys: Sequence[str]
) -> int:
    total = 0
    for key in keys:
        total += _outcome_count(agent_outcomes, key)
    return total


def _behavior_count(behavior_cards: Sequence[Dict[str, Any]], key: str) -> Tuple[int, Optional[int]]:
    """返回 (命中数, 失败分母)；找不到卡片时 (0, None)。"""
    for card in behavior_cards:
        if str(card.get("key") or "") != key:
            continue
        n = int(card.get("n_attempts") or 0)
        n_failed = card.get("n_failed")
        denom = int(n_failed) if isinstance(n_failed, (int, float)) else None
        return n, denom
    return 0, None


def _diagnosis_count(
    clusters: Sequence[Dict[str, Any]], substrings: Sequence[str]
) -> int:
    needles = [str(s).lower() for s in substrings if s]
    if not needles:
        return 0
    total = 0
    for row in clusters:
        reason = str(row.get("reason") or "").lower()
        if any(n in reason for n in needles):
            total += int(row.get("count") or 0)
    return total


def _category_count(totals: Dict[str, Any], key: str) -> int:
    mapping = {
        "abort": "n_abort",
        "timeout": "n_timeout",
        "fail": "n_fail",
        "pass": "n_pass",
        "other": "n_other",
        "attempts": "n_attempts",
    }
    field = mapping.get(key, key)
    return int(totals.get(field) or 0)


def _pick_examples(
    attempts: Sequence[Dict[str, Any]],
    *,
    agent_outcomes: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
    reason_substrings: Optional[Sequence[str]] = None,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    agents = {str(x) for x in (agent_outcomes or []) if x}
    cats = {str(x) for x in (categories or []) if x}
    needles = [str(s).lower() for s in (reason_substrings or []) if s]
    out: List[Dict[str, Any]] = []
    for item in attempts:
        if agents and str(item.get("agent_outcome") or "") not in agents:
            continue
        if cats and str(item.get("category") or "") not in cats:
            continue
        if needles:
            reason = str(item.get("reason") or "").lower()
            if not any(n in reason for n in needles):
                continue
        pk = item.get("attempt_pk")
        out.append(
            {
                "attempt_pk": pk,
                "rollout_id": item.get("rollout_id"),
                "group_index": item.get("group_index"),
                "attempt_id": item.get("attempt_id"),
                "instance_id": item.get("instance_id") or "",
                "agent_outcome": item.get("agent_outcome") or "",
                "reason": item.get("reason") or "",
                "category": item.get("category") or "",
                "detail_url": (
                    f"/api/failures/attempts/{int(pk)}" if pk is not None else None
                ),
            }
        )
        if len(out) >= limit:
            break
    return out


def _failure_list_url(task_id: int, query: Dict[str, Any]) -> Optional[str]:
    if not query:
        return f"/api/tasks/{int(task_id)}/failures"
    parts = [f"{k}={v}" for k, v in query.items() if v is not None and str(v) != ""]
    base = f"/api/tasks/{int(task_id)}/failures"
    return f"{base}?{'&'.join(parts)}" if parts else base


def assemble_problem_cards(
    *,
    catalog: Optional[Sequence[Dict[str, Any]]] = None,
    n_attempts: int,
    totals: Optional[Dict[str, Any]] = None,
    agent_outcomes: Optional[Sequence[Dict[str, Any]]] = None,
    clusters: Optional[Sequence[Dict[str, Any]]] = None,
    behavior: Optional[Dict[str, Any]] = None,
    gradient_latest: Optional[Dict[str, Any]] = None,
    pool_health: Optional[Dict[str, Any]] = None,
    attempts: Optional[Sequence[Dict[str, Any]]] = None,
    task_id: Optional[int] = None,
    tb_trends: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """把各类计数拼成问题卡片列表（按严重度排序）。

    不读库；缺字段时 count=0、ratio=None 或 0，status=unknown/ok。
    """
    defs = list(catalog) if catalog is not None else list(AGENTIC_PROBLEMS)
    totals = dict(totals or {})
    agent_outcomes = list(agent_outcomes or [])
    clusters = list(clusters or [])
    behavior = dict(behavior or {})
    behavior_cards = list(behavior.get("cards") or [])
    n_failed_behavior = behavior.get("n_failed")
    if isinstance(n_failed_behavior, (int, float)):
        n_failed = int(n_failed_behavior)
    else:
        n_failed = int(totals.get("n_failure") or 0) or (
            int(totals.get("n_fail") or 0)
            + int(totals.get("n_timeout") or 0)
            + int(totals.get("n_abort") or 0)
            + int(totals.get("n_other") or 0)
        )
    gradient_latest = dict(gradient_latest or {})
    pool_health = dict(pool_health or {})
    attempts = list(attempts or [])
    tb_trends = dict(tb_trends or {})
    tid = int(task_id) if task_id is not None else None

    cards: List[Dict[str, Any]] = []
    for spec in defs:
        source = str(spec.get("source") or "")
        match = dict(spec.get("match") or {})
        count = 0
        denom: Optional[int] = int(n_attempts) if n_attempts else None
        denom_label = "attempts"
        examples: List[Dict[str, Any]] = []
        note = ""

        if source == "agent_outcome":
            key = str(match.get("agent_outcome") or "")
            count = _outcome_count(agent_outcomes, key)
            denom_label = "attempts"
            examples = _pick_examples(attempts, agent_outcomes=[key])
        elif source == "agent_outcome_any":
            keys = [str(x) for x in (match.get("agent_outcomes") or ())]
            count = _outcome_count_any(agent_outcomes, keys)
            denom_label = "attempts"
            examples = _pick_examples(attempts, agent_outcomes=keys)
        elif source == "behavior":
            bkey = str(match.get("behavior_key") or "")
            count, b_failed = _behavior_count(behavior_cards, bkey)
            denom = b_failed if b_failed is not None else n_failed
            denom_label = "failed"
            note = str(behavior.get("turn_coverage_note") or "")
        elif source == "abort_or_sdk":
            sdk = _outcome_count(agent_outcomes, "sdk_error")
            abort = _category_count(totals, "abort")
            # abort 已含部分 sdk；取 max 避免双计主导，同时保留两者信息
            count = max(sdk, abort)
            if sdk and abort and sdk != abort:
                note = f"agent sdk_error={sdk}，category abort={abort}（取较大值）。"
            elif abort:
                note = f"category abort={abort}" + (f"，其中 sdk_error={sdk}" if sdk else "")
            denom_label = "attempts"
            examples = _pick_examples(attempts, agent_outcomes=["sdk_error"])
            if not examples:
                examples = _pick_examples(attempts, categories=["abort"])
        elif source == "diagnosis":
            subs = [str(x) for x in (match.get("reason_substrings") or ())]
            count = _diagnosis_count(clusters, subs)
            denom_label = "attempts"
            examples = _pick_examples(attempts, reason_substrings=subs)
        elif source == "gradient":
            n_all_pass = int(gradient_latest.get("n_all_pass") or 0)
            n_all_fail = int(gradient_latest.get("n_all_fail") or 0)
            n_groups = int(gradient_latest.get("n_groups") or 0)
            count = n_all_pass + n_all_fail
            denom = n_groups if n_groups > 0 else None
            denom_label = "groups"
            note = (
                f"最近一步全过组 {n_all_pass}、全挂组 {n_all_fail}、"
                f"总组数 {n_groups}。"
                if n_groups
                else "尚无组统计。"
            )
        elif source == "pool":
            pool_key = str(match.get("pool_key") or "")
            ratio_direct = pool_health.get(pool_key)
            n_problems = int(pool_health.get("n_problems") or 0)
            if pool_key == "saturated_pool_ratio":
                count = int(pool_health.get("n_saturated") or 0)
            elif pool_key == "hard_pool_ratio":
                count = int(pool_health.get("n_hard") or 0)
            else:
                count = 0
            denom = n_problems if n_problems > 0 else None
            denom_label = "problems"
            # 直接用池子比例，避免 count/n 与浮点阈值漂移
            ratio_override = (
                float(ratio_direct) if isinstance(ratio_direct, (int, float)) else None
            )
            ratio = ratio_override if ratio_override is not None else _ratio(count, denom)
            status = problem_status(
                ratio,
                warn_ratio=float(spec.get("warn_ratio") or 0.35),
                bad_ratio=float(spec.get("bad_ratio") or 0.6),
            )
            card = _card_dict(
                spec,
                count=count,
                ratio=ratio,
                status=status,
                denom=denom,
                denom_label=denom_label,
                examples=examples,
                note=note,
                task_id=tid,
                tb=tb_trends.get(str(spec.get("tb_key") or "")),
            )
            cards.append(card)
            continue
        else:
            note = f"未知 source={source}"
            denom = None

        if source != "pool":
            if denom_label == "failed":
                denom = denom if denom is not None else n_failed
            elif denom_label == "groups":
                pass
            else:
                denom = int(n_attempts) if n_attempts else None
            ratio = _ratio(count, denom)
            status = problem_status(
                ratio,
                warn_ratio=float(spec.get("warn_ratio") or 0.05),
                bad_ratio=float(spec.get("bad_ratio") or 0.1),
            )
            cards.append(
                _card_dict(
                    spec,
                    count=count,
                    ratio=ratio,
                    status=status,
                    denom=denom,
                    denom_label=denom_label,
                    examples=examples,
                    note=note,
                    task_id=tid,
                    tb=tb_trends.get(str(spec.get("tb_key") or "")),
                )
            )

    order = {"bad": 0, "warn": 1, "unknown": 2, "ok": 3}
    cards.sort(
        key=lambda c: (
            order.get(str(c.get("status") or ""), 9),
            -(float(c["ratio"]) if isinstance(c.get("ratio"), (int, float)) else -1.0),
            str(c.get("id") or ""),
        )
    )
    return cards


def _card_dict(
    spec: Dict[str, Any],
    *,
    count: int,
    ratio: Optional[float],
    status: str,
    denom: Optional[int],
    denom_label: str,
    examples: List[Dict[str, Any]],
    note: str,
    task_id: Optional[int],
    tb: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    query = dict(spec.get("failure_query") or {})
    return {
        "id": spec["id"],
        "title": spec["title"],
        "why_blocks_reward": spec.get("why_blocks_reward") or "",
        "grpo_meaning": spec.get("grpo_meaning") or "",
        "reward_action": spec.get("reward_action") or "",
        "source": spec.get("source") or "",
        "count": int(count),
        "denominator": denom,
        "denominator_label": denom_label,
        "ratio": ratio,
        "ratio_pct": round(100.0 * ratio, 1) if isinstance(ratio, (int, float)) else None,
        "status": status,
        "examples": examples,
        "note": note,
        "failure_url": (
            _failure_list_url(task_id, query) if task_id is not None else None
        ),
        "tb_trend": tb,
    }


def _tb_first_last(
    event_dir: Optional[Path], tags: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """读取指定 TB tag 的首尾值；缺 tag / 读失败则跳过。"""
    if event_dir is None or not tags:
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception:  # noqa: BLE001
        return {}
    try:
        acc = EventAccumulator(str(event_dir), size_guidance={"scalars": 10000})
        acc.Reload()
    except Exception as exc:  # noqa: BLE001
        logger.info("问题分析跳过 TB 趋势 event_dir={} err={}", event_dir, exc)
        return {}
    available = set((acc.Tags() or {}).get("scalars") or [])
    out: Dict[str, Dict[str, Any]] = {}
    # 逻辑名 → 优先 tag
    prefer = {
        "max_fake_responses": (
            "rollout/outcome/max_fake_responses",
            "max_fake_responses",
        ),
        "context_exhausted": (
            "rollout/outcome/context_exhausted",
            "rollout/context_exhausted",
        ),
    }
    for logical in tags:
        candidates = prefer.get(logical, (logical,))
        hit = next((t for t in candidates if t in available), None)
        if not hit:
            continue
        try:
            events = acc.Scalars(hit)
        except Exception:  # noqa: BLE001
            continue
        if not events:
            continue
        first, last = events[0], events[-1]
        try:
            fv, lv = float(first.value), float(last.value)
        except (TypeError, ValueError):
            continue
        out[logical] = {
            "tag": hit,
            "first_value": fv,
            "last_value": lv,
            "delta": lv - fv,
            "first_step": int(getattr(first, "step", 0) or 0),
            "last_step": int(getattr(last, "step", 0) or 0),
        }
    return out


def build_task_problem_report(task_id: int) -> Dict[str, Any]:
    """组装任务问题分析页 / JSON：目录卡片 + 任务摘要。"""
    from data.rollout_insight_exception import RolloutInsightException
    from db.db_attempt_service import DBAttemptService
    from db.db_segment_turn_service import DBSegmentTurnService
    from db.db_task_service import DBTaskService
    from service.failure_analysis_service import get_task_failure_overview
    from service.reward_diagnosis_service import (
        summarize_behavior_cards,
        summarize_gradient_groups,
        summarize_pool_health,
        summarize_reward_split,
    )
    from utils.tb_run_paths import discover_tb_event_dir, resolve_run_dir

    task = DBTaskService.get_by_id(int(task_id))
    if task is None:
        raise RolloutInsightException(
            RolloutInsightException.TASK_NOT_FOUND, f"task_id={task_id} 不存在"
        )

    overview = get_task_failure_overview(int(task_id))
    rows = DBAttemptService.list_outcome_rows_by_task(int(task_id))
    coverage = DBSegmentTurnService.segment_turn_coverage(int(task_id))
    n_with_turns = int(coverage.get("n_segments_with_turns") or 0)
    turn_flags: List[Dict[str, Any]] = []
    if n_with_turns > 0:
        turn_flags = DBSegmentTurnService.list_assistant_turn_flags(int(task_id))

    reward_split = summarize_reward_split(rows)
    gradient = summarize_gradient_groups(rows)
    pool = summarize_pool_health(rows)
    behavior = summarize_behavior_cards(
        rows,
        turn_flags,
        reward_split_series=reward_split.get("series"),
    )

    event_dir = None
    rollout_root = str(getattr(task, "rollout_root", "") or "").strip()
    if rollout_root:
        run_dir = resolve_run_dir(rollout_root)
        event_dir = discover_tb_event_dir(run_dir) if run_dir is not None else None
    tb_trends = _tb_first_last(
        event_dir, ("max_fake_responses", "context_exhausted")
    )

    totals = dict(overview.get("totals") or {})
    n_attempts = int(totals.get("n_attempts") or len(rows))
    cards = assemble_problem_cards(
        n_attempts=n_attempts,
        totals=totals,
        agent_outcomes=overview.get("agent_outcomes") or [],
        clusters=overview.get("clusters") or [],
        behavior=behavior,
        gradient_latest=gradient.get("latest") or {},
        pool_health=pool,
        attempts=overview.get("attempts") or [],
        task_id=int(task_id),
        tb_trends=tb_trends,
    )

    n_bad = sum(1 for c in cards if c.get("status") == "bad")
    n_warn = sum(1 for c in cards if c.get("status") == "warn")
    logger.info(
        "问题分析 task_id={} attempts={} cards={} bad={} warn={}",
        task_id,
        n_attempts,
        len(cards),
        n_bad,
        n_warn,
    )
    return {
        "task": {
            "id": int(task.primary_id),
            "task_name": task.task_name,
            "status": getattr(task, "status", None),
            "rollout_root": rollout_root or None,
        },
        "n_attempts": n_attempts,
        "totals": totals,
        "source": overview.get("source"),
        "catalog": list_agentic_problems(),
        "problems": cards,
        "summary": {
            "n_problems": len(cards),
            "n_bad": n_bad,
            "n_warn": n_warn,
            "n_ok": sum(1 for c in cards if c.get("status") == "ok"),
        },
        "tb_trends": tb_trends,
        "page_note": (
            "按占比严重度排序。每条问题给出奖励设计动作与失败分析样例；"
            "假响应等同时对照训练曲线 TB tag（若有）。"
        ),
    }
