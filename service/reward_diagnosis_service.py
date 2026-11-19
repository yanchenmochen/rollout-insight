#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""任务奖励诊断：有效梯度组、奖励分解、行为失败卡片。

数据来自 attempt 结局行与已落库的 assistant turn 标记，不新扫 .pt 正文。
组判定与 ``DBAttemptService.summarize_reward_metrics_by_rollout`` 一致：
去掉 abort/timeout 后同时有过有挂 → 混合（可学习）组。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

__all__ = [
    "classify_group_kind",
    "summarize_gradient_groups",
    "summarize_reward_split",
    "aggregate_attempt_turn_flags",
    "classify_behavior_primary",
    "summarize_behavior_cards",
    "build_reward_diagnosis",
]

_RATE_EPS = 1e-9
_DIP_DROP = 0.15
_SAMPLE_LIMIT = 3
_EARLY_MAX_TOOL_CALLS = 2
_EARLY_MAX_DIFF_LINES = 5
_EARLY_MAX_AGENT_TURNS = 2

_GROUP_KINDS = ("mixed", "all_pass", "all_fail", "infra_only")

_BEHAVIOR_META: Dict[str, Dict[str, str]] = {
    "abort": {
        "title": "环境 / 基建中止",
        "phenomenon": "attempt 被标为 abort（沙箱、driver、调度等），奖励不可测。",
        "lever": "先修环境与 harness；不要把 abort 当策略负样本。",
    },
    "timeout": {
        "title": "超时 / 上下文打满",
        "phenomenon": "status 为 truncated/timeout，策略可能尚未收尾。",
        "lever": "调 max_turns / 上下文长度，或加长度惩罚；勿把 timeout 当普通 fail。",
    },
    "no_tool_call": {
        "title": "无 tool_call",
        "phenomenon": "assistant 回复里出现过没有工具调用的轮次。",
        "lever": "格式惩罚或丢弃无 tool_call 样本，强化工具调用格式。",
    },
    "last_not_finish": {
        "title": "最后不是 finish",
        "phenomenon": "该 attempt 最后一条 assistant 回复的最后一个工具不是 finish。",
        "lever": "强化完成条件：最后一步应调用 finish。",
    },
    "early_finish": {
        "title": "过早 finish",
        "phenomenon": "已调用 finish，但工具步很少且 diff 很小。",
        "lever": "收紧完成条件，或对过短轨迹降权 / 惩罚。",
    },
}

_BEHAVIOR_ORDER = (
    "abort",
    "timeout",
    "no_tool_call",
    "last_not_finish",
    "early_finish",
)


def classify_group_kind(categories: Sequence[str]) -> str:
    """按组内 attempt 的 category 判定：mixed / all_pass / all_fail / infra_only。"""
    has_one = False
    has_zero = False
    measurable = 0
    for cat in categories:
        token = str(cat or "")
        if token in {"abort", "timeout"}:
            continue
        measurable += 1
        if token == "pass":
            has_one = True
        elif token == "fail":
            has_zero = True
    if measurable <= 0:
        return "infra_only"
    if has_one and has_zero:
        return "mixed"
    if has_one:
        return "all_pass"
    if has_zero:
        return "all_fail"
    return "all_fail"


def _rollout_ids(rows: Sequence[Dict[str, Any]]) -> List[int]:
    seen = set()
    out: List[int] = []
    for row in rows:
        rid = row.get("rollout_id")
        if rid is None:
            continue
        rid_i = int(rid)
        if rid_i in seen:
            continue
        seen.add(rid_i)
        out.append(rid_i)
    return out


def _groups_by_rollout(
    rows: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[int, List[str]]]:
    by_rollout: Dict[int, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("rollout_id") is None or row.get("group_index") is None:
            continue
        rid = int(row["rollout_id"])
        gi = int(row["group_index"])
        by_rollout[rid][gi].append(str(row.get("category") or "other"))
    return by_rollout


def summarize_gradient_groups(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """按 rollout 统计四类组数量与可学习组占比，并标出骤降点。"""
    by_rollout = _groups_by_rollout(rows)
    series: List[Dict[str, Any]] = []
    for rid in sorted(by_rollout.keys()):
        counts = {kind: 0 for kind in _GROUP_KINDS}
        for cats in by_rollout[rid].values():
            kind = classify_group_kind(cats)
            counts[kind] = int(counts.get(kind, 0)) + 1
        n_groups = sum(counts.values())
        n_mixed = int(counts["mixed"])
        learnable_ratio = (n_mixed / n_groups) if n_groups else None
        series.append(
            {
                "rollout_id": rid,
                "n_groups": n_groups,
                "n_mixed": n_mixed,
                "n_all_pass": int(counts["all_pass"]),
                "n_all_fail": int(counts["all_fail"]),
                "n_infra_only": int(counts["infra_only"]),
                "learnable_ratio": learnable_ratio,
            }
        )

    dips: List[Dict[str, Any]] = []
    for idx in range(1, len(series)):
        prev = series[idx - 1].get("learnable_ratio")
        cur = series[idx].get("learnable_ratio")
        if prev is None or cur is None:
            continue
        drop = float(prev) - float(cur)
        if drop >= _DIP_DROP - _RATE_EPS:
            dips.append(
                {
                    "rollout_id": series[idx]["rollout_id"],
                    "prev_rollout_id": series[idx - 1]["rollout_id"],
                    "prev_ratio": float(prev),
                    "ratio": float(cur),
                    "drop": drop,
                }
            )
    dips.sort(key=lambda item: (-float(item["drop"]), int(item["rollout_id"])))

    latest = series[-1] if series else None
    return {
        "n_rollouts": len(series),
        "series": series,
        "dips": dips,
        "latest": latest,
        "note": (
            "混合组 = 去掉 abort/timeout 后既有 reward≈1 又有 reward≈0；"
            "全过 / 全挂优势为 0；只剩基建表示可测样本被清光。"
        ),
    }


def summarize_reward_split(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """同一 rollout 轴：原始平均奖励、可测通过率、超时占比、abort 占比。"""
    buckets: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if row.get("rollout_id") is None:
            continue
        rid = int(row["rollout_id"])
        bucket = buckets.setdefault(
            rid,
            {
                "n_attempts": 0,
                "reward_sum": 0.0,
                "reward_n": 0,
                "n_pass": 0,
                "n_fail": 0,
                "n_timeout": 0,
                "n_abort": 0,
                "n_other": 0,
            },
        )
        bucket["n_attempts"] += 1
        cat = str(row.get("category") or "other")
        if cat == "pass":
            bucket["n_pass"] += 1
        elif cat == "fail":
            bucket["n_fail"] += 1
        elif cat == "timeout":
            bucket["n_timeout"] += 1
        elif cat == "abort":
            bucket["n_abort"] += 1
        else:
            bucket["n_other"] += 1
        reward = row.get("reward")
        if isinstance(reward, (int, float)):
            bucket["reward_sum"] += float(reward)
            bucket["reward_n"] += 1

    series: List[Dict[str, Any]] = []
    for rid in sorted(buckets.keys()):
        bucket = buckets[rid]
        n = int(bucket["n_attempts"])
        measurable = int(bucket["n_pass"]) + int(bucket["n_fail"]) + int(bucket["n_other"])
        raw_mean = (
            float(bucket["reward_sum"]) / int(bucket["reward_n"])
            if bucket["reward_n"]
            else None
        )
        measurable_pass_rate = (
            int(bucket["n_pass"]) / measurable if measurable > 0 else None
        )
        series.append(
            {
                "rollout_id": rid,
                "n_attempts": n,
                "raw_mean_reward": raw_mean,
                "measurable_pass_rate": measurable_pass_rate,
                "timeout_ratio": (int(bucket["n_timeout"]) / n) if n else None,
                "abort_ratio": (int(bucket["n_abort"]) / n) if n else None,
                "n_pass": int(bucket["n_pass"]),
                "n_fail": int(bucket["n_fail"]),
                "n_timeout": int(bucket["n_timeout"]),
                "n_abort": int(bucket["n_abort"]),
                "n_other": int(bucket["n_other"]),
                "n_measurable": measurable,
            }
        )
    return {
        "n_rollouts": len(series),
        "series": series,
        "note": (
            "原始平均奖励 = 有数值 reward 的 attempt 均值；"
            "可测通过率分母不含 abort/timeout；"
            "超时 / abort 占比分母为该步全部 attempt。"
        ),
    }


def aggregate_attempt_turn_flags(
    flags: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """把 assistant turn 标记收成 per-attempt 摘要。"""
    by_attempt: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in flags:
        pk = row.get("attempt_pk")
        if pk is None:
            continue
        by_attempt[int(pk)].append(row)

    out: Dict[int, Dict[str, Any]] = {}
    for pk, turns in by_attempt.items():
        turns_sorted = sorted(
            turns,
            key=lambda item: (
                int(item.get("segment_index") or 0),
                int(item.get("turn_index") or 0),
            ),
        )
        any_without = False
        for turn in turns_sorted:
            names = [str(x) for x in (turn.get("tool_call_names") or []) if x]
            has_call = bool(turn.get("has_tool_call")) or bool(names)
            if not has_call:
                any_without = True
        last_names = [
            str(x) for x in (turns_sorted[-1].get("tool_call_names") or []) if x
        ]
        last_finish = bool(last_names) and last_names[-1].lower() == "finish"
        out[pk] = {
            "n_assistant": len(turns_sorted),
            "any_without_tool_call": any_without,
            "last_is_finish": last_finish,
            "rollout_id": turns_sorted[0].get("rollout_id"),
        }
    return out


def _is_early_finish(row: Dict[str, Any], turn_summary: Dict[str, Any]) -> bool:
    if not turn_summary.get("last_is_finish"):
        return False
    try:
        tool_calls = int(row.get("tool_calls") or 0)
    except (TypeError, ValueError):
        tool_calls = 0
    try:
        agent_turns = int(row.get("agent_turns") or 0)
    except (TypeError, ValueError):
        agent_turns = 0
    try:
        diff_lines = int(row.get("diff_line_count") or 0)
    except (TypeError, ValueError):
        diff_lines = 0
    short_tools = tool_calls <= _EARLY_MAX_TOOL_CALLS or agent_turns <= _EARLY_MAX_AGENT_TURNS
    return short_tools and diff_lines <= _EARLY_MAX_DIFF_LINES


def classify_behavior_primary(
    row: Dict[str, Any],
    turn_summary: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """给非通过 attempt 打一个主行为类；通过样本返回 None。"""
    cat = str(row.get("category") or "")
    if cat == "pass":
        return None
    if cat == "abort":
        return "abort"
    if cat == "timeout":
        return "timeout"
    summary = turn_summary or {}
    if summary:
        if summary.get("any_without_tool_call"):
            return "no_tool_call"
        if not summary.get("last_is_finish"):
            return "last_not_finish"
        if _is_early_finish(row, summary):
            return "early_finish"
    return None


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _alignment_label(
    pass_series: Sequence[Dict[str, Any]],
    behavior_by_rollout: Dict[int, Dict[str, int]],
) -> Optional[str]:
    """行为占比上升且可测通过率下降 → explains_drop。"""
    if len(pass_series) < 2:
        return None
    paired: List[Tuple[float, float]] = []
    for point in pass_series:
        rid = int(point["rollout_id"])
        pass_rate = point.get("measurable_pass_rate")
        counts = behavior_by_rollout.get(rid) or {}
        n_failed = int(counts.get("n_failed") or 0)
        n_hit = int(counts.get("n_hit") or 0)
        if pass_rate is None or n_failed <= 0:
            continue
        paired.append((float(pass_rate), n_hit / n_failed))
    if len(paired) < 2:
        return None
    mid = len(paired) // 2
    first_pass = _mean([p[0] for p in paired[:mid]])
    last_pass = _mean([p[0] for p in paired[mid:]])
    first_beh = _mean([p[1] for p in paired[:mid]])
    last_beh = _mean([p[1] for p in paired[mid:]])
    if None in (first_pass, last_pass, first_beh, last_beh):
        return None
    pass_delta = float(last_pass) - float(first_pass)
    beh_delta = float(last_beh) - float(first_beh)
    if pass_delta < -_RATE_EPS and beh_delta > _RATE_EPS:
        return "explains_drop"
    if pass_delta > _RATE_EPS and beh_delta < -_RATE_EPS:
        return "improving_together"
    return "unclear"


def summarize_behavior_cards(
    rows: Sequence[Dict[str, Any]],
    turn_flags: Sequence[Dict[str, Any]],
    *,
    reward_split_series: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """按主行为类汇总卡片：占比、均值奖励、样例链接、与通过率是否同向。"""
    turn_by_attempt = aggregate_attempt_turn_flags(turn_flags)
    n_failed = 0
    n_with_turns = 0
    hits: Dict[str, List[Dict[str, Any]]] = {key: [] for key in _BEHAVIOR_ORDER}
    by_rollout_beh: Dict[str, Dict[int, Dict[str, int]]] = {
        key: defaultdict(lambda: {"n_hit": 0, "n_failed": 0}) for key in _BEHAVIOR_ORDER
    }
    failed_by_rollout: Dict[int, int] = defaultdict(int)

    for row in rows:
        cat = str(row.get("category") or "")
        if cat == "pass":
            continue
        n_failed += 1
        rid = int(row["rollout_id"]) if row.get("rollout_id") is not None else None
        if rid is not None:
            failed_by_rollout[rid] += 1
        pk = row.get("attempt_pk")
        summary = turn_by_attempt.get(int(pk)) if pk is not None else None
        if summary:
            n_with_turns += 1
        key = classify_behavior_primary(row, summary)
        if key is None:
            continue
        hits[key].append(row)
        if rid is not None:
            by_rollout_beh[key][rid]["n_hit"] += 1

    for key in _BEHAVIOR_ORDER:
        for rid, n in failed_by_rollout.items():
            by_rollout_beh[key][rid]["n_failed"] = int(n)

    pass_series = list(reward_split_series or [])
    cards: List[Dict[str, Any]] = []
    for key in _BEHAVIOR_ORDER:
        items = hits[key]
        meta = _BEHAVIOR_META[key]
        rewards: List[float] = []
        for item in items:
            reward = item.get("reward")
            if isinstance(reward, (int, float)):
                rewards.append(float(reward))
        examples = []
        for item in items[:_SAMPLE_LIMIT]:
            pk = item.get("attempt_pk")
            examples.append(
                {
                    "attempt_pk": pk,
                    "rollout_id": item.get("rollout_id"),
                    "group_index": item.get("group_index"),
                    "attempt_id": item.get("attempt_id"),
                    "instance_id": item.get("instance_id") or "",
                    "reward": item.get("reward"),
                    "detail_url": (
                        f"/api/failures/attempts/{int(pk)}" if pk is not None else None
                    ),
                }
            )
        cards.append(
            {
                "key": key,
                "title": meta["title"],
                "phenomenon": meta["phenomenon"],
                "lever": meta["lever"],
                "n_attempts": len(items),
                "n_failed": n_failed,
                "ratio_of_failed": (len(items) / n_failed) if n_failed else None,
                "mean_reward": _mean(rewards),
                "pass_rate_relation": _alignment_label(
                    pass_series, by_rollout_beh[key]
                ),
                "examples": examples,
            }
        )

    return {
        "n_failed": n_failed,
        "n_failed_with_turns": n_with_turns,
        "turn_coverage_note": (
            "无 tool_call / 非 finish / 过早 finish 依赖已落库的 assistant turn；"
            f"当前非通过 attempt 中有 turn 标记的 {n_with_turns}/{n_failed}。"
            if n_failed
            else "当前没有非通过 attempt。"
        ),
        "cards": cards,
    }


def build_reward_diagnosis(task_id: int) -> Dict[str, Any]:
    """组装任务奖励诊断页 / JSON 载荷。"""
    from data.rollout_insight_exception import RolloutInsightException
    from db.db_attempt_service import DBAttemptService
    from db.db_segment_turn_service import DBSegmentTurnService
    from db.db_task_service import DBTaskService

    task = DBTaskService.get_by_id(int(task_id))
    if task is None:
        raise RolloutInsightException(
            RolloutInsightException.TASK_NOT_FOUND, f"task_id={task_id} 不存在"
        )

    rows = DBAttemptService.list_outcome_rows_by_task(int(task_id))
    coverage = DBSegmentTurnService.segment_turn_coverage(int(task_id))
    n_segments = int(coverage.get("n_segments") or 0)
    n_with_turns = int(coverage.get("n_segments_with_turns") or 0)
    turn_flags: List[Dict[str, Any]] = []
    if n_with_turns > 0:
        turn_flags = DBSegmentTurnService.list_assistant_turn_flags(int(task_id))

    gradient = summarize_gradient_groups(rows)
    reward_split = summarize_reward_split(rows)
    behavior = summarize_behavior_cards(
        rows,
        turn_flags,
        reward_split_series=reward_split["series"],
    )

    logger.info(
        "奖励诊断 task_id={} attempts={} rollouts={} turns={}",
        task_id,
        len(rows),
        reward_split["n_rollouts"],
        len(turn_flags),
    )
    return {
        "task": {
            "id": int(task.primary_id),
            "task_name": task.task_name,
            "status": getattr(task, "status", None),
            "rollout_root": getattr(task, "rollout_root", None),
        },
        "n_attempts": len(rows),
        "turn_coverage": {
            "n_segments": n_segments,
            "n_segments_with_turns": n_with_turns,
            "ratio": (n_with_turns / n_segments) if n_segments else None,
        },
        "gradient_groups": gradient,
        "reward_split": reward_split,
        "behavior": behavior,
        "page_note": (
            "先看可学习组占比与奖励分解，再看行为卡片对应的奖励杠杆；"
            "样例链到失败详情页。"
        ),
    }
