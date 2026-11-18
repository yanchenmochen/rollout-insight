#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""任务问题分析：跨 epoch 通过率，以及 sglang assistant 回复的 tool_call 统计。

通过率只读 attempt 表。响应统计优先用已落库的 turn 标记；覆盖不全时按 rollout
流式切 assistant 回复并只累计计数，不把正文写入 turn 表。
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

__all__ = [
    "build_pass_trend",
    "build_response_tool_stats",
    "advance_response_tool_stats",
    "build_instance_appearances",
    "group_instance_appearances",
    "summarize_pass_visits",
    "summarize_response_attempts",
    "merge_response_stats",
    "public_response_stats",
]

_BIN_WIDTH = 0.125
_N_BINS = 16
_RATE_EPS = 1e-9
_VALID_TOOLS = frozenset({"execute_bash", "str_replace_editor", "think", "finish"})
_CACHE_NAME = ".ri_response_stats.json"
_CACHE_VERSION = 1
_PAGE_NOTE = (
    "通过率 = 该次出现中 reward≈1 的条数 / 该次 attempt 条数；"
    "abort 与 timeout 计入分母、不计入分子。"
    "变好或变坏只比较该题最后一次与第一次。"
)


def summarize_pass_visits(
    rows: Sequence[Dict[str, Any]],
    *,
    batch_size: Optional[int] = None,
    batch_size_source: str = "missing",
) -> Dict[str, Any]:
    """把 attempt 结局行收成按题目出现次序的通过率。

    一次出现 = 同一个 ``rollout_id`` 下的同一个 ``instance_id``。
    出现次序按 ``rollout_id`` 升序，第 k 次即该题的第 k 个 epoch。
    ``rows`` 每项需含 ``rollout_id`` / ``group_index`` / ``instance_id`` / ``category``
    （``category`` 为 pass 才计入分子）。
    """
    buckets: Dict[Tuple[int, str], Dict[str, int]] = {}
    groups_per_rollout: Dict[int, set] = defaultdict(set)
    instances_per_rollout: Dict[int, set] = defaultdict(set)
    n_without_instance = 0

    for row in rows:
        if row.get("rollout_id") is None:
            continue
        rid = int(row["rollout_id"])
        gi = row.get("group_index")
        groups_per_rollout[rid].add(int(gi) if gi is not None else -1)
        iid = str(row.get("instance_id") or "").strip()
        if not iid:
            n_without_instance += 1
            continue
        instances_per_rollout[rid].add(iid)
        bucket = buckets.setdefault((rid, iid), {"n_pass": 0, "n_attempts": 0})
        bucket["n_attempts"] += 1
        if row.get("category") == "pass":
            bucket["n_pass"] += 1

    by_instance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (rid, iid), bucket in buckets.items():
        n_attempts = int(bucket["n_attempts"])
        if n_attempts <= 0:
            continue
        n_pass = int(bucket["n_pass"])
        by_instance[iid].append(
            {
                "rollout_id": rid,
                "n_pass": n_pass,
                "n_attempts": n_attempts,
                "rate": n_pass / n_attempts,
            }
        )

    tasks: List[Dict[str, Any]] = []
    for iid, visits in by_instance.items():
        visits.sort(key=lambda item: item["rollout_id"])
        first = float(visits[0]["rate"])
        last = float(visits[-1]["rate"])
        if len(visits) < 2:
            delta = None
            trend = "once"
        else:
            delta = last - first
            if delta > _RATE_EPS:
                trend = "improved"
            elif delta < -_RATE_EPS:
                trend = "worsened"
            else:
                trend = "unchanged"
        tasks.append(
            {
                "instance_id": iid,
                "visits": visits,
                "first_rate": first,
                "last_rate": last,
                "delta": delta,
                "trend": trend,
            }
        )

    trend_order = {"worsened": 0, "improved": 1, "unchanged": 2, "once": 3}
    tasks.sort(
        key=lambda item: (
            trend_order.get(item["trend"], 9),
            -(abs(item["delta"]) if item["delta"] is not None else 0.0),
            item["instance_id"],
        )
    )

    summary = {"improved": 0, "unchanged": 0, "worsened": 0, "once": 0}
    deltas: List[float] = []
    for item in tasks:
        summary[item["trend"]] = summary.get(item["trend"], 0) + 1
        if item["delta"] is not None:
            deltas.append(float(item["delta"]))

    n_problems = len(tasks)
    n_rollouts = len(groups_per_rollout)
    if batch_size is None:
        batch_size, batch_size_source = _mode_batch_size(groups_per_rollout)
    steps_per_epoch = None
    steps_exact = False
    if batch_size and batch_size > 0 and n_problems:
        steps_per_epoch = n_problems / float(batch_size)
        steps_exact = abs(steps_per_epoch - round(steps_per_epoch)) < 1e-9
    n_epochs_observed = max((len(item["visits"]) for item in tasks), default=0)
    n_epochs_from_steps = None
    if steps_per_epoch:
        n_epochs_from_steps = n_rollouts / steps_per_epoch

    alignment = _check_epoch_windows(
        sorted(groups_per_rollout),
        instances_per_rollout,
        steps_per_epoch if steps_exact else None,
    )

    return {
        "n_problems": n_problems,
        "n_rollouts": n_rollouts,
        "n_attempts_without_instance": n_without_instance,
        "batch_size": batch_size,
        "batch_size_source": batch_size_source,
        "steps_per_epoch": steps_per_epoch,
        "steps_per_epoch_exact": steps_exact,
        "n_epochs_observed": n_epochs_observed,
        "n_epochs_from_steps": n_epochs_from_steps,
        "epoch_aligned": alignment["epoch_aligned"],
        "alignment_checked": alignment["alignment_checked"],
        "alignment_violations": alignment["alignment_violations"],
        "summary": summary,
        "histogram": _histogram(deltas),
        "tasks": tasks,
        "note": _PAGE_NOTE,
    }


def group_instance_appearances(
    instance_id: str,
    rows: Sequence[Dict[str, Any]],
    rollout_pks: Optional[Dict[int, int]] = None,
    segment_ids: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """把一道题的 attempt 按 rollout 出现次序收成可跳转的列表。

    ``rollout_pks`` 是业务 ``rollout_id`` → rollout 表主键。
    ``segment_ids`` 是 attempt 主键 → 第一条 segment 主键。
    """
    name = str(instance_id or "").strip()
    pks = rollout_pks or {}
    segments = segment_ids or {}
    grouped: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if str(row.get("instance_id") or "").strip() != name:
            continue
        if row.get("rollout_id") is None:
            continue
        rid = int(row["rollout_id"])
        visit = grouped.setdefault(
            rid,
            {
                "rollout_id": rid,
                "rollout_pk": pks.get(rid),
                "n_pass": 0,
                "n_attempts": 0,
                "groups": set(),
                "attempts": [],
            },
        )
        visit["n_attempts"] += 1
        if row.get("category") == "pass":
            visit["n_pass"] += 1
        gi = row.get("group_index")
        if gi is not None:
            visit["groups"].add(int(gi))
        attempt_pk = row.get("attempt_pk")
        visit["attempts"].append(
            {
                "attempt_pk": int(attempt_pk) if attempt_pk is not None else None,
                "attempt_id": row.get("attempt_id"),
                "group_index": gi,
                "reward": row.get("reward"),
                "category": row.get("category") or "other",
                "segment_id": segments.get(int(attempt_pk)) if attempt_pk is not None else None,
            }
        )

    visits: List[Dict[str, Any]] = []
    for index, rid in enumerate(sorted(grouped), start=1):
        visit = grouped[rid]
        n_attempts = int(visit["n_attempts"])
        visit["epoch"] = index
        visit["groups"] = sorted(visit["groups"])
        visit["rate"] = (visit["n_pass"] / n_attempts) if n_attempts else None
        visit["attempts"].sort(
            key=lambda item: (
                int(item["group_index"]) if item.get("group_index") is not None else -1,
                int(item["attempt_id"]) if isinstance(item.get("attempt_id"), int) else 0,
                item.get("attempt_pk") or 0,
            )
        )
        visits.append(visit)
    return {
        "instance_id": name,
        "n_visits": len(visits),
        "visits": visits,
    }


def build_instance_appearances(task_id: int, instance_id: str) -> Dict[str, Any]:
    """按题目名列出每次出现的 rollout 与 attempt，供轨迹跳转。"""
    from data.rollout_insight_exception import RolloutInsightException
    from db.db_attempt_service import DBAttemptService
    from db.db_rollout_service import DBRolloutService
    from db.db_segment_service import DBSegmentService
    from db.db_task_service import DBTaskService

    name = str(instance_id or "").strip()
    if not name:
        raise RolloutInsightException(
            RolloutInsightException.INVALID_ARGUMENT, "instance_id 不能为空"
        )
    task = DBTaskService.get_by_id(int(task_id))
    if task is None:
        raise RolloutInsightException(
            RolloutInsightException.TASK_NOT_FOUND, f"task_id={task_id} 不存在"
        )
    rows = DBAttemptService.list_outcome_rows_by_task(int(task_id))
    matched = [row for row in rows if str(row.get("instance_id") or "").strip() == name]
    rollout_rows = DBRolloutService.list_by_task(int(task_id), limit=100000, offset=0)
    rollout_pks = {
        int(row.rollout_id): int(row.primary_id)
        for row in rollout_rows
        if row.rollout_id is not None
    }
    attempt_pks = [
        int(row["attempt_pk"]) for row in matched if row.get("attempt_pk") is not None
    ]
    segment_ids = DBSegmentService.map_first_segment_id(attempt_pks)
    view = group_instance_appearances(name, matched, rollout_pks, segment_ids)
    logger.info(
        "题目出现 task_id={} instance_id={} visits={}",
        task_id,
        name,
        view["n_visits"],
    )
    return view


def build_pass_trend(task_id: int) -> Dict[str, Any]:
    """按任务读取 attempt 结局，计算题目通过率曲线。"""
    from data.rollout_insight_exception import RolloutInsightException
    from db.db_attempt_service import DBAttemptService
    from db.db_task_service import DBTaskService

    task = DBTaskService.get_by_id(int(task_id))
    if task is None:
        raise RolloutInsightException(
            RolloutInsightException.TASK_NOT_FOUND, f"task_id={task_id} 不存在"
        )
    rows = DBAttemptService.list_outcome_rows_by_task(int(task_id))
    batch_size, source = _batch_size_from_task(task, rows)
    trend = summarize_pass_visits(rows, batch_size=batch_size, batch_size_source=source)
    logger.info(
        "问题分析通过率 task_id={} problems={} epochs={} batch={} source={}",
        task_id,
        trend["n_problems"],
        trend["n_epochs_observed"],
        trend["batch_size"],
        trend["batch_size_source"],
    )
    return trend


def summarize_response_attempts(attempts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """按 attempt 的 assistant 回复列表累计两个维度。

    每个 attempt 为 ``{"assistant_turns": [{"has_tool_call", "tool_call_names"}, ...]}``，
    顺序即轨迹顺序。最后一条是否 finish 只看该 attempt 最后一条 assistant 回复的
    最后一个工具名。
    """
    raw = _empty_response_stats()
    raw["n_attempts"] = len(attempts)
    for attempt in attempts:
        turns = list(attempt.get("assistant_turns") or [])
        if not turns:
            continue
        raw["n_attempts_with_assistant"] += 1
        any_without = False
        for turn in turns:
            names = _tool_names(turn)
            has_call = bool(turn.get("has_tool_call")) or bool(names)
            raw["n_assistant"] += 1
            if not has_call:
                raw["n_without_tool_call"] += 1
                any_without = True
                continue
            raw["n_with_tool_call"] += 1
            invalid = (not names) or any(name.lower() not in _VALID_TOOLS for name in names)
            if invalid:
                raw["n_invalid_responses"] += 1
            for name in names:
                raw["tool_names"][name] = int(raw["tool_names"].get(name, 0)) + 1
                if name.lower() == "finish":
                    raw["n_finish_calls"] += 1
                elif name.lower() not in _VALID_TOOLS:
                    raw["invalid_names"][name] = int(raw["invalid_names"].get(name, 0)) + 1
        last_names = _tool_names(turns[-1])
        last_finish = bool(last_names) and last_names[-1].lower() == "finish"
        if any_without:
            raw["n_attempts_no_tool_call"] += 1
        if last_finish:
            raw["n_attempts_last_is_finish"] += 1
        else:
            raw["n_attempts_last_not_finish"] += 1
        if any_without or not last_finish:
            raw["n_attempts_incomplete"] += 1
    return raw


def merge_response_stats(parts: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """把若干 rollout 的原始计数加总。"""
    merged = _empty_response_stats()
    for part in parts:
        if not part:
            continue
        for key in (
            "n_assistant",
            "n_with_tool_call",
            "n_without_tool_call",
            "n_finish_calls",
            "n_invalid_responses",
            "n_attempts",
            "n_attempts_with_assistant",
            "n_attempts_no_tool_call",
            "n_attempts_last_not_finish",
            "n_attempts_last_is_finish",
            "n_attempts_incomplete",
        ):
            merged[key] += int(part.get(key) or 0)
        _add_counts(merged["tool_names"], part.get("tool_names") or {})
        _add_counts(merged["invalid_names"], part.get("invalid_names") or {})
    return merged


def public_response_stats(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """把内部计数字典转成页面用的列表，并附上比例。"""
    if not raw:
        return None
    assistant = int(raw.get("n_assistant") or 0)
    with_assistant = int(raw.get("n_attempts_with_assistant") or 0)

    def _ratio(num: int, den: int) -> Optional[float]:
        if den <= 0:
            return None
        return num / den

    return {
        "n_assistant": assistant,
        "n_with_tool_call": int(raw.get("n_with_tool_call") or 0),
        "n_without_tool_call": int(raw.get("n_without_tool_call") or 0),
        "n_finish_calls": int(raw.get("n_finish_calls") or 0),
        "n_invalid_responses": int(raw.get("n_invalid_responses") or 0),
        "tool_names": _count_list(raw.get("tool_names") or {}),
        "invalid_names": _count_list(raw.get("invalid_names") or {}),
        "n_attempts": int(raw.get("n_attempts") or 0),
        "n_attempts_with_assistant": with_assistant,
        "n_attempts_no_tool_call": int(raw.get("n_attempts_no_tool_call") or 0),
        "n_attempts_last_not_finish": int(raw.get("n_attempts_last_not_finish") or 0),
        "n_attempts_last_is_finish": int(raw.get("n_attempts_last_is_finish") or 0),
        "n_attempts_incomplete": int(raw.get("n_attempts_incomplete") or 0),
        "ratios": {
            "with_tool_call": _ratio(int(raw.get("n_with_tool_call") or 0), assistant),
            "without_tool_call": _ratio(int(raw.get("n_without_tool_call") or 0), assistant),
            "attempts_no_tool_call": _ratio(
                int(raw.get("n_attempts_no_tool_call") or 0), with_assistant
            ),
            "attempts_last_not_finish": _ratio(
                int(raw.get("n_attempts_last_not_finish") or 0), with_assistant
            ),
            "attempts_incomplete": _ratio(
                int(raw.get("n_attempts_incomplete") or 0), with_assistant
            ),
        },
    }


def build_response_tool_stats(task_id: int) -> Dict[str, Any]:
    """返回当前能直接给出的响应统计：完整 turn、完整 .pt 缓存，或其中一部分。"""
    from data.rollout_insight_exception import RolloutInsightException
    from db.db_attempt_service import DBAttemptService
    from db.db_segment_turn_service import DBSegmentTurnService
    from db.db_task_service import DBTaskService

    task = DBTaskService.get_by_id(int(task_id))
    if task is None:
        raise RolloutInsightException(
            RolloutInsightException.TASK_NOT_FOUND, f"task_id={task_id} 不存在"
        )
    coverage = _coverage_view(DBSegmentTurnService.segment_turn_coverage(int(task_id)))
    rollouts = _rollout_fingerprints(int(task_id))
    n_attempts = len(DBAttemptService.list_outcome_rows_by_task(int(task_id)))
    cache_path = _cache_path(getattr(task, "rollout_root", "") or "")
    matched, pending = _match_cache(rollouts, _read_cache(cache_path))
    progress = _progress(rollouts, pending)

    if rollouts and not pending and matched:
        raw = merge_response_stats(item["stats"] for item in matched)
        return _response_view(
            source="pt_cache",
            complete=True,
            coverage=coverage,
            progress=progress,
            n_attempts_in_task=n_attempts,
            stats=public_response_stats(raw),
        )

    flags = DBSegmentTurnService.list_assistant_turn_flags(int(task_id))
    db_public = public_response_stats(_stats_from_flags(flags)) if flags else None
    db_complete = coverage["n_segments"] > 0 and coverage["ratio"] == 1.0
    if db_complete and db_public is not None:
        return _response_view(
            source="turns",
            complete=True,
            coverage=coverage,
            progress=progress,
            n_attempts_in_task=n_attempts,
            stats=db_public,
        )
    if matched:
        raw = merge_response_stats(item["stats"] for item in matched)
        return _response_view(
            source="pt_cache",
            complete=False,
            coverage=coverage,
            progress=progress,
            n_attempts_in_task=n_attempts,
            stats=public_response_stats(raw),
        )
    if db_public is not None:
        return _response_view(
            source="partial_turns",
            complete=False,
            coverage=coverage,
            progress=progress,
            n_attempts_in_task=n_attempts,
            stats=db_public,
        )
    return _response_view(
        source="none",
        complete=False,
        coverage=coverage,
        progress=progress,
        n_attempts_in_task=n_attempts,
        stats=None,
    )


def advance_response_tool_stats(
    task_id: int,
    rollout_limit: int = 1,
) -> Dict[str, Any]:
    """按指纹补统计尚未覆盖的 rollout，只计数，不写 turn 正文。"""
    from data.rollout_insight_exception import RolloutInsightException
    from db.db_attempt_service import DBAttemptService
    from db.db_segment_turn_service import DBSegmentTurnService
    from db.db_task_service import DBTaskService

    task = DBTaskService.get_by_id(int(task_id))
    if task is None:
        raise RolloutInsightException(
            RolloutInsightException.TASK_NOT_FOUND, f"task_id={task_id} 不存在"
        )
    limit = max(1, int(rollout_limit))
    rollouts = _rollout_fingerprints(int(task_id))
    cache_path = _cache_path(getattr(task, "rollout_root", "") or "")
    cache = _read_cache(cache_path)
    matched, pending = _match_cache(rollouts, cache)
    errors: List[Dict[str, Any]] = []
    batch = pending[:limit]
    if cache_path is None and batch:
        errors.append({"rollout_id": None, "error": "任务没有 rollout_root，无法写入统计缓存"})
        batch = []
    for item in batch:
        path = item["pt_path"]
        try:
            stats = count_rollout_file(path)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "响应统计失败 task_id={} rollout_id={} path={} err={}",
                task_id,
                item["rollout_id"],
                path,
                exc,
            )
            errors.append({"rollout_id": item["rollout_id"], "error": str(exc)})
            continue
        cache.setdefault("rollouts", {})[str(item["rollout_id"])] = {
            "pt_path": path,
            "pt_size_bytes": item["pt_size_bytes"],
            "pt_mtime": item["pt_mtime"],
            "stats": stats,
        }
        logger.info(
            "响应统计 task_id={} rollout_id={} assistant={} attempts={}",
            task_id,
            item["rollout_id"],
            stats.get("n_assistant"),
            stats.get("n_attempts"),
        )
    if cache_path is not None and batch:
        cache["version"] = _CACHE_VERSION
        cache["task_id"] = int(task_id)
        try:
            _write_cache(cache_path, cache)
        except OSError as exc:
            logger.warning("响应统计缓存写入失败 path={} err={}", cache_path, exc)
            errors.append({"rollout_id": None, "error": str(exc)})

    matched, pending = _match_cache(rollouts, cache)
    raw = merge_response_stats(item["stats"] for item in matched) if matched else None
    coverage = _coverage_view(DBSegmentTurnService.segment_turn_coverage(int(task_id)))
    n_attempts = len(DBAttemptService.list_outcome_rows_by_task(int(task_id)))
    view = _response_view(
        source="pt_cache" if matched else "none",
        complete=bool(rollouts) and not pending and bool(matched),
        coverage=coverage,
        progress=_progress(rollouts, pending),
        n_attempts_in_task=n_attempts,
        stats=public_response_stats(raw),
        errors=errors,
    )
    return view


def count_rollout_file(pt_path: str) -> Dict[str, Any]:
    """读取一份 .pt，只累计 assistant 回复计数。"""
    from utils.rollout_pt_parser import (
        load_pt,
        parse_attempts_and_segments,
        split_response_to_turns,
    )

    parsed = parse_attempts_and_segments(load_pt(pt_path))
    attempts: List[Dict[str, Any]] = []
    for attempt in parsed.get("attempts") or []:
        turns: List[Dict[str, Any]] = []
        segments = list(attempt.get("segments") or [])
        segments.sort(
            key=lambda seg: seg.get("segment_index")
            if isinstance(seg.get("segment_index"), int)
            else 0
        )
        for seg in segments:
            text = seg.get("response_text") or ""
            if not isinstance(text, str) or not text:
                continue
            for turn in split_response_to_turns(text):
                if (turn.get("role_label") or "") != "assistant":
                    continue
                turns.append(
                    {
                        "has_tool_call": bool(turn.get("has_tool_call")),
                        "tool_call_names": list(turn.get("tool_call_names") or []),
                    }
                )
        attempts.append({"assistant_turns": turns})
    return summarize_response_attempts(attempts)


def _stats_from_flags(flags: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for flag in flags:
        grouped[int(flag["attempt_pk"])].append(flag)
    attempts = []
    for pk in sorted(grouped):
        turns = sorted(
            grouped[pk],
            key=lambda item: (int(item.get("segment_index") or 0), int(item.get("turn_index") or 0)),
        )
        attempts.append(
            {
                "assistant_turns": [
                    {
                        "has_tool_call": bool(item.get("has_tool_call")),
                        "tool_call_names": list(item.get("tool_call_names") or []),
                    }
                    for item in turns
                ]
            }
        )
    return summarize_response_attempts(attempts)


def _empty_response_stats() -> Dict[str, Any]:
    return {
        "n_assistant": 0,
        "n_with_tool_call": 0,
        "n_without_tool_call": 0,
        "tool_names": {},
        "n_finish_calls": 0,
        "n_invalid_responses": 0,
        "invalid_names": {},
        "n_attempts": 0,
        "n_attempts_with_assistant": 0,
        "n_attempts_no_tool_call": 0,
        "n_attempts_last_not_finish": 0,
        "n_attempts_last_is_finish": 0,
        "n_attempts_incomplete": 0,
    }


def _tool_names(turn: Dict[str, Any]) -> List[str]:
    names = []
    for item in turn.get("tool_call_names") or []:
        text = str(item).strip()
        if text:
            names.append(text)
    return names


def _add_counts(target: Dict[str, int], source: Dict[str, Any]) -> None:
    for name, count in source.items():
        target[str(name)] = int(target.get(str(name), 0)) + int(count or 0)


def _count_list(counts: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [
        {"name": str(name), "count": int(count)}
        for name, count in counts.items()
        if int(count or 0) > 0
    ]
    rows.sort(key=lambda item: (-item["count"], item["name"]))
    return rows


def _histogram(deltas: Sequence[float]) -> List[Dict[str, Any]]:
    counts = [0] * _N_BINS
    for delta in deltas:
        if delta >= 1:
            idx = _N_BINS - 1
        elif delta <= -1:
            idx = 0
        else:
            idx = int((delta - (-1.0)) / _BIN_WIDTH)
            idx = max(0, min(_N_BINS - 1, idx))
        counts[idx] += 1
    bins = []
    for idx, count in enumerate(counts):
        lo = -1.0 + idx * _BIN_WIDTH
        hi = lo + _BIN_WIDTH
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "count": count,
                "label": f"{lo:.3f}~{hi:.3f}",
            }
        )
    return bins


def _mode_batch_size(groups_per_rollout: Dict[int, set]) -> Tuple[Optional[int], str]:
    if not groups_per_rollout:
        return None, "missing"
    counts = [len(groups) for groups in groups_per_rollout.values() if groups]
    if not counts:
        return None, "missing"
    mode, _ = Counter(counts).most_common(1)[0]
    return int(mode), "group_mode"


def _batch_size_from_task(task: Any, rows: Sequence[Dict[str, Any]]) -> Tuple[Optional[int], str]:
    groups: Dict[int, set] = defaultdict(set)
    for row in rows:
        if row.get("rollout_id") is None:
            continue
        gi = row.get("group_index")
        groups[int(row["rollout_id"])].add(int(gi) if gi is not None else -1)
    env_size = _env_batch_size(getattr(task, "rollout_root", "") or "")
    if env_size:
        return env_size, "env"
    return _mode_batch_size(groups)


def _env_batch_size(rollout_root: str) -> Optional[int]:
    try:
        from utils.slime_job_type import load_env_from_run_dir
        from utils.tb_run_paths import resolve_run_dir
    except Exception:
        return None
    run_dir = resolve_run_dir(rollout_root)
    env, _, _ = load_env_from_run_dir(run_dir)
    raw = str(env.get("ROLLOUT_BATCH_SIZE") or "").strip()
    if not raw:
        return None
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _check_epoch_windows(
    rollout_ids: Sequence[int],
    instances_per_rollout: Dict[int, set],
    steps_per_epoch: Optional[float],
) -> Dict[str, Any]:
    if not steps_per_epoch or steps_per_epoch < 1:
        return {
            "alignment_checked": False,
            "epoch_aligned": None,
            "alignment_violations": [],
        }
    step_n = int(round(steps_per_epoch))
    seen: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for index, rid in enumerate(rollout_ids):
        window = index // step_n
        for iid in instances_per_rollout.get(rid, ()):
            seen[window][iid].append(rid)
    violations = []
    for window, mapping in sorted(seen.items()):
        for iid, rids in sorted(mapping.items()):
            if len(rids) > 1:
                violations.append(
                    {
                        "window": window,
                        "instance_id": iid,
                        "rollout_ids": rids,
                        "count": len(rids),
                    }
                )
    return {
        "alignment_checked": True,
        "epoch_aligned": not violations,
        "alignment_violations": violations[:8],
    }


def _rollout_fingerprints(task_id: int) -> List[Dict[str, Any]]:
    from db.db_rollout_service import DBRolloutService

    rows = DBRolloutService.list_by_task(int(task_id), limit=100000, offset=0)
    out = []
    for row in rows:
        out.append(
            {
                "rollout_id": int(row.rollout_id),
                "pt_path": row.pt_path or "",
                "pt_size_bytes": int(row.pt_size_bytes or 0),
                "pt_mtime": _mtime_token(row.pt_mtime),
            }
        )
    return out


def _cache_path(rollout_root: str) -> Optional[Path]:
    text = (rollout_root or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_file():
        path = path.parent
    return path / _CACHE_NAME


def _read_cache(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {"version": _CACHE_VERSION, "rollouts": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": _CACHE_VERSION, "rollouts": {}}
    if not isinstance(payload, dict) or int(payload.get("version") or 0) != _CACHE_VERSION:
        return {"version": _CACHE_VERSION, "rollouts": {}}
    if not isinstance(payload.get("rollouts"), dict):
        payload["rollouts"] = {}
    return payload


def _write_cache(path: Path, payload: Dict[str, Any]) -> None:
    if not path.parent.is_dir():
        raise OSError(f"缓存目录不存在: {path.parent}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _mtime_token(value: Any) -> str:
    """mtime 统一到微秒字符串，避免 JSON 浮点往返后缓存永远对不上。"""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def _match_cache(
    rollouts: Sequence[Dict[str, Any]],
    cache: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    stored = cache.get("rollouts") if isinstance(cache.get("rollouts"), dict) else {}
    matched: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for item in rollouts:
        saved = stored.get(str(item["rollout_id"]))
        stats = saved.get("stats") if isinstance(saved, dict) else None
        same = (
            isinstance(saved, dict)
            and saved.get("pt_path") == item["pt_path"]
            and int(saved.get("pt_size_bytes") or 0) == int(item["pt_size_bytes"])
            and _mtime_token(saved.get("pt_mtime")) == _mtime_token(item["pt_mtime"])
            and isinstance(stats, dict)
        )
        if same:
            matched.append({"rollout_id": item["rollout_id"], "stats": stats})
        else:
            pending.append(item)
    return matched, pending


def _coverage_view(raw: Dict[str, int]) -> Dict[str, Any]:
    n_segments = int(raw.get("n_segments") or 0)
    n_with = int(raw.get("n_segments_with_turns") or 0)
    ratio = (n_with / n_segments) if n_segments else 0.0
    return {
        "n_segments": n_segments,
        "n_segments_with_turns": n_with,
        "ratio": ratio,
    }


def _progress(
    rollouts: Sequence[Dict[str, Any]],
    pending: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    pending_ids = [int(item["rollout_id"]) for item in pending]
    return {
        "processed_rollouts": len(rollouts) - len(pending_ids),
        "total_rollouts": len(rollouts),
        "pending_rollout_ids": pending_ids,
    }


def _response_view(
    *,
    source: str,
    complete: bool,
    coverage: Dict[str, Any],
    progress: Dict[str, Any],
    n_attempts_in_task: int,
    stats: Optional[Dict[str, Any]],
    errors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "source": source,
        "complete": bool(complete),
        "coverage": coverage,
        "progress": progress,
        "n_attempts_in_task": int(n_attempts_in_task),
        "stats": stats,
        "errors": list(errors or []),
    }
