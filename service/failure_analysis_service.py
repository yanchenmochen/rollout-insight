#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""任务级失败样本分析：把 ri_attempt_table 与 run 目录 ``attempt_artifacts`` 关联。

- 总览：每个 rollout 的结局分布 + 非通过 attempt 列表 + 按原因聚类
- 详情：单个 attempt 的结论 / 评测证据 / 耗时 / patch / OpenHands 轨迹
- 无 artifacts 时回退 ``.pt`` 注解（sidecar 存在才读）与 ``failure_insight``
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from data.rollout_insight_exception import RolloutInsightException
from utils.attempt_artifacts import (
    find_trajectory_path,
    load_patch,
    load_summary,
    load_summary_index,
    load_trajectory_events,
    locate_artifacts_dir,
    strip_ansi,
)

__all__ = [
    "FAILURE_CATEGORIES",
    "link_attempts_to_entries",
    "build_failure_clusters",
    "list_tasks_with_failure_counts",
    "get_task_failure_overview",
    "get_attempt_failure_detail",
    "get_attempt_trajectory",
]

FAILURE_CATEGORIES = ("fail", "timeout", "abort", "other")

_MATCH_EXACT = "exact"
_MATCH_GROUP_UNIQUE = "group_unique"
_MATCH_GROUP_ORDER = "group_order"
_STDOUT_MAX = 20_000
_SUB_CLUSTER_TOP = 8


# ---------------------------------------------------------------------------
# 关联：DB attempt ↔ artifacts summary
# ---------------------------------------------------------------------------

def link_attempts_to_entries(
    rows: List[Dict[str, Any]], entries: List[Dict[str, Any]]
) -> Dict[int, Tuple[Dict[str, Any], str]]:
    """返回 ``{attempt_pk: (entry, match)}``。

    1. ``(group_index, rollout_id) == (group_index, attempt_id)`` 精确匹配；
    2. 剩余（abort 样本 summary.rollout_id 为空）按 ``(group_index, instance_id)``：
       候选唯一 → ``group_unique``；多个 → 按 attempt_id / sample_index 顺序配对 ``group_order``。
    """
    exact: Dict[Tuple[int, int], Dict[str, Any]] = {}
    loose: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        gi = e.get("group_index")
        if gi is None:
            continue
        if e.get("rollout_id") is not None:
            exact[(int(gi), int(e["rollout_id"]))] = e
        else:
            loose[(int(gi), str(e.get("instance_id") or ""))].append(e)

    out: Dict[int, Tuple[Dict[str, Any], str]] = {}
    pending: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        gi, aid = r.get("group_index"), r.get("attempt_id")
        if gi is None or aid is None:
            continue
        hit = exact.get((int(gi), int(aid)))
        if hit is not None:
            out[int(r["attempt_pk"])] = (hit, _MATCH_EXACT)
        else:
            pending[(int(gi), str(r.get("instance_id") or ""))].append(r)

    for key, group_rows in pending.items():
        cands = loose.get(key) or []
        if not cands:
            continue
        if len(cands) == 1 and len(group_rows) == 1:
            out[int(group_rows[0]["attempt_pk"])] = (cands[0], _MATCH_GROUP_UNIQUE)
            continue
        ordered_rows = sorted(group_rows, key=lambda r: int(r.get("attempt_id") or 0))
        ordered_cands = sorted(cands, key=lambda e: int(e.get("sample_index") or 0))
        for r, e in zip(ordered_rows, ordered_cands):
            out[int(r["attempt_pk"])] = (e, _MATCH_GROUP_ORDER)
    return out


# ---------------------------------------------------------------------------
# 原因 / 聚类
# ---------------------------------------------------------------------------

def _reason_of(row: Dict[str, Any], entry: Optional[Dict[str, Any]], ann: Optional[Dict[str, Any]]) -> str:
    if entry:
        if entry.get("abort_reason"):
            return str(entry["abort_reason"])
        if entry.get("diagnosis"):
            return str(entry["diagnosis"])
        if entry.get("agent_outcome"):
            return f"agent:{entry['agent_outcome']}"
    if ann:
        reason = ann.get("abort_reason") or ann.get("outcome_reason")
        if reason:
            return str(reason)
    return f"status:{row.get('status') or 'unknown'}"


def _sub_key_of(item: Dict[str, Any]) -> Optional[str]:
    return item.get("failed_test") or item.get("error_summary") or None


def build_failure_clusters(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 ``(category, reason)`` 聚类非通过 attempt；每簇附 rollout 分布与子原因 Top。"""
    buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for it in items:
        key = (str(it.get("category")), str(it.get("reason")))
        b = buckets.setdefault(
            key,
            {"category": key[0], "reason": key[1], "count": 0,
             "by_rollout": Counter(), "sub": Counter(), "instances": set()},
        )
        b["count"] += 1
        if it.get("rollout_id") is not None:
            b["by_rollout"][int(it["rollout_id"])] += 1
        sub = _sub_key_of(it)
        if sub:
            b["sub"][sub] += 1
        if it.get("instance_id"):
            b["instances"].add(it["instance_id"])
    total = sum(b["count"] for b in buckets.values()) or 1
    out: List[Dict[str, Any]] = []
    for b in sorted(buckets.values(), key=lambda x: (-x["count"], x["category"], x["reason"])):
        out.append({
            "category": b["category"],
            "reason": b["reason"],
            "count": b["count"],
            "pct": round(100.0 * b["count"] / total, 1),
            "n_instances": len(b["instances"]),
            "by_rollout": [{"rollout_id": k, "count": v} for k, v in sorted(b["by_rollout"].items())],
            "sub": [{"key": k, "count": v} for k, v in b["sub"].most_common(_SUB_CLUSTER_TOP)],
        })
    return out


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------

def _require_task(task_id: int):
    from db.db_task_service import DBTaskService

    task = DBTaskService.get_by_id(int(task_id))
    if task is None:
        raise RolloutInsightException(RolloutInsightException.TASK_NOT_FOUND, f"任务 #{task_id} 不存在")
    return task


def _task_brief(task: Any) -> Dict[str, Any]:
    return {
        "id": int(task.primary_id),
        "task_name": task.task_name,
        "status": task.status,
        "rollout_root": task.rollout_root or "",
    }


def _rollout_maps(task_id: int) -> Tuple[Dict[int, int], Dict[int, str]]:
    """业务 rollout_id → (rollout 主键, pt_path)。"""
    from db.db_rollout_service import DBRolloutService

    pk_of: Dict[int, int] = {}
    pt_of: Dict[int, str] = {}
    for r in DBRolloutService.list_by_task(int(task_id), limit=100_000):
        if r.rollout_id is None:
            continue
        pk_of[int(r.rollout_id)] = int(r.primary_id)
        pt_of[int(r.rollout_id)] = r.pt_path or ""
    return pk_of, pt_of


def _artifact_entries(rollout_root: str) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    artifacts_dir = locate_artifacts_dir(rollout_root)
    if artifacts_dir is None:
        return None, []
    try:
        return artifacts_dir, load_summary_index(artifacts_dir)
    except OSError as exc:
        logger.warning("读取 attempt_artifacts 失败 dir={} err={}", artifacts_dir, exc)
        return artifacts_dir, []


def _sidecar_outcomes(pt_path: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """仅当 .pt 注解 sidecar 已存在时读取结局索引，避免为列表页冷读 .pt。"""
    from service.task_annotation_service import load_pt_annotations, sidecar_path_for

    if not pt_path or not os.path.isfile(sidecar_path_for(pt_path)):
        return {}
    return load_pt_annotations(pt_path).get("attempt_outcomes") or {}


def _attempt_item(
    row: Dict[str, Any],
    linked: Optional[Tuple[Dict[str, Any], str]],
    ann: Optional[Dict[str, Any]],
    rollout_pk: Optional[int],
) -> Dict[str, Any]:
    entry, match = linked if linked else (None, None)
    entry = entry or {}
    return {
        "attempt_pk": row["attempt_pk"],
        "rollout_pk": rollout_pk,
        "rollout_id": row.get("rollout_id"),
        "group_index": row.get("group_index"),
        "attempt_id": row.get("attempt_id"),
        "instance_id": row.get("instance_id") or entry.get("instance_id") or "",
        "status": row.get("status"),
        "category": row.get("category"),
        "reason": _reason_of(row, entry or None, ann),
        "failed_test": entry.get("failed_test"),
        "error_summary": entry.get("error_summary"),
        "agent_outcome": entry.get("agent_outcome"),
        "agent_turns": entry.get("agent_turns") if entry.get("agent_turns") is not None else row.get("agent_turns"),
        "tool_calls": entry.get("agent_tool_calls") if entry.get("agent_tool_calls") is not None else row.get("tool_calls"),
        "tool_errors": entry.get("agent_tool_errors"),
        "peak_context_tokens": entry.get("peak_context_tokens"),
        "diff_line_count": entry.get("diff_line_count"),
        "has_trajectory": bool(entry.get("trajectory_kind")),
        "match": match,
    }


def _progress_row(rollout_id: int, rollout_pk: Optional[int], rows: List[Dict[str, Any]], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(r.get("category") for r in rows)
    top = Counter(f"{it['category']}|{it['reason']}" for it in items).most_common(1)
    top_reason = None
    if top:
        cat, reason = top[0][0].split("|", 1)
        top_reason = {"category": cat, "reason": reason, "count": top[0][1]}
    return {
        "rollout_id": rollout_id,
        "rollout_pk": rollout_pk,
        "n_attempts": len(rows),
        "n_pass": counts.get("pass", 0),
        "n_fail": counts.get("fail", 0),
        "n_timeout": counts.get("timeout", 0),
        "n_abort": counts.get("abort", 0),
        "n_other": counts.get("other", 0),
        "top_reason": top_reason,
    }


# ---------------------------------------------------------------------------
# 对外：任务选择 / 总览
# ---------------------------------------------------------------------------

def list_tasks_with_failure_counts(limit: int = 50) -> List[Dict[str, Any]]:
    """失败分析入口的任务选择器：最近任务 + 各结局计数。"""
    from db.db_attempt_service import DBAttemptService
    from db.db_task_service import DBTaskService

    out: List[Dict[str, Any]] = []
    for task in DBTaskService.list_tasks(limit=limit):
        metrics = DBAttemptService.summarize_reward_metrics_by_rollout(int(task.primary_id))
        agg = Counter()
        for m in metrics.values():
            for key in ("n_attempts", "n_passed", "n_failed", "n_timeout", "n_aborted", "n_other"):
                agg[key] += int(m.get(key) or 0)
        brief = _task_brief(task)
        brief.update({
            "n_rollouts": len(metrics),
            "n_attempts": agg["n_attempts"],
            "n_pass": agg["n_passed"],
            "n_fail": agg["n_failed"],
            "n_timeout": agg["n_timeout"],
            "n_abort": agg["n_aborted"],
            "n_other": agg["n_other"],
        })
        out.append(brief)
    return out


def get_task_failure_overview(task_id: int, rollout_id: Optional[int] = None) -> Dict[str, Any]:
    """任务级失败总览：进度（每 rollout 结局分布）+ 非通过 attempt 列表 + 原因聚类。"""
    from db.db_attempt_service import DBAttemptService

    task = _require_task(task_id)
    rows = DBAttemptService.list_outcome_rows_by_task(int(task_id), rollout_id)
    pk_of, pt_of = _rollout_maps(int(task_id))
    artifacts_dir, entries = _artifact_entries(task.rollout_root or "")
    links = link_attempts_to_entries(rows, entries)

    rows_by_rollout: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("rollout_id") is not None:
            rows_by_rollout[int(r["rollout_id"])].append(r)

    items: List[Dict[str, Any]] = []
    progress: List[Dict[str, Any]] = []
    for rid in sorted(rows_by_rollout):
        rrows = rows_by_rollout[rid]
        outcomes = {} if entries else _sidecar_outcomes(pt_of.get(rid, ""))
        ritems = []
        for r in rrows:
            if r.get("category") == "pass":
                continue
            ann = outcomes.get((int(r["group_index"]), int(r["attempt_id"]))) if outcomes else None
            ritems.append(_attempt_item(r, links.get(int(r["attempt_pk"])), ann, pk_of.get(rid)))
        items.extend(ritems)
        progress.append(_progress_row(rid, pk_of.get(rid), rrows, ritems))

    totals = Counter(r.get("category") for r in rows)
    n_linked = sum(1 for it in items if it.get("match"))
    linked_stems = {links[pk][0]["stem"] for pk in links}
    return {
        "task": _task_brief(task),
        "rollout_filter": rollout_id,
        "source": "attempt_artifacts" if entries else "pt_annotations",
        "artifacts_dir": str(artifacts_dir) if artifacts_dir else "",
        "n_summaries": len(entries),
        "n_unlinked_summaries": 0 if rollout_id is not None else max(0, len(entries) - len(linked_stems)),
        "totals": {
            "n_attempts": len(rows),
            "n_pass": totals.get("pass", 0),
            "n_fail": totals.get("fail", 0),
            "n_timeout": totals.get("timeout", 0),
            "n_abort": totals.get("abort", 0),
            "n_other": totals.get("other", 0),
            "n_failure": len(items),
            "n_linked": n_linked,
        },
        "progress": progress,
        "clusters": build_failure_clusters(items),
        "attempts": items,
    }


# ---------------------------------------------------------------------------
# 对外：单 attempt 详情 / 轨迹
# ---------------------------------------------------------------------------

def _load_attempt_context(attempt_pk: int) -> Dict[str, Any]:
    from db.db_attempt_service import DBAttemptService

    attempt = DBAttemptService.get_by_id(int(attempt_pk))
    if attempt is None:
        raise RolloutInsightException(RolloutInsightException.INVALID_ARGUMENT, f"attempt #{attempt_pk} 不存在")
    task = _require_task(int(attempt.task_id))
    rows = DBAttemptService.list_outcome_rows_by_task(int(attempt.task_id), int(attempt.rollout_id))
    row = next((r for r in rows if r["attempt_pk"] == int(attempt_pk)), None)
    if row is None:
        raise RolloutInsightException(RolloutInsightException.INVALID_ARGUMENT, f"attempt #{attempt_pk} 不存在")
    artifacts_dir, entries = _artifact_entries(task.rollout_root or "")
    links = link_attempts_to_entries(rows, entries)
    return {
        "task": task,
        "row": row,
        "rows": rows,
        "artifacts_dir": artifacts_dir,
        "linked": links.get(int(attempt_pk)),
    }


def _phase_times(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = summary.get("phase_times") if isinstance(summary.get("phase_times"), dict) else {}
    out = []
    for key, val in raw.items():
        if not isinstance(val, (int, float)):
            continue
        out.append({"name": str(key).replace("phase_time/", ""), "seconds": round(float(val), 2)})
    return out


def _evaluation_view(summary: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    ev = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else {}
    return {
        "grader": ev.get("grader"),
        "exit_code": ev.get("exit_code"),
        "timed_out": ev.get("timed_out"),
        "stdout_tail": strip_ansi(ev.get("stdout_tail"))[-_STDOUT_MAX:],
        "stderr_tail": strip_ansi(ev.get("stderr_tail"))[-_STDOUT_MAX:],
        "failed_test": entry.get("failed_test"),
        "error_summary": entry.get("error_summary"),
    }


_CONCLUSION_FIELDS = (
    "status", "diagnosis", "abort_reason", "agent_outcome", "agent_exit_code", "applied",
    "reward", "raw_reward", "shaped_reward", "agent_turns", "agent_tool_calls",
    "agent_tool_errors", "attempt_peak_context_tokens", "attempt_generated_response_tokens",
    "diff_line_count", "segments", "recorded_at", "session_id",
)


def _pt_fallback_detail(pt_path: str, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """无 artifacts 时从 .pt 抽该 attempt 的失败细节（只取首个 segment）。"""
    from utils.failure_insight import extract_sample_failure_detail
    from utils.rollout_pt_parser import load_pt

    if not pt_path or not os.path.isfile(pt_path):
        return None
    try:
        dump = load_pt(pt_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("回退读取 pt 失败 pt={} err={}", pt_path, exc)
        return None
    for i, sample in enumerate(dump.get("samples") or []):
        if not isinstance(sample, dict) or sample.get("group_index") != row.get("group_index"):
            continue
        aid = sample.get("rollout_id")
        if (aid if aid is not None else i) == row.get("attempt_id"):
            return extract_sample_failure_detail(sample, sample_index=i)
    return None


def _group_siblings(rows: List[Dict[str, Any]], group_index: Any) -> List[Dict[str, Any]]:
    return [
        {"attempt_pk": r["attempt_pk"], "attempt_id": r["attempt_id"], "category": r["category"]}
        for r in rows
        if r.get("group_index") == group_index
    ]


def _segments_of(attempt_pk: int) -> List[Dict[str, Any]]:
    from db.db_segment_service import DBSegmentService

    return [
        {"primary_id": int(s.primary_id), "segment_index": s.segment_index, "status": s.status, "reward": s.reward}
        for s in DBSegmentService.list_by_attempt(int(attempt_pk))
    ]


def get_attempt_failure_detail(attempt_pk: int) -> Dict[str, Any]:
    """单个 attempt 的「怎么失败的」：结论 / 评测证据 / 耗时 / patch / 轨迹入口 / 回跳链接。"""
    from db.db_rollout_service import DBRolloutService

    ctx = _load_attempt_context(attempt_pk)
    task, row, artifacts_dir, linked = ctx["task"], ctx["row"], ctx["artifacts_dir"], ctx["linked"]
    pk_of, pt_of = _rollout_maps(int(task.primary_id))
    rollout_id = int(row["rollout_id"])
    rollout_pk = pk_of.get(rollout_id)

    entry, match = linked if linked else ({}, None)
    summary: Dict[str, Any] = {}
    patch = {"exists": False, "text": "", "truncated": False, "n_chars": 0}
    trajectory = {"exists": False, "kind": None}
    if entry and artifacts_dir is not None:
        summary = load_summary(artifacts_dir, entry["stem"]) or {}
        patch = load_patch(artifacts_dir, entry["stem"])
        trajectory = {"exists": bool(find_trajectory_path(artifacts_dir, entry["stem"])), "kind": entry.get("trajectory_kind")}

    item = _attempt_item(row, linked, None, rollout_pk)
    fallback = None if entry else _pt_fallback_detail(pt_of.get(rollout_id, ""), row)
    if fallback and item["reason"].startswith("status:"):
        item["reason"] = fallback.get("reason") or item["reason"]

    rollout_obj = DBRolloutService.get_by_id(rollout_pk) if rollout_pk else None
    return {
        "task": _task_brief(task),
        "rollout": {"rollout_pk": rollout_pk, "rollout_id": rollout_id,
                    "pt_path": getattr(rollout_obj, "pt_path", "") or ""},
        "attempt": item,
        "stem": entry.get("stem") if entry else None,
        "match": match,
        "source": "attempt_artifacts" if entry else ("pt" if fallback else "db"),
        "conclusion": {k: summary.get(k) for k in _CONCLUSION_FIELDS if k in summary},
        "evaluation": _evaluation_view(summary, entry) if summary else None,
        "phase_times": _phase_times(summary),
        "patch": patch,
        "trajectory": trajectory,
        "pt_fallback": fallback,
        "siblings": _group_siblings(ctx["rows"], row.get("group_index")),
        "segments": _segments_of(attempt_pk),
        "accordion_url": (
            f"/api/tasks/{int(task.primary_id)}/rollouts/{rollout_pk}?group={row.get('group_index')}#attempt-{int(attempt_pk)}"
            if rollout_pk else None
        ),
    }


def get_attempt_trajectory(
    attempt_pk: int, offset: int = 0, limit: int = 200, errors_only: bool = False
) -> Dict[str, Any]:
    """分页返回 attempt 对应的 OpenHands 轨迹事件。"""
    ctx = _load_attempt_context(attempt_pk)
    artifacts_dir, linked = ctx["artifacts_dir"], ctx["linked"]
    if artifacts_dir is None or not linked:
        raise RolloutInsightException(
            RolloutInsightException.PARSE_DUMP_NOT_FOUND, f"attempt #{attempt_pk} 未关联到 attempt_artifacts"
        )
    path = find_trajectory_path(artifacts_dir, linked[0]["stem"])
    if path is None:
        raise RolloutInsightException(
            RolloutInsightException.PARSE_DUMP_NOT_FOUND, f"attempt #{attempt_pk} 无轨迹文件（通过样本通常不落盘）"
        )
    payload = load_trajectory_events(path, offset=offset, limit=limit, errors_only=errors_only)
    payload["path"] = str(path)
    payload["attempt_pk"] = int(attempt_pk)
    return payload
