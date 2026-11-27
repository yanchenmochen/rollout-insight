#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""任务环境变量：单任务快照 + 多任务对比。

复用 ``task_job_context_service.load_merged_task_env``（本地 resolved_config + 南湖），
再经 ``utils.task_env_catalog`` 做关键键摘要与差异。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from utils.task_env_catalog import compare_env_maps, pick_highlight_env, redact_env_value, summarize_env_map

__all__ = [
    "load_task_env_snapshot",
    "build_task_env_view",
    "build_task_env_compare",
]


def load_task_env_snapshot(
    task_id: int,
    *,
    local_user_id: Optional[int] = None,
    fetch_nanhu_env: bool = False,
    include_full: bool = False,
) -> Optional[Dict[str, Any]]:
    """按 task_id 读取环境变量快照；任务不存在返回 None。"""
    from db.db_task_service import DBTaskService
    from service.task_job_context_service import load_merged_task_env

    task = DBTaskService.get_by_id(int(task_id))
    if task is None:
        return None

    meta = load_merged_task_env(
        rollout_root=getattr(task, "rollout_root", None),
        nanhu_job_id=getattr(task, "nanhu_job_id", None),
        local_user_id=local_user_id,
        fetch_nanhu_env=fetch_nanhu_env,
    )
    env: Dict[str, str] = meta.get("env") or {}
    summary = summarize_env_map(env)

    out: Dict[str, Any] = {
        "task_id": int(task.primary_id),
        "task_name": str(getattr(task, "task_name", "") or ""),
        "rollout_root": str(getattr(task, "rollout_root", "") or ""),
        "nanhu_job_id": str(getattr(task, "nanhu_job_id", "") or ""),
        "run_dir": meta.get("run_dir"),
        "env_source": meta.get("env_source") or "",
        "launcher": meta.get("launcher") or "",
        "job_type": meta.get("job_type") or "unknown",
        "job_type_label": meta.get("job_type_label") or "未知",
        "training_script": meta.get("training_script") or "",
        "env_count": summary["env_count"],
        "highlight_count": summary["highlight_count"],
        "highlight": summary["highlight"],
        "highlight_catalog": summary["highlight_catalog"],
        "by_category": summary["by_category"],
        "_raw_env": env,  # 仅进程内对比用，API 层会剥掉
    }
    if include_full:
        out["env"] = {
            k: redact_env_value(k, v)
            for k, v in sorted(env.items(), key=lambda kv: kv[0].upper())
        }
        out["env_all_rows"] = pick_highlight_env(env, include_unknown=True)
    else:
        out["env"] = None
    return out


def build_task_env_view(
    task_id: int,
    *,
    local_user_id: Optional[int] = None,
    fetch_nanhu_env: bool = True,
    include_full: bool = False,
) -> Dict[str, Any]:
    """单任务环境变量视图（给详情页 / JSON API）。"""
    snap = load_task_env_snapshot(
        int(task_id),
        local_user_id=local_user_id,
        fetch_nanhu_env=fetch_nanhu_env,
        include_full=include_full,
    )
    if snap is None:
        from data.rollout_insight_exception import RolloutInsightException

        raise RolloutInsightException(RolloutInsightException.TASK_NOT_FOUND)
    public = {k: v for k, v in snap.items() if not str(k).startswith("_")}
    return public


def build_task_env_compare(
    task_ids: Sequence[int],
    *,
    baseline_task_id: Optional[int] = None,
    local_user_id: Optional[int] = None,
    fetch_nanhu_env: bool = False,
    only_highlight: bool = True,
    only_diff: bool = False,
) -> Dict[str, Any]:
    """多任务环境变量对比。

    - 默认只比目录内关键键（``only_highlight=True``）
    - 默认不拉南湖（本地 resolved_config 足够；加速）
    - ``only_diff=True`` 时 rows 只保留有差异的键
    """
    ids: List[int] = []
    for raw in task_ids or []:
        try:
            tid = int(raw)
        except (TypeError, ValueError):
            continue
        if tid > 0 and tid not in ids:
            ids.append(tid)
    if not ids:
        return {
            "ok": False,
            "error": "请至少选择 1 个任务",
            "tasks": [],
            "compare": compare_env_maps([]),
        }
    if len(ids) > 6:
        ids = ids[:6]

    snaps: List[Dict[str, Any]] = []
    envs: List[Dict[str, str]] = []
    labels: List[str] = []
    for tid in ids:
        snap = load_task_env_snapshot(
            tid,
            local_user_id=local_user_id,
            fetch_nanhu_env=fetch_nanhu_env,
            include_full=False,
        )
        if snap is None:
            logger.warning("环境变量对比跳过不存在的任务 task_id={}", tid)
            continue
        snaps.append(snap)
        envs.append(snap.get("_raw_env") or {})
        labels.append(f"#{snap['task_id']} {snap['task_name']}")

    if not snaps:
        return {
            "ok": False,
            "error": "所选任务均不存在或无法读取环境变量",
            "tasks": [],
            "compare": compare_env_maps([]),
        }

    base_idx = 0
    if baseline_task_id is not None:
        for i, snap in enumerate(snaps):
            if int(snap["task_id"]) == int(baseline_task_id):
                base_idx = i
                break

    compare = compare_env_maps(
        envs,
        labels=labels,
        baseline_index=base_idx,
        only_highlight=only_highlight,
    )
    if only_diff:
        compare = {
            **compare,
            "rows": list(compare.get("diff_rows") or []),
        }

    notes: List[str] = []
    missing_source = [
        s
        for s in snaps
        if not (s.get("env_source") or "").strip() and int(s.get("env_count") or 0) == 0
    ]
    if missing_source:
        names = ", ".join(f"#{s['task_id']}" for s in missing_source)
        notes.append(
            f"以下任务未读到 resolved_config.yaml / 南湖 env：{names}。"
            "请确认 rollout_root 指向含 generated 的 run 目录。"
        )
    if len(snaps) == 1:
        notes.append("只选了 1 个任务：展示其关键环境变量；再勾选任务可看差异。")

    return {
        "ok": True,
        "error": "",
        "tasks": [
            {
                "task_id": s["task_id"],
                "task_name": s["task_name"],
                "env_source": s.get("env_source") or "",
                "env_count": s.get("env_count") or 0,
                "highlight_count": s.get("highlight_count") or 0,
                "job_type": s.get("job_type"),
                "job_type_label": s.get("job_type_label"),
                "run_dir": s.get("run_dir"),
            }
            for s in snaps
        ],
        "baseline_task_id": snaps[base_idx]["task_id"],
        "compare": compare,
        "notes": notes,
        "only_highlight": only_highlight,
        "only_diff": only_diff,
    }
