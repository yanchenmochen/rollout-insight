#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""多任务曲线对比：把多个任务的同名指标画在同一张图里。

与「单任务训练曲线」（``service/train_curve_service.py``）共用同一套指标目录
（``utils/train_curve_catalog.py``）与 epoch 聚合口径（``utils/train_curve_epoch.py``），
因此两边看到的是**同样的图、同样的按钮组**，只是这里一次可以勾选多个任务叠加。

数据来源仍是各任务 ``rollout_root`` 下的 TB 事件目录（``tbs/`` 或 ``tf_logs/``）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from utils.train_curve_catalog import (
    CURVE_METRICS,
    DEFAULT_CURVE_KEYS,
    available_curve_metrics,
    is_custom_key,
    resolve_metric_tags,
)

__all__ = [
    "list_comparable_tasks",
    "list_curve_compare_brief",
    "build_curve_compare",
]


def _series_label(tag: str) -> str:
    """把 tag 变成人能读懂的短标签（去掉公共前缀）。"""
    text = str(tag or "").strip()
    for prefix in ("rollout/", "train/", "perf/", "actor/", "critic/"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def list_comparable_tasks() -> List[Dict[str, Any]]:
    """列出可用于曲线对比的任务（含事件目录）。

    这里**不读**事件目录里的 scalar tag（读 tag 要加载事件文件，开销大），
    只解析 run 目录；真正的 tag 在 ``build_curve_compare`` 里按需读取。

    Returns:
        ``[{"task_id", "task_name", "event_dir"}]``；没有事件目录的任务不入列。
    """
    from db.db_task_service import DBTaskService
    from service.train_curve_service import resolve_event_dir

    tasks = DBTaskService.list_tasks(limit=5000, offset=0)
    out: List[Dict[str, Any]] = []
    for task in tasks:
        task_id = int(getattr(task, "primary_id", 0) or 0)
        if task_id <= 0:
            continue
        rollout_root = str(getattr(task, "rollout_root", "") or "")
        event_dir = resolve_event_dir(rollout_root)
        if not event_dir:
            continue
        out.append(
            {
                "task_id": task_id,
                "task_name": str(getattr(task, "task_name", "") or ""),
                "event_dir": event_dir,
            }
        )
    return out


def list_curve_compare_brief() -> Dict[str, Any]:
    """只返回任务清单与指标目录，**不读**事件目录。

    页面首屏用它渲染勾选框；真正画曲线时再按选中的任务去读事件文件。
    """
    from utils.train_curve_catalog import (
        DEFAULT_CURVE_KEYS,
        available_curve_metrics,
    )

    tasks = list_comparable_tasks()
    return {
        "tasks": tasks,
        "available_metrics": available_curve_metrics(),
        "default_keys": list(DEFAULT_CURVE_KEYS),
    }


def _task_metrics(
    task: Dict[str, Any],
    keys: Sequence[str],
    series_by_tag: Dict[str, List[Dict[str, Any]]],
    steps_per_epoch: Optional[int],
    tags: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """把一个任务的各指标组装成「子曲线」列表。

    一个指标可能命中多个 tag（例如「奖励」= generated / paired / filtered），
    这些 tag 会作为**同图内的多条子曲线**返回。
    """
    from service.train_curve_service import _series_view

    available = list(tags if tags is not None else series_by_tag.keys())
    metrics: List[Dict[str, Any]] = []
    for key in keys:
        matched = resolve_metric_tags(key, available)
        if not matched:
            continue
        meta = CURVE_METRICS.get(key)
        label = (
            meta["label"]
            if meta
            else (key.split(":", 1)[-1] if is_custom_key(key) else key)
        )
        sub_series: List[Dict[str, Any]] = []
        for tag in matched:
            view = _series_view(tag, series_by_tag.get(tag) or [], steps_per_epoch)
            if view["n_points"] or view["n_epoch_points"]:
                sub_series.append(view)
        if not sub_series:
            continue
        metrics.append(
            {
                "key": key,
                "label": label,
                "tags": matched,
                "sub_series": sub_series,
            }
        )
    return metrics


def build_curve_compare(
    task_ids: Optional[Sequence[int]] = None,
    keys: Optional[Sequence[str]] = None,
    view: str = "step",
) -> Dict[str, Any]:
    """组装多任务曲线对比页数据。

    Args:
        task_ids: 要对比的任务；``None`` 表示全部（最多 12 个，避免一次读太多事件目录）。
        keys: 要展示的指标 key；``None`` 用内置默认按钮组。
        view: ``step``（逐 step）或 ``epoch``（epoch 内取平均）。

    Returns:
        ``{"tasks": [...], "metrics": [...], "view": str, "available_metrics": [...]}``
    """
    from service.train_curve_service import resolve_event_dir
    from utils.tb_series_reader import read_scalar_series
    from utils.train_curve_epoch import epoch_view

    comparable = list_comparable_tasks()
    if task_ids:
        wanted = {int(t) for t in task_ids if int(t) > 0}
        comparable = [t for t in comparable if int(t["task_id"]) in wanted]
    else:
        comparable = comparable[:12]

    metric_keys = list(keys or DEFAULT_CURVE_KEYS)
    use_epoch = str(view or "step").lower() == "epoch"

    task_payloads: List[Dict[str, Any]] = []
    metric_map: Dict[str, Dict[str, Any]] = {}

    for task in comparable:
        event_dir = str(task.get("event_dir") or "")
        series_by_tag = read_scalar_series(event_dir)
        tags = list(series_by_tag.keys())
        steps_per_epoch = _resolve_steps_per_epoch(int(task["task_id"]))

        task_metrics = _task_metrics(task, metric_keys, series_by_tag, steps_per_epoch, tags)
        payload = {
            "task_id": int(task["task_id"]),
            "task_name": str(task.get("task_name") or ""),
            "event_dir": event_dir,
            "metrics": task_metrics,
        }
        task_payloads.append(payload)

        # 汇总所有任务命中该指标的子曲线
        for metric in task_metrics:
            entry = metric_map.setdefault(
                metric["key"],
                {"key": metric["key"], "label": metric["label"], "series": []},
            )
            for sub in metric["sub_series"]:
                points = sub["epoch_points"] if use_epoch else sub["points"]
                if not points:
                    continue
                entry["series"].append(
                    {
                        "task_id": payload["task_id"],
                        "task_name": payload["task_name"],
                        "tag": sub["tag"],
                        "label": sub["label"] or _series_label(sub["tag"]),
                        "points": points,
                        "n_points": len(points),
                    }
                )

    metrics: List[Dict[str, Any]] = []
    for key in metric_keys:
        entry = metric_map.get(key)
        if not entry or not entry["series"]:
            continue
        metrics.append(entry)

    logger.info(
        "多任务曲线对比 tasks={} metrics={} view={}",
        len(task_payloads),
        len(metrics),
        "epoch" if use_epoch else "step",
    )
    return {
        "tasks": task_payloads,
        "metrics": metrics,
        "view": "epoch" if use_epoch else "step",
        "available_metrics": available_curve_metrics(),
        "default_keys": list(DEFAULT_CURVE_KEYS),
    }


def _resolve_steps_per_epoch(task_id: int) -> Optional[int]:
    """算出该任务的 epoch 边界（口径与单任务训练曲线一致）。"""
    from service.problem_analysis_service import build_pass_trend
    from utils.train_curve_epoch import resolve_steps_per_epoch

    try:
        trend = build_pass_trend(int(task_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("计算 epoch 边界失败 task_id={} err={}", task_id, exc)
        return None
    if not trend.get("steps_per_epoch_exact"):
        return None
    return resolve_steps_per_epoch(trend.get("n_problems"), trend.get("batch_size"))
