#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""训练曲线页数据：把任务的 TB 事件目录组装成「指标按钮组 + 折线」。

页面逻辑：
1. 读出事件目录下全部 scalar tag；
2. 按用户配置（全局默认 + 任务级追加）解析出要展示的指标；
3. 每个指标命中若干 tag 时，取第一个作为主曲线，其余作为同指标备选；
4. 输出可直接画折线的 ``series``，以及未命中任何 tag 的指标（前端提示"无数据"）。

偏好读取与写入都走 ``DBTrainCurvePreferenceService``（用户默认 + 任务级追加）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from utils.train_curve_catalog import (
    CURVE_METRICS,
    DEFAULT_CURVE_KEYS,
    available_curve_metrics,
    is_custom_key,
    normalize_curve_keys,
    resolve_metric_tags,
)

__all__ = [
    "resolve_event_dir",
    "resolve_curve_keys",
    "build_train_curve",
    "save_curve_preference",
]


def resolve_event_dir(rollout_root: Optional[str]) -> Optional[str]:
    """从 ``task.rollout_root`` 解析 TB 事件目录；没有返回 None。"""
    from utils.tb_run_paths import discover_tb_event_dir, resolve_run_dir

    run_dir = resolve_run_dir(rollout_root)
    if run_dir is None:
        return None
    event_dir = discover_tb_event_dir(run_dir)
    return str(event_dir) if event_dir else None


def resolve_curve_keys(
    user_id: Optional[int],
    task_id: int,
) -> Dict[str, Any]:
    """解析某用户对某任务最终生效的指标列表。

    合并顺序：内置默认按钮组 → 用户全局偏好 → 任务级追加。
    用户全局偏好一旦保存，就完全覆盖内置默认按钮组。
    """
    from db.db_train_curve_preference_service import DBTrainCurvePreferenceService

    global_keys: List[str] = list(DEFAULT_CURVE_KEYS)
    global_custom: List[str] = []
    global_saved = False

    if user_id is not None:
        try:
            row = DBTrainCurvePreferenceService.get_global(int(user_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("读训练曲线全局偏好失败 user_id={} err={}", user_id, exc)
            row = None
        if row is not None:
            global_saved = True
            payload = row.to_dict()
            global_keys = normalize_curve_keys(payload.get("keys"))
            global_custom = normalize_curve_keys(payload.get("custom_keys"))

    task_keys: List[str] = []
    task_custom: List[str] = []
    if user_id is not None:
        try:
            row = DBTrainCurvePreferenceService.get_for_task(int(user_id), int(task_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "读训练曲线任务偏好失败 user_id={} task_id={} err={}",
                user_id,
                task_id,
                exc,
            )
            row = None
        if row is not None:
            payload = row.to_dict()
            task_keys = normalize_curve_keys(payload.get("keys"))
            task_custom = normalize_curve_keys(payload.get("custom_keys"))

    merged: List[str] = []
    for key in list(global_keys) + list(global_custom) + list(task_keys) + list(task_custom):
        if key not in merged:
            merged.append(key)

    return {
        "keys": merged,
        "global_keys": global_keys,
        "global_custom": global_custom,
        "task_keys": task_keys,
        "task_custom": task_custom,
        "global_saved": global_saved,
        "has_task_override": bool(task_keys or task_custom),
    }


def _metric_view(
    key: str,
    tags: Sequence[str],
    series: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """把一个指标 key 组装成前端可用的视图对象。"""
    matched = resolve_metric_tags(key, list(tags))
    meta = CURVE_METRICS.get(key)
    label = meta["label"] if meta else (key.split(":", 1)[-1] if is_custom_key(key) else key)
    desc = meta["desc"] if meta else "自定义指标：直接按 TB tag 名匹配。"

    points: List[Dict[str, Any]] = []
    primary_tag = matched[0] if matched else ""
    if primary_tag:
        points = list(series.get(primary_tag) or [])

    return {
        "key": key,
        "label": label,
        "desc": desc,
        "is_custom": is_custom_key(key),
        "tags": matched,
        "primary_tag": primary_tag,
        "points": points,
        "n_points": len(points),
        "has_data": bool(points),
    }


def build_train_curve(task_id: int, user_id: Optional[int] = None) -> Dict[str, Any]:
    """组装任务训练曲线页 / JSON 载荷。"""
    from data.rollout_insight_exception import RolloutInsightException
    from db.db_task_service import DBTaskService
    from utils.tb_series_reader import list_scalar_tags, read_scalar_series

    task = DBTaskService.get_by_id(int(task_id))
    if task is None:
        raise RolloutInsightException(
            RolloutInsightException.TASK_NOT_FOUND, f"task_id={task_id} 不存在"
        )

    rollout_root = str(getattr(task, "rollout_root", "") or "")
    event_dir = resolve_event_dir(rollout_root)

    preference = resolve_curve_keys(user_id, int(task_id))
    keys: List[str] = preference["keys"]

    tags: List[str] = []
    series: Dict[str, List[Dict[str, Any]]] = {}
    notice = ""
    if event_dir:
        tags = list_scalar_tags(event_dir)
        series = read_scalar_series(event_dir)
    else:
        notice = (
            "该任务的 rollout_root 下没有找到 tbs/ 或 tf_logs/ 事件目录，"
            "因此没有可画的训练曲线。"
        )

    metrics = [_metric_view(key, tags, series) for key in keys]

    logger.info(
        "训练曲线 task_id={} event_dir={} tags={} metrics={}",
        task_id,
        event_dir,
        len(tags),
        len(metrics),
    )
    return {
        "task": {
            "id": int(task.primary_id),
            "task_name": getattr(task, "task_name", None),
            "status": getattr(task, "status", None),
            "rollout_root": rollout_root,
        },
        "event_dir": event_dir,
        "notice": notice,
        "tags": tags,
        "metrics": metrics,
        "preference": preference,
        "available_metrics": available_curve_metrics(),
        "default_keys": list(DEFAULT_CURVE_KEYS),
        "page_note": (
            "指标按钮组来自「用户默认 + 本任务追加」，保存后长期生效；"
            "同一指标命中多个 tag 时取第一个作为主曲线。"
        ),
    }


def save_curve_preference(
    task_id: int,
    user_id: Optional[int],
    *,
    keys: Optional[List[str]] = None,
    custom_keys: Optional[List[str]] = None,
    scope: str = "global",
) -> Dict[str, Any]:
    """保存曲线指标偏好。

    ``scope='global'``：写用户默认按钮组（长期生效）。
    ``scope='task'``：写该任务的追加展示项。
    """
    from data.rollout_insight_exception import RolloutInsightException
    from db.db_train_curve_preference_service import DBTrainCurvePreferenceService

    if user_id is None:
        raise RolloutInsightException(
            RolloutInsightException.INVALID_ARGUMENT, "未登录，无法保存曲线配置"
        )

    normalized = normalize_curve_keys(keys)
    normalized_custom = normalize_curve_keys(custom_keys)
    scope_norm = (scope or "global").strip().lower()
    if scope_norm not in ("global", "task"):
        raise RolloutInsightException(
            RolloutInsightException.INVALID_ARGUMENT, "scope 须为 global 或 task"
        )

    if scope_norm == "task":
        from db.db_task_service import DBTaskService

        task = DBTaskService.get_by_id(int(task_id))
        if task is None:
            raise RolloutInsightException(
                RolloutInsightException.TASK_NOT_FOUND, f"task_id={task_id} 不存在"
            )
        DBTrainCurvePreferenceService.upsert_for_task(
            int(user_id),
            int(task_id),
            keys=normalized,
            custom_keys=normalized_custom,
        )
    else:
        DBTrainCurvePreferenceService.upsert_global(
            int(user_id),
            keys=normalized,
            custom_keys=normalized_custom,
        )

    logger.info(
        "保存训练曲线配置 user_id={} task_id={} scope={} keys={}",
        user_id,
        task_id,
        scope_norm,
        normalized,
    )
    return resolve_curve_keys(user_id, int(task_id))
