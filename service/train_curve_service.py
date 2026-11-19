#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""训练曲线页数据：把任务的 TB 事件目录组装成「指标按钮组 + 折线」。

页面逻辑：
1. 读出事件目录下全部 scalar tag；
2. 按用户配置（全局默认 + 任务级追加）解析出要展示的指标；
3. 每个指标可以命中多个 tag（例如「奖励」= generated / paired / filtered 三条
   raw_reward），这些 tag **画在同一张图**里，用不同颜色区分；
4. 除逐 step 曲线外，还给出 epoch 聚合曲线（epoch 内所有 step 取平均），
   口径与「问题分析」页一致（题目池大小 ÷ 每轮抽样题数 = 每 epoch 的 step 数）；
5. 输出可直接画折线的 ``series``，以及未命中任何 tag 的指标（前端提示"无数据"）。

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
from utils.train_curve_epoch import epoch_view, resolve_steps_per_epoch

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


def _series_view(
    tag: str,
    points: Sequence[Dict[str, Any]],
    steps_per_epoch: Optional[int],
) -> Dict[str, Any]:
    """单条子曲线：逐 step 点 + epoch 聚合点。"""
    step_points = [
        {"step": int(item.get("step", 0) or 0), "value": float(item.get("value", 0.0) or 0.0)}
        for item in points or []
    ]
    epoch = epoch_view(step_points, steps_per_epoch)
    return {
        "tag": tag,
        "label": _series_label(tag),
        "points": step_points,
        "n_points": len(step_points),
        "epoch_points": epoch["points"],
        "n_epoch_points": len(epoch["points"]),
    }


def _series_label(tag: str) -> str:
    """把 tag 变成人能读懂的短标签（去掉公共前缀）。"""
    text = str(tag or "").strip()
    for prefix in ("rollout/", "train/", "perf/", "actor/", "critic/"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _metric_view(
    key: str,
    tags: Sequence[str],
    series: Dict[str, List[Dict[str, Any]]],
    steps_per_epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """把一个指标 key 组装成前端可用的视图对象。

    一个指标可能命中多个 tag（例如「奖励」命中 generated / paired / filtered
    三条 raw_reward），这些子曲线放在 ``sub_series`` 里，**画在同一张图**。
    """
    matched = resolve_metric_tags(key, list(tags))
    meta = CURVE_METRICS.get(key)
    label = meta["label"] if meta else (key.split(":", 1)[-1] if is_custom_key(key) else key)
    desc = meta["desc"] if meta else "自定义指标：直接按 TB tag 名匹配。"

    sub_series: List[Dict[str, Any]] = []
    for tag in matched:
        view = _series_view(tag, series.get(tag) or [], steps_per_epoch)
        if view["n_points"] or view["n_epoch_points"]:
            sub_series.append(view)

    primary_tag = matched[0] if matched else ""
    primary_points = list(series.get(primary_tag) or []) if primary_tag else []

    return {
        "key": key,
        "label": label,
        "desc": desc,
        "is_custom": is_custom_key(key),
        "tags": matched,
        "primary_tag": primary_tag,
        # 主曲线：逐 step（保持旧字段，前端可只画一条时用）
        "points": primary_points,
        "n_points": len(primary_points),
        # 多条子曲线：同一张图内叠加
        "sub_series": sub_series,
        "n_series": len(sub_series),
        "has_data": bool(sub_series),
    }


def _resolve_steps_per_epoch(task_id: int) -> Optional[int]:
    """算出一个 epoch 覆盖多少个 rollout step（口径与问题分析页一致）。

    复用 ``service.problem_analysis_service.build_pass_trend``：它已经按
    「题目池大小 ÷ 每轮抽样题数」算好了 ``steps_per_epoch``，并标注了是否整除。
    拿不到（缺题目池大小或每轮抽样题数）就返回 None，页面退化成只展示逐 step 曲线。
    """
    from service.problem_analysis_service import build_pass_trend

    try:
        trend = build_pass_trend(int(task_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("计算 epoch 边界失败 task_id={} err={}", task_id, exc)
        return None

    if not trend.get("steps_per_epoch_exact"):
        return None
    span = trend.get("steps_per_epoch")
    if not span:
        return None
    return resolve_steps_per_epoch(trend.get("n_problems"), trend.get("batch_size"))


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

    steps_per_epoch = _resolve_steps_per_epoch(int(task_id))
    metrics = [
        _metric_view(key, tags, series, steps_per_epoch) for key in keys
    ]

    logger.info(
        "训练曲线 task_id={} event_dir={} tags={} metrics={} steps_per_epoch={}",
        task_id,
        event_dir,
        len(tags),
        len(metrics),
        steps_per_epoch,
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
        "epoch": {
            "steps_per_epoch": steps_per_epoch,
            "available": bool(steps_per_epoch),
            "note": (
                "epoch 视图 = 每个 epoch 内所有 step 取平均；"
                "一个 epoch 覆盖 %s 个 rollout step。"
                % (steps_per_epoch if steps_per_epoch else "?")
                if steps_per_epoch
                else "无法确定 epoch 边界（缺题目池大小或每轮抽样题数），只展示逐 step 曲线。"
            ),
        },
        "page_note": (
            "指标按钮组来自「用户默认 + 本任务追加」，保存后长期生效；"
            "同一指标命中多个 tag 时会在同一张图里画多条子曲线。"
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
