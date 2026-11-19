#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""训练曲线的 epoch 聚合：把逐 step 的曲线折成「epoch 内取平均」。

背景：
- TensorBoard 的横轴是 **step**（一次 rollout 采样 + 一次训练更新），粒度很细，
  单 step 的值抖动大，看整体趋势要靠人眼扫。
- 与「问题分析」页的 epoch 口径保持一致：一个 epoch = 该题第 k 次出现，
  对应 rollout 序列上**连续 steps_per_epoch 个 rollout**（由题目池大小除以
  每轮抽样题数算出），也就是 TB 曲线上连续 steps_per_epoch 个 step。

因此本模块只做一件事：给定 step 序列与每个 epoch 覆盖多少 step，把曲线重采样成
「每个 epoch 内所有 step 取平均」的一条曲线。

step 与 rollout 的对应：
- slime 每个 rollout step 都会往 TB 写一次 scalar，step 序号即 rollout 序号
  （rollout_id 从 0 开始连续）。
- TB 的 step 与本项目的 rollout_id 是同一套编号，因此可以直接把
  rollout_id 映射成 step 下标来定位 epoch 边界。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "resolve_steps_per_epoch",
    "aggregate_by_epoch",
    "epoch_view",
]


def resolve_steps_per_epoch(
    n_problems: Optional[int],
    batch_size: Optional[int],
) -> Optional[int]:
    """算出「一个 epoch 覆盖多少个 rollout step」。

    Args:
        n_problems: 题目池大小（该任务一共出现过的题目数）。
        batch_size: 每轮 rollout 抽多少题（ROLLOUT_BATCH_SIZE）。

    Returns:
        每个 epoch 的 step 数；无法确定时返回 None。
    """
    if not n_problems or not batch_size:
        return None
    if int(n_problems) <= 0 or int(batch_size) <= 0:
        return None
    steps = int(n_problems) / float(int(batch_size))
    rounded = int(round(steps))
    if rounded <= 0:
        return None
    # 只在整除（误差可忽略）时给出口径明确的 epoch 边界
    if abs(steps - rounded) > 1e-6:
        return None
    return rounded


def aggregate_by_epoch(
    points: Sequence[Dict[str, Any]],
    steps_per_epoch: Optional[int],
    *,
    max_epochs: int = 0,
) -> List[Dict[str, Any]]:
    """把逐 step 的点按 epoch 聚合（epoch 内所有 step 取平均）。

    Args:
        points: [{"step": int, "value": float}, ...]（按 step 升序）。
        steps_per_epoch: 每个 epoch 覆盖多少个 step；None 或小于等于 0 时原样返回。
        max_epochs: 只保留最后 N 个 epoch（0 表示全部）。

    Returns:
        [{"epoch": int, "step": int, "value": float, "n_steps": int}, ...]
        —— step 取该 epoch 最后一个 step 的序号（作为横坐标锚点）。
    """
    if not points:
        return []
    span = int(steps_per_epoch or 0)
    if span <= 0:
        return []

    ordered = sorted(points, key=lambda item: int(item.get("step", 0) or 0))
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for item in ordered:
        step = int(item.get("step", 0) or 0)
        epoch = step // span + 1
        buckets.setdefault(epoch, []).append(item)

    out: List[Dict[str, Any]] = []
    for epoch in sorted(buckets):
        group = buckets[epoch]
        values = [
            float(item["value"])
            for item in group
            if isinstance(item.get("value"), (int, float))
        ]
        if not values:
            continue
        out.append(
            {
                "epoch": epoch,
                "step": max(int(item.get("step", 0) or 0) for item in group),
                "value": sum(values) / len(values),
                "n_steps": len(group),
            }
        )

    if max_epochs > 0 and len(out) > max_epochs:
        out = out[-max_epochs:]
    return out


def epoch_view(
    points: Sequence[Dict[str, Any]],
    steps_per_epoch: Optional[int],
) -> Dict[str, Any]:
    """聚合出 epoch 曲线，并附带口径说明。

    Returns:
        {"points": [...], "steps_per_epoch": int|None, "available": bool,
        "note": str}
    """
    span = int(steps_per_epoch or 0)
    if span <= 0:
        return {
            "points": [],
            "steps_per_epoch": None,
            "available": False,
            "note": (
                "无法确定 epoch 边界（缺题目池大小或每轮抽样题数），"
                "因此只展示逐 step 曲线。"
            ),
        }
    aggregated = aggregate_by_epoch(points, span)
    return {
        "points": aggregated,
        "steps_per_epoch": span,
        "available": bool(aggregated),
        "note": (
            "epoch 曲线 = 每个 epoch 内所有 step 取平均；"
            "一个 epoch 覆盖 %d 个 rollout step"
            "（题目池大小除以每轮抽样题数，与问题分析页同口径）。" % span
        ),
    }
