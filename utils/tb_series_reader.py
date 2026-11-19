#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""从 TensorBoard 事件目录读取 scalar 序列（多 tag）。

与 ``utils/eval_tb_scores``（只取每个 tag 最新一步）、``service/insight_page_service``
（同样只取最新一步）不同：这里保留每个 tag 的**完整 step→value 序列**，
供训练曲线页与跨任务曲线对比画折线。

读取参数对齐 TensorBoard 服务端启动参数（``--load_fast=false
--purge_orphaned_data=false``）：水库大小给足，且关闭 orphan purge，
避免训练重启后写的短事件文件把历史 step 截掉。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

__all__ = [
    "read_scalar_series",
    "list_scalar_tags",
    "series_points",
]

# 单个 tag 保留的最大点数；0 表示全部保留。
_DEFAULT_MAX_POINTS = 4000

# 交给 TensorBoard 的 scalar 水库容量。
# 必须远大于实际点数：TB 的 Reservoir 是**随机抽样**，一旦超过容量就会
# 随机丢点（连首步都可能丢），曲线会莫名其妙缺一段。真正的抽稀由本模块
# 的 ``_downsample`` 做确定性的等距抽样（保留首尾）。
_TB_RESERVOIR_SIZE = 20000


def _event_accumulator(path: Path):
    """构造 EventAccumulator；tensorboard 不可用时返回 None。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("无法导入 tensorboard，读取事件目录失败: {}", exc)
        return None

    guidance = {"scalars": _TB_RESERVOIR_SIZE}
    try:
        accumulator = _make_accumulator(EventAccumulator, str(path), guidance)
        accumulator.Reload()
    except Exception as exc:  # noqa: BLE001
        logger.warning("EventAccumulator 读取失败 path={} err={}", path, exc)
        return None
    return accumulator


def _make_accumulator(EventAccumulator, path: str, guidance: Dict[str, int]):
    """构造 EventAccumulator；老版本没有 ``purge_orphaned_data`` 时退化为默认参数。"""
    try:
        return EventAccumulator(path, size_guidance=guidance, purge_orphaned_data=False)
    except TypeError:
        return EventAccumulator(path, size_guidance=guidance)


def list_scalar_tags(event_dir: Optional[Path | str]) -> List[str]:
    """列出事件目录下的全部 scalar tag（已排序）。"""
    accumulator = _load_accumulator(event_dir)
    if accumulator is None:
        return []
    tags = (accumulator.Tags() or {}).get("scalars") or []
    return sorted({str(tag).strip() for tag in tags if str(tag or "").strip()})


def read_scalar_series(
    event_dir: Optional[Path | str],
    *,
    tags: Optional[Sequence[str]] = None,
    max_points: int = _DEFAULT_MAX_POINTS,
) -> Dict[str, List[Dict[str, Any]]]:
    """读取事件目录下若干 tag 的完整 scalar 序列。

    Args:
        event_dir: TensorBoard 事件目录（``tbs/`` 或 ``tf_logs/``）。
        tags: 需要读取的 tag；``None`` 表示读取全部 scalar tag。
        max_points: 单 tag 最多保留的点数（``0`` 表示不限）。抽稀是**确定性**
            等距抽样（保留首尾），不会随机丢点。

    Returns:
        ``{tag: [{"step": int, "value": float}, ...]}``；目录不存在或
        tensorboard 不可用时返回空 dict。读取失败的 tag 不会出现在结果里。
    """
    accumulator = _load_accumulator(event_dir)
    if accumulator is None:
        return {}

    available = (accumulator.Tags() or {}).get("scalars") or []
    wanted = [str(t).strip() for t in (tags or []) if str(t or "").strip()]
    if wanted:
        target = [t for t in wanted if t in set(available)]
    else:
        target = sorted({str(t).strip() for t in available if str(t or "").strip()})

    out: Dict[str, List[Dict[str, Any]]] = {}
    for tag in target:
        try:
            events = accumulator.Scalars(tag)
        except Exception:  # noqa: BLE001
            continue
        points: List[Dict[str, Any]] = []
        for event in events or []:
            try:
                value = float(event.value)
            except (TypeError, ValueError):
                continue
            try:
                step = int(getattr(event, "step", 0) or 0)
            except (TypeError, ValueError):
                step = 0
            points.append({"step": step, "value": value})
        points.sort(key=lambda item: item["step"])
        if max_points > 0 and len(points) > max_points:
            # 保留首尾，中间等距抽样，避免长训练把曲线抽稀到看不出趋势。
            points = _downsample(points, max_points)
        if points:
            out[tag] = points
    return out


def series_points(series: Dict[str, List[Dict[str, Any]]], tag: str) -> List[Dict[str, Any]]:
    """从序列字典里取某个 tag 的点列表（不存在返回空 list）。"""
    return list((series or {}).get(str(tag or ""), []) or [])


def _downsample(points: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """等距抽样到 ``limit`` 个点，保留首尾。"""
    if limit <= 0 or len(points) <= limit:
        return points
    step = (len(points) - 1) / float(limit - 1)
    picked: List[Dict[str, Any]] = []
    seen = set()
    for i in range(limit):
        index = int(round(i * step))
        if index in seen:
            continue
        seen.add(index)
        picked.append(points[index])
    if picked and picked[-1] is not points[-1]:
        picked[-1] = points[-1]
    return picked


def _load_accumulator(event_dir: Optional[Path | str]):
    """校验目录并构造 EventAccumulator。"""
    if not event_dir:
        return None
    try:
        path = Path(event_dir)
    except (TypeError, ValueError):
        return None
    if not path.is_dir():
        return None
    return _event_accumulator(path)
