"""从 TensorBoard 事件目录读取评测主分数（``eval/<dataset>``）。

对齐 slime ``_log_eval_rollout_data``：主 tag 为 ``eval/{key}``，
不含子路径与 ``-truncated_ratio`` 等后缀。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.tb_run_paths import discover_tb_event_dir, resolve_run_dir

__all__ = [
    "is_primary_eval_tag",
    "read_eval_scores_from_event_dir",
    "read_eval_scores_for_rollout_root",
]

_PRIMARY_EVAL_TAG_RE = re.compile(r"^eval/([^/\-]+)$")


def is_primary_eval_tag(tag: str) -> bool:
    """是否为评测主分数 tag（排除子指标与 truncated_ratio 等）。"""
    return bool(_PRIMARY_EVAL_TAG_RE.match(str(tag or "").strip()))


def _dataset_name(tag: str) -> str:
    m = _PRIMARY_EVAL_TAG_RE.match(str(tag or "").strip())
    return m.group(1) if m else str(tag or "")


def read_eval_scores_from_event_dir(
    event_dir: Optional[Path | str],
) -> List[Dict[str, Any]]:
    """读取事件目录下全部评测主分数（取每个 tag 最新一步）。"""
    if not event_dir:
        return []
    try:
        path = Path(event_dir)
    except (TypeError, ValueError):
        return []
    if not path.is_dir():
        return []

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception:
        return []

    try:
        ea = EventAccumulator(str(path), size_guidance={"scalars": 100})
        ea.Reload()
    except Exception:
        return []

    tags = (ea.Tags() or {}).get("scalars") or []
    rows: List[Dict[str, Any]] = []
    for tag in tags:
        if not is_primary_eval_tag(tag):
            continue
        try:
            events = ea.Scalars(tag)
        except Exception:
            continue
        if not events:
            continue
        last = events[-1]
        rows.append(
            {
                "tag": tag,
                "name": _dataset_name(tag),
                "value": float(last.value),
                "step": int(getattr(last, "step", 0) or 0),
                "wall_time": float(getattr(last, "wall_time", 0.0) or 0.0),
            }
        )

    rows.sort(key=lambda r: str(r.get("name") or ""))
    return rows


def read_eval_scores_for_rollout_root(
    rollout_root: Optional[str],
) -> Dict[str, Any]:
    """从 ``task.rollout_root`` 解析 run → tbs/tf_logs → 评测分数。

    Returns:
        ``{"scores": [...], "event_dir": str|None, "run_dir": str|None}``
    """
    run_dir = resolve_run_dir(rollout_root)
    event_dir = discover_tb_event_dir(run_dir) if run_dir is not None else None
    scores = read_eval_scores_from_event_dir(event_dir) if event_dir else []
    return {
        "scores": scores,
        "event_dir": str(event_dir) if event_dir else None,
        "run_dir": str(run_dir) if run_dir else None,
    }
