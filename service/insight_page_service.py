"""洞察页数据：解析任务 run 目录，列出 scalar 的最新 step / value。

事件读取方式对齐 ``utils.eval_tb_scores`` 的 EventAccumulator；
run 目录解析复用 ``utils.tb_run_paths``（与 tensorboard_service 相同）。
没有事件目录或读取失败时返回说明文字，不向外抛给页面变成 500。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from utils.tb_deep_link import tag_filter_supported, tensorboard_tag_filter
from utils.tb_metric_catalog import classify_tb_tag, describe_tb_tag
from utils.tb_run_paths import discover_tb_event_dir, resolve_run_dir

__all__ = [
    "build_insight_page",
    "read_latest_scalars",
]

_GROUP_ORDER = ("primary", "truncated_ratio", "attempt_count", "unknown")

_PICKER_LIMIT = 200


def build_insight_page(task_id: Optional[int]) -> Dict[str, Any]:
    """组装洞察页上下文。``task_id`` 为空时只给任务选择列表。"""
    page: Dict[str, Any] = {
        "tasks": _list_picker_tasks(),
        "task": None,
        "task_missing": False,
        "notice": None,
        "run_dir": None,
        "event_dir": None,
        "scalar_groups": _empty_groups(),
        "scalar_count": 0,
        "tb_locates_tag": tag_filter_supported(),
    }
    if task_id is None:
        return page

    try:
        task_id_int = int(task_id)
    except (TypeError, ValueError):
        page["task_missing"] = True
        page["notice"] = "task_id 不是有效的任务编号。"
        return page

    task = _load_task(task_id_int)
    if task is None:
        page["task_missing"] = True
        page["notice"] = f"任务 #{task_id_int} 不存在。"
        return page

    page["task"] = task
    rollout_root = str(task.get("rollout_root") or "").strip()
    if not rollout_root:
        page["notice"] = "该任务没有 rollout_root，无法解析训练 run 目录，也没有可列出的 scalar。"
        return page

    run_dir = resolve_run_dir(rollout_root)
    page["run_dir"] = str(run_dir) if run_dir else None
    event_dir = discover_tb_event_dir(run_dir) if run_dir is not None else None
    if event_dir is None:
        where = f"（run 目录 {run_dir}）" if run_dir else ""
        page["notice"] = (
            "没有找到 tbs/ 或 tf_logs/ 事件目录"
            + where
            + "。本页不能列出 scalar，也不会因此报错。"
        )
        logger.info(
            "洞察页无事件目录 task_id={} rollout_root={} run_dir={}",
            task_id_int,
            rollout_root,
            run_dir,
        )
        return page

    page["event_dir"] = str(event_dir)
    rows, read_error = read_latest_scalars(event_dir)
    if read_error:
        page["notice"] = read_error
        logger.warning(
            "洞察页读取事件失败 task_id={} event_dir={} err={}",
            task_id_int,
            event_dir,
            read_error,
        )
        return page

    page["scalar_groups"] = _group_rows(rows)
    page["scalar_count"] = len(rows)
    if not rows:
        page["notice"] = (
            f"事件目录 {event_dir} 里没有直接存放的 scalar tag。"
            "本页与评测分数一样，只读这个目录本身，不展开子目录。"
        )
    logger.info(
        "洞察页 task_id={} event_dir={} scalars={}",
        task_id_int,
        event_dir,
        len(rows),
    )
    return page


def read_latest_scalars(
    event_dir: Optional[Path | str],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """读取事件目录下每个 scalar tag 的最新一步。

    返回 ``(rows, error)``。出错时 rows 为空，error 是给页面的说明。
    """
    if not event_dir:
        return [], "没有事件目录。"
    try:
        path = Path(event_dir)
    except (TypeError, ValueError):
        return [], "事件目录路径无效。"
    if not path.is_dir():
        return [], f"事件目录不存在：{path}"

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception:  # noqa: BLE001
        return [], "当前环境无法导入 tensorboard，不能读取事件文件。"

    try:
        # 与 eval_tb_scores 相同：水库保留最近样本，且默认始终留下最后一步。
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 100})
        accumulator.Reload()
    except Exception as exc:  # noqa: BLE001
        logger.warning("EventAccumulator 读取失败 path={} err={}", path, exc)
        return [], f"读取事件目录失败（{path}）。事件文件可能不完整，本页不展示数值。"

    tags = (accumulator.Tags() or {}).get("scalars") or []
    rows: List[Dict[str, Any]] = []
    for tag in tags:
        text = str(tag or "").strip()
        if not text:
            continue
        try:
            events = accumulator.Scalars(text)
        except Exception:  # noqa: BLE001
            continue
        if not events:
            continue
        last = events[-1]
        try:
            value = float(last.value)
        except (TypeError, ValueError):
            continue
        try:
            step = int(getattr(last, "step", 0) or 0)
        except (TypeError, ValueError):
            step = 0
        rows.append(_scalar_row(text, step, value))

    rows.sort(key=lambda row: str(row.get("tag") or ""))
    return rows, None


def _scalar_row(tag: str, step: int, value: float) -> Dict[str, Any]:
    return {
        "tag": tag,
        "kind": classify_tb_tag(tag),
        "description": describe_tb_tag(tag),
        "step": step,
        "value": value,
        "value_text": format(value, ".6g"),
        "tag_filter": tensorboard_tag_filter(tag),
    }


def _group_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = _empty_groups()
    for row in rows:
        kind = str(row.get("kind") or "unknown")
        if kind not in groups:
            kind = "unknown"
        groups[kind].append(row)
    for kind in _GROUP_ORDER:
        groups[kind].sort(key=lambda row: str(row.get("tag") or ""))
    return groups


def _empty_groups() -> Dict[str, List[Dict[str, Any]]]:
    return {kind: [] for kind in _GROUP_ORDER}


def _load_task(task_id: int) -> Optional[Dict[str, Any]]:
    from db.db_task_service import DBTaskService

    task = DBTaskService.get_by_id(int(task_id))
    if task is None:
        return None
    return {
        "primary_id": int(getattr(task, "primary_id", 0) or 0),
        "task_name": str(getattr(task, "task_name", "") or ""),
        "status": str(getattr(task, "status", "") or ""),
        "rollout_root": str(getattr(task, "rollout_root", "") or ""),
    }


def _list_picker_tasks() -> List[Dict[str, Any]]:
    from db.db_task_service import DBTaskService

    try:
        tasks = DBTaskService.list_tasks(limit=_PICKER_LIMIT)
    except Exception as exc:  # noqa: BLE001
        logger.warning("洞察页列出任务失败 err={}", exc)
        return []
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        primary_id = int(getattr(task, "primary_id", 0) or 0)
        if primary_id <= 0:
            continue
        rows.append(
            {
                "primary_id": primary_id,
                "task_name": str(getattr(task, "task_name", "") or ""),
                "status": str(getattr(task, "status", "") or ""),
            }
        )
    return rows
