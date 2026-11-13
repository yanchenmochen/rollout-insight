"""任务级训练 P0 指标（tbs / 日志）加载。"""
from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from utils.run_train_metrics import empty_p0_payload, load_p0_metrics

__all__ = ["get_task_p0_train_metrics"]


def get_task_p0_train_metrics(task_id: int) -> Dict[str, Any]:
    """按 task.rollout_root 解析 P0 奖励设计指标。"""
    from data.rollout_insight_exception import RolloutInsightException
    from db.db_element import Session
    from db.db_task import DBTask

    with Session() as session:
        task = session.query(DBTask).filter(DBTask.primary_id == task_id).first()
        if task is None:
            raise RolloutInsightException(
                RolloutInsightException.TASK_NOT_FOUND, f"任务 #{task_id} 不存在"
            )
        rollout_root = getattr(task, "rollout_root", None) or ""
        task_name = getattr(task, "task_name", "") or ""

    try:
        payload = load_p0_metrics(rollout_root, use_cache=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "加载 P0 训练指标失败 task_id={} root={} err={}",
            task_id,
            rollout_root,
            exc,
        )
        payload = empty_p0_payload(source="none")
        payload["warning"] = f"加载失败: {exc}"

    payload["task_id"] = int(task_id)
    payload["task_name"] = task_name
    payload["rollout_root"] = rollout_root
    return payload
