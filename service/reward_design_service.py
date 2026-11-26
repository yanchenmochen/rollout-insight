#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""奖励设计页：静态流程说明 + 实践策略 + 可选任务问题摘要。"""
from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

from data.agentic_problem_catalog import list_agentic_problems
from data.reward_design_playbook import REWARD_FLOW_NOTES, list_playbook_entries

__all__ = ["build_reward_design_page"]

_PICKER_LIMIT = 200


def build_reward_design_page(task_id: Optional[int] = None) -> Dict[str, Any]:
    """组装奖励设计页上下文。``task_id`` 为空时只给目录与任务选择列表。"""
    page: Dict[str, Any] = {
        "tasks": _list_picker_tasks(),
        "task": None,
        "task_missing": False,
        "notice": None,
        "flow_notes": list(REWARD_FLOW_NOTES),
        "playbook": list_playbook_entries(),
        "catalog": list_agentic_problems(),
        "problem_report": None,
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
    try:
        from service.agentic_problem_service import build_task_problem_report

        report = build_task_problem_report(task_id_int)
        page["problem_report"] = report
    except Exception as exc:  # noqa: BLE001
        logger.warning("奖励设计页加载问题摘要失败 task_id={} err={}", task_id_int, exc)
        page["notice"] = f"任务已选定，但问题摘要加载失败：{exc}"
    return page


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


def _list_picker_tasks() -> list:
    from db.db_task_service import DBTaskService

    try:
        tasks = DBTaskService.list_tasks(limit=_PICKER_LIMIT)
    except Exception as exc:  # noqa: BLE001
        logger.warning("奖励设计页列出任务失败 err={}", exc)
        return []
    rows = []
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
