"""任务问题分析路由（agentic RL 常见问题）。

- GET /api/tasks/{task_id}/agent-problems       问题分析页
- GET /api/tasks/{task_id}/agent-problems.json  问题卡片 JSON
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from loguru import logger

from data.response import Response
from utils.auth_session import require_login
from utils.template_factory import create_templates

router = APIRouter(dependencies=[Depends(require_login)])
templates = create_templates()


async def _task_name(task_id: int) -> Optional[str]:
    from db.db_task_service import DBTaskService

    task = await asyncio.to_thread(DBTaskService.get_by_id, int(task_id))
    if task is None:
        return None
    return task.task_name


@router.get("/api/tasks/{task_id}/agent-problems", include_in_schema=False)
async def page_agent_problems(request: Request, task_id: int) -> HTMLResponse:
    """任务级问题分析：常见 agentic 问题占比与奖励动作。"""
    task_name = await _task_name(task_id)
    return templates.TemplateResponse(
        request,
        "agent_problems.html",
        {
            "page_title": "问题分析",
            "task_id": task_id,
            "task_name": task_name,
        },
    )


@router.get("/api/tasks/{task_id}/agent-problems.json")
async def api_agent_problems(task_id: int) -> Response:
    """返回按严重度排序的问题卡片。"""
    from service.agentic_problem_service import build_task_problem_report

    logger.info("问题分析 task_id={}", task_id)
    data = await asyncio.to_thread(build_task_problem_report, int(task_id))
    return Response.success(msg="ok", data=data)
