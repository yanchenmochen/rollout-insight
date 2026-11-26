"""洞察 · 奖励设计。

- GET /api/insights/reward-design           奖励设计页（可选 task_id）
- GET /api/insights/reward-design.json      同页 JSON（含可选问题摘要）
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


@router.get("/api/insights/reward-design", include_in_schema=False)
async def page_reward_design(
    request: Request,
    task_id: Optional[int] = None,
) -> HTMLResponse:
    """渲染奖励设计页：流程说明、实践策略、可选任务问题摘要。"""
    from service.reward_design_service import build_reward_design_page

    page = await asyncio.to_thread(build_reward_design_page, task_id)
    task = page.get("task") or None
    title = "奖励设计"
    if task and task.get("task_name"):
        title = f"奖励设计 · {task['task_name']}"
    return templates.TemplateResponse(
        request,
        "reward_design.html",
        {
            "page_title": title,
            "task_id": task_id,
            **page,
        },
    )


@router.get("/api/insights/reward-design.json")
async def api_reward_design(task_id: Optional[int] = None) -> Response:
    """返回奖励设计静态目录；带 task_id 时附问题摘要。"""
    from service.reward_design_service import build_reward_design_page

    logger.info("奖励设计 task_id={}", task_id)
    data = await asyncio.to_thread(build_reward_design_page, task_id)
    return Response.success(msg="ok", data=data)
