"""洞察页：指标说明、分析流程、跳转 TensorBoard。

- GET /api/insights              先选任务
- GET /api/insights?task_id=     该任务的 scalar 与分析顺序
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from utils.auth_session import require_login
from utils.config_loader import config

router = APIRouter(dependencies=[Depends(require_login)])
templates = Jinja2Templates(directory=str(config.templates_dir))


@router.get("/api/insights", include_in_schema=False)
async def page_insights(
    request: Request,
    task_id: Optional[int] = None,
) -> HTMLResponse:
    """渲染洞察页。没有事件目录时页面给出说明，不返回 500。"""
    from service.insight_page_service import build_insight_page

    page = await asyncio.to_thread(build_insight_page, task_id)
    task = page.get("task") or None
    title = "洞察"
    if task and task.get("task_name"):
        title = f"洞察 · {task['task_name']}"
    return templates.TemplateResponse(
        request,
        "insights.html",
        {
            "page_title": title,
            "task_id": task_id,
            **page,
        },
    )
