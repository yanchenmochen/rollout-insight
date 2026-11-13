"""失败样本分析路由（任务级）。

- GET /api/failures                                    失败总览页（无 task_id 时为任务选择）
- GET /api/failures/attempts/{attempt_pk}              单 attempt 失败轨迹页
- GET /api/failures/tasks/{task_id}/overview.json      进度 + 非通过 attempt 列表 + 聚类
- GET /api/failures/attempts/{attempt_pk}/detail.json  单 attempt 结论 / 证据 / patch
- GET /api/failures/attempts/{attempt_pk}/trajectory.json  分页轨迹事件

读 summary 索引 / .pt / 大 history.jsonl 会阻塞，统一放到线程池执行。
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from loguru import logger
from starlette.templating import Jinja2Templates

from data.response import Response
from utils.auth_session import require_login
from utils.config_loader import config

router = APIRouter(dependencies=[Depends(require_login)])
templates = Jinja2Templates(directory=str(config.templates_dir))


@router.get("/api/failures", include_in_schema=False)
async def page_failure_overview(
    request: Request,
    task_id: Optional[int] = None,
    rollout_id: Optional[int] = None,
) -> HTMLResponse:
    """渲染 failure_overview.html；无 task_id 时列出任务供选择。"""
    from service.failure_analysis_service import list_tasks_with_failure_counts

    tasks = [] if task_id else await asyncio.to_thread(list_tasks_with_failure_counts)
    return templates.TemplateResponse(
        request,
        "failure_overview.html",
        {
            "page_title": "失败分析",
            "task_id": task_id,
            "rollout_id": rollout_id,
            "tasks": tasks,
        },
    )


@router.get("/api/failures/attempts/{attempt_pk}", include_in_schema=False)
async def page_failure_attempt(request: Request, attempt_pk: int) -> HTMLResponse:
    """渲染 failure_attempt.html：单 attempt 怎么失败的。"""
    from service.failure_analysis_service import get_attempt_failure_detail

    detail = await asyncio.to_thread(get_attempt_failure_detail, attempt_pk)
    return templates.TemplateResponse(
        request,
        "failure_attempt.html",
        {"page_title": f"失败轨迹 · attempt #{attempt_pk}", "d": detail},
    )


@router.get("/api/failures/tasks/{task_id}/overview.json")
async def api_failure_overview(task_id: int, rollout_id: Optional[int] = None) -> Response:
    """任务失败总览：每 rollout 结局分布 + 非通过 attempt 列表 + 原因聚类。"""
    from service.failure_analysis_service import get_task_failure_overview

    logger.info("失败总览 task_id={} rollout_id={}", task_id, rollout_id)
    data = await asyncio.to_thread(get_task_failure_overview, task_id, rollout_id)
    return Response.success(msg="ok", data=data)


@router.get("/api/failures/attempts/{attempt_pk}/detail.json")
async def api_failure_attempt_detail(attempt_pk: int) -> Response:
    """单 attempt 失败详情（结论 / 评测证据 / 耗时 / patch / 回跳链接）。"""
    from service.failure_analysis_service import get_attempt_failure_detail

    data = await asyncio.to_thread(get_attempt_failure_detail, attempt_pk)
    return Response.success(msg="ok", data=data)


@router.get("/api/failures/attempts/{attempt_pk}/trajectory.json")
async def api_failure_attempt_trajectory(
    attempt_pk: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
    errors_only: bool = False,
) -> Response:
    """分页返回 OpenHands 轨迹事件（errors_only=true 只看报错步）。"""
    from service.failure_analysis_service import get_attempt_trajectory

    data = await asyncio.to_thread(get_attempt_trajectory, attempt_pk, offset, limit, errors_only)
    return Response.success(msg="ok", data=data)
