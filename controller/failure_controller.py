"""失败样本分析路由（任务级）。

- GET /api/tasks/{task_id}/failures                    任务下的失败总览页
- GET /api/failures                                    旧入口：有 task_id 时同总览，否则短说明
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

from data.response import Response
from utils.auth_session import require_login
from utils.config_loader import config
from utils.template_factory import create_templates

router = APIRouter(dependencies=[Depends(require_login)])
templates = create_templates()


async def _failure_page_context(
    task_id: Optional[int],
    rollout_id: Optional[int],
) -> dict:
    """总览页上下文。有 task_id 时补任务名，供面包屑使用。"""
    task_name = None
    if task_id is not None:
        from db.db_task_service import DBTaskService

        task = await asyncio.to_thread(DBTaskService.get_by_id, int(task_id))
        if task is not None:
            task_name = task.task_name
    return {
        "page_title": "失败分析",
        "task_id": task_id,
        "task_name": task_name,
        "rollout_id": rollout_id,
    }


@router.get("/api/tasks/{task_id}/failures", include_in_schema=False)
async def page_task_failures(
    request: Request,
    task_id: int,
    rollout_id: Optional[int] = None,
    cat: Optional[str] = None,
    reason: Optional[str] = None,
    agent: Optional[str] = None,
) -> HTMLResponse:
    """任务详情下的失败分析。复用 failure_overview.html 与 overview.json，不重写聚类。

    支持 ``cat`` / ``reason`` / ``agent`` 初始筛选，便于从任务详情页的
    「具体类型」清单直接跳到对应的 attempt 列表。
    """
    ctx = await _failure_page_context(task_id, rollout_id)
    ctx["init_cat"] = cat or ""
    ctx["init_reason"] = reason or ""
    ctx["init_agent"] = agent or ""
    return templates.TemplateResponse(request, "failure_overview.html", ctx)


@router.get("/api/failures", include_in_schema=False)
async def page_failure_overview(
    request: Request,
    task_id: Optional[int] = None,
    rollout_id: Optional[int] = None,
) -> HTMLResponse:
    """旧地址。有 task_id 时与任务失败分析相同；没有 task_id 时只给回任务列表的说明。"""
    ctx = await _failure_page_context(task_id, rollout_id)
    return templates.TemplateResponse(request, "failure_overview.html", ctx)


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
