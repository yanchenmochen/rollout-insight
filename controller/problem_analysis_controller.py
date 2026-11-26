"""任务题目分析路由。

- GET  /api/tasks/{task_id}/problem-analysis            题目分析页
- GET  /api/tasks/{task_id}/problem-analysis/pass.json  只返回按题目的通过率
- GET  /api/tasks/{task_id}/problem-analysis.json       通过率曲线 + 已有响应统计
- POST /api/tasks/{task_id}/response-stats              按 rollout 分批补响应计数

读 attempt / turn / .pt 会阻塞，统一放到线程池。
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


async def _task_name(task_id: int) -> Optional[str]:
    from db.db_task_service import DBTaskService

    task = await asyncio.to_thread(DBTaskService.get_by_id, int(task_id))
    if task is None:
        return None
    return task.task_name


@router.get("/api/tasks/{task_id}/problem-analysis", include_in_schema=False)
async def page_problem_analysis(request: Request, task_id: int) -> HTMLResponse:
    """任务详情下的题目分析：通过率曲线与 sglang 响应统计。"""
    task_name = await _task_name(task_id)
    return templates.TemplateResponse(
        request,
        "problem_analysis.html",
        {
            "page_title": "题目分析",
            "task_id": task_id,
            "task_name": task_name,
        },
    )


@router.get("/api/tasks/{task_id}/problem-analysis/instance", include_in_schema=False)
async def page_instance_appearances(
    request: Request,
    task_id: int,
    instance_id: str = Query("", description="题目名，即 instance_id"),
) -> HTMLResponse:
    """按题目名列出每次出现的 rollout 与轨迹入口。"""
    from service.problem_analysis_service import build_instance_appearances

    task_name = await _task_name(task_id)
    view = await asyncio.to_thread(build_instance_appearances, int(task_id), instance_id)
    return templates.TemplateResponse(
        request,
        "problem_instance.html",
        {
            "page_title": "题目出现",
            "task_id": task_id,
            "task_name": task_name,
            "view": view,
        },
    )


@router.get("/api/tasks/{task_id}/problem-analysis/pass.json")
async def api_problem_pass_trend(task_id: int) -> Response:
    """按题目列出历次出现的 rollout 与通过率，不读响应统计。"""
    from service.problem_analysis_service import build_pass_trend

    logger.info("题目通过率 task_id={}", task_id)
    trend = await asyncio.to_thread(build_pass_trend, int(task_id))
    return Response.success(msg="ok", data=trend)


@router.get("/api/tasks/{task_id}/problem-analysis.json")
async def api_problem_analysis(task_id: int) -> Response:
    """通过率（按题目出现次序）以及当前已能给出的响应统计。"""
    from service.problem_analysis_service import (
        build_pass_trend,
        build_response_tool_stats,
    )

    logger.info("题目分析 task_id={}", task_id)
    pass_trend = await asyncio.to_thread(build_pass_trend, int(task_id))
    response = await asyncio.to_thread(build_response_tool_stats, int(task_id))
    return Response.success(
        msg="ok",
        data={"pass_trend": pass_trend, "response": response},
    )


@router.post("/api/tasks/{task_id}/response-stats")
async def api_advance_response_stats(
    task_id: int,
    rollout_limit: int = Query(1, ge=1, le=8),
) -> Response:
    """补统计下一批 rollout 的 assistant 回复。不写入 turn 正文。"""
    from service.problem_analysis_service import advance_response_tool_stats

    logger.info("补响应统计 task_id={} limit={}", task_id, rollout_limit)
    data = await asyncio.to_thread(advance_response_tool_stats, int(task_id), int(rollout_limit))
    return Response.success(msg="ok", data=data)
