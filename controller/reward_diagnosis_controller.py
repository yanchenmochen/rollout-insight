"""任务奖励诊断路由。

- GET /api/tasks/{task_id}/reward-diagnosis       诊断页
- GET /api/tasks/{task_id}/reward-diagnosis.json  有效梯度组 + 奖励分解 + 行为卡片
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from loguru import logger
from starlette.templating import Jinja2Templates

from data.response import Response
from utils.auth_session import require_login
from utils.config_loader import config

router = APIRouter(dependencies=[Depends(require_login)])
templates = Jinja2Templates(directory=str(config.templates_dir))


async def _task_name(task_id: int) -> Optional[str]:
    from db.db_task_service import DBTaskService

    task = await asyncio.to_thread(DBTaskService.get_by_id, int(task_id))
    if task is None:
        return None
    return task.task_name


@router.get("/api/tasks/{task_id}/reward-diagnosis", include_in_schema=False)
async def page_reward_diagnosis(request: Request, task_id: int) -> HTMLResponse:
    """任务详情旁的奖励诊断：梯度组、奖励分解、行为失败卡片。"""
    task_name = await _task_name(task_id)
    return templates.TemplateResponse(
        request,
        "reward_diagnosis.html",
        {
            "page_title": "奖励诊断",
            "task_id": task_id,
            "task_name": task_name,
        },
    )


@router.get("/api/tasks/{task_id}/reward-diagnosis.json")
async def api_reward_diagnosis(task_id: int) -> Response:
    """返回有效梯度组曲线、奖励分解曲线与行为失败卡片。"""
    from service.reward_diagnosis_service import build_reward_diagnosis

    logger.info("奖励诊断 task_id={}", task_id)
    data = await asyncio.to_thread(build_reward_diagnosis, int(task_id))
    return Response.success(msg="ok", data=data)
