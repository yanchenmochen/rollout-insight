"""训练曲线路由。

- GET  /api/tasks/{task_id}/train-curve                训练曲线页（指标按钮组 + 折线）
- GET  /api/tasks/{task_id}/train-curve.json           同上，返回 JSON
- POST /api/tasks/{task_id}/train-curve/preference     保存指标偏好（全局 / 任务级）

读 TB 事件目录会阻塞，统一放到线程池。
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel, Field

from data.response import Response
from utils.auth_session import current_user, require_login
from utils.config_loader import config
from utils.template_factory import create_templates

router = APIRouter(dependencies=[Depends(require_login)])
templates = create_templates()


class CurvePreferenceRequest(BaseModel):
    """指标偏好保存请求。"""

    keys: List[str] = Field(default_factory=list, description="内置指标 key 列表")
    custom_keys: List[str] = Field(
        default_factory=list, description="自定义指标 key 列表（custom:<tag>）"
    )
    scope: str = Field(default="global", description="global | task")


def _user_id(request: Request) -> Optional[int]:
    """取当前登录用户主键；未登录返回 None。"""
    user = current_user(request)
    if user is None:
        return None
    return int(getattr(user, "primary_id", 0) or 0) or None


async def _task_name(task_id: int) -> Optional[str]:
    from db.db_task_service import DBTaskService

    task = await asyncio.to_thread(DBTaskService.get_by_id, int(task_id))
    if task is None:
        return None
    return task.task_name


@router.get("/api/tasks/{task_id}/train-curve", include_in_schema=False)
async def page_train_curve(request: Request, task_id: int) -> HTMLResponse:
    """渲染训练曲线页：指标按钮组 + 折线，可保存指标配置。"""
    from utils.train_curve_catalog import DEFAULT_CURVE_KEYS

    task_name = await _task_name(task_id)
    return templates.TemplateResponse(
        request,
        "train_curve.html",
        {
            "page_title": "训练曲线",
            "task_id": task_id,
            "task_name": task_name,
            "default_keys": list(DEFAULT_CURVE_KEYS),
        },
    )


@router.get("/api/tasks/{task_id}/train-curve.json")
async def api_train_curve(request: Request, task_id: int) -> Response:
    """返回训练曲线数据：指标按钮组 + 每个指标的折线点。"""
    from service.train_curve_service import build_train_curve

    logger.info("训练曲线 task_id={}", task_id)
    data = await asyncio.to_thread(
        build_train_curve, int(task_id), _user_id(request)
    )
    return Response.success(msg="ok", data=data)


@router.post("/api/tasks/{task_id}/train-curve/preference")
async def api_save_train_curve_preference(
    request: Request,
    task_id: int,
    body: CurvePreferenceRequest,
) -> Response:
    """保存指标偏好。scope=global 写用户默认，scope=task 写该任务追加项。"""
    from service.train_curve_service import save_curve_preference

    logger.info(
        "保存训练曲线配置 task_id={} scope={}", task_id, body.scope
    )
    data = await asyncio.to_thread(
        save_curve_preference,
        int(task_id),
        _user_id(request),
        keys=list(body.keys or []),
        custom_keys=list(body.custom_keys or []),
        scope=body.scope,
    )
    return Response.success(msg="ok", data=data)
