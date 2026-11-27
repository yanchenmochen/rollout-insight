"""跨任务对比路由。

- GET  /api/cross-task-compare                   任务题目变化页（单任务或跨任务）
- GET  /api/cross-task-compare.json              对比数据 JSON（task_ids 逗号分隔，1–6 个）
- GET  /api/cross-task-compare/instance.json      只看某一道题在所选任务上的变化

单任务：看该任务全部题目随训练的首末变化；多任务：只看共同题目，并可指定基线。
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from loguru import logger

from data.response import Response
from utils.auth_session import require_login
from utils.config_loader import config
from utils.template_factory import create_templates

router = APIRouter(dependencies=[Depends(require_login)])
templates = create_templates()


def _parse_task_ids(raw: Optional[str]) -> List[int]:
    """把 ``"1,2,3"`` 解析成 ``[1, 2, 3]``。"""
    out: List[int] = []
    for part in str(raw or "").split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if value > 0 and value not in out:
            out.append(value)
    return out


@router.get("/api/cross-task-compare", include_in_schema=False)
async def page_cross_task_compare(request: Request) -> HTMLResponse:
    """渲染跨任务对比页。"""
    from service.cross_task_compare_service import list_compareable_tasks

    tasks = await asyncio.to_thread(list_compareable_tasks)
    return templates.TemplateResponse(
        request,
        "cross_task_compare.html",
        {
            "page_title": "跨任务对比",
            "tasks": tasks,
        },
    )


@router.get("/api/cross-task-compare.json")
async def api_cross_task_compare(
    request: Request,
    task_ids: str = Query("", description="参与对比的任务 id，逗号分隔"),
    instance_id: str = Query("", description="只看某一道题时传入"),
    baseline_task_id: Optional[int] = Query(
        None, description="用户主动选择的基线任务 id；为空取第一个任务"
    ),
) -> Response:
    """返回跨任务对比数据：共同题目的 epoch 趋势 + 变好/变坏清单 + 奖励指引。"""
    from service.cross_task_compare_service import build_cross_task_compare

    ids = _parse_task_ids(task_ids)
    logger.info(
        "跨任务对比 task_ids={} instance_id={} baseline={}",
        ids,
        instance_id,
        baseline_task_id,
    )
    data = await asyncio.to_thread(
        build_cross_task_compare,
        ids,
        instance_id=instance_id,
        baseline_task_id=baseline_task_id,
    )
    return Response.success(msg="ok", data=data)


@router.get("/api/cross-task-compare/instance.json")
async def api_cross_task_instance(
    request: Request,
    task_ids: str = Query("", description="参与对比的任务 id，逗号分隔"),
    instance_id: str = Query("", description="题目名（instance_id）"),
    baseline_task_id: Optional[int] = Query(
        None, description="用户主动选择的基线任务 id；为空取第一个任务"
    ),
) -> Response:
    """返回某一道题在多个任务上的逐次出现与通过率。"""
    from service.cross_task_compare_service import build_instance_compare

    ids = _parse_task_ids(task_ids)
    logger.info(
        "跨任务单题对比 task_ids={} instance_id={} baseline={}",
        ids,
        instance_id,
        baseline_task_id,
    )
    data = await asyncio.to_thread(
        build_instance_compare,
        ids,
        instance_id,
        baseline_task_id=baseline_task_id,
    )
    return Response.success(msg="ok", data=data)


@router.get("/api/cross-task-compare/env.json")
async def api_cross_task_env_compare(
    request: Request,
    task_ids: str = Query("", description="参与对比的任务 id，逗号分隔"),
    baseline_task_id: Optional[int] = Query(
        None, description="基线任务 id；为空取第一个"
    ),
    only_diff: bool = Query(
        True, description="true 时只返回相对基线有差异的关键键"
    ),
    full_catalog: bool = Query(
        False, description="true 时对比全部 env 键（默认只比训练关键键目录）"
    ),
) -> Response:
    """返回多任务训练环境变量差异（相对基线）。"""
    from service.task_env_service import build_task_env_compare
    from utils.auth_session import current_user

    ids = _parse_task_ids(task_ids)
    user = current_user(request)
    local_user_id = getattr(user, "primary_id", None) if user is not None else None
    logger.info(
        "跨任务环境变量对比 task_ids={} baseline={} only_diff={}",
        ids,
        baseline_task_id,
        only_diff,
    )
    data = await asyncio.to_thread(
        build_task_env_compare,
        ids,
        baseline_task_id=baseline_task_id,
        local_user_id=local_user_id,
        fetch_nanhu_env=False,
        only_highlight=not bool(full_catalog),
        only_diff=bool(only_diff),
    )
    return Response.success(msg="ok", data=data)
