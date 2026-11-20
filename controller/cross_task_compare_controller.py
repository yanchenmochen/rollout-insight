"""跨任务对比路由。

- GET  /api/cross-task-compare                   跨任务对比页（选任务 + 共同题目趋势）
- GET  /api/cross-task-compare.json              对比数据 JSON（task_ids 逗号分隔）
- GET  /api/cross-task-compare/instance.json      只看某一道题在多任务上的变化

回答的问题：不同训练配置下，共同题目里哪些变好了、哪些变坏了。
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
