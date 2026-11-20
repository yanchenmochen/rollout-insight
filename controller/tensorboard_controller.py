"""TensorBoard 启停与多任务曲线对比入口。

- GET  /api/tensorboard                 对比页（新标签打开 TB，不嵌入）
- GET  /api/tensorboard/curve-compare.json  多任务曲线叠加数据（勾选任务 + 指标）
- POST /api/tensorboard/start           启动/复用 TB（scope=all|task）
- GET  /api/tensorboard/status
- POST /api/tensorboard/stop
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel, Field

from data.response import Response
from data.rollout_insight_exception import RolloutInsightException
from utils.auth_session import require_login
from utils.config_loader import config
from utils.template_factory import create_templates

router = APIRouter(dependencies=[Depends(require_login)])
templates = create_templates()


class TensorBoardStartRequest(BaseModel):
    scope: str = Field(default="all", description="all | task")
    task_id: Optional[int] = None


@router.get("/api/tensorboard", include_in_schema=False)
async def page_tensorboard_compare(request: Request) -> HTMLResponse:
    """渲染多任务曲线对比页（勾选任务 + 指标，叠图）。"""
    from service.tensorboard_service import tensorboard_status

    status = tensorboard_status()
    return templates.TemplateResponse(
        request,
        "curve_compare.html",
        {
            "page_title": "多任务曲线对比",
            "tb_status": status,
            "tensorboard_port": int(config.get_int("tensorboard_port", 6006)),
        },
    )


@router.get("/api/tensorboard/curve-compare.json")
async def api_curve_compare(
    task_ids: str = Query("", description="逗号分隔的任务 ID；留空表示全部"),
    keys: str = Query("", description="逗号分隔的指标 key；留空用默认按钮组"),
    view: str = Query("step", description="step | epoch"),
    brief: bool = Query(False, description="只返回任务清单与指标目录，不读事件目录"),
) -> Response:
    """返回多任务曲线叠加数据：勾选的任务 × 勾选的指标。

    ``brief=true`` 时只回任务清单与指标目录（页面首屏用，不读事件文件）。
    """
    from service.curve_compare_service import (
        build_curve_compare,
        list_curve_compare_brief,
    )

    if brief:
        data = await asyncio.to_thread(list_curve_compare_brief)
        return Response.success(msg="ok", data=data)

    ids = _parse_int_list(task_ids)
    metric_keys = _parse_str_list(keys)
    logger.info(
        "多任务曲线对比 tasks={} keys={} view={}",
        len(ids) if ids else 0,
        len(metric_keys),
        view,
    )
    data = await asyncio.to_thread(build_curve_compare, ids, metric_keys, view)
    return Response.success(msg="ok", data=data)


def _parse_int_list(raw: str) -> List[int]:
    """解析逗号分隔的整数列表，忽略非法项。"""
    out: List[int] = []
    for piece in str(raw or "").split(","):
        text = piece.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if value > 0 and value not in out:
            out.append(value)
    return out


def _parse_str_list(raw: str) -> List[str]:
    """解析逗号分隔的字符串列表。"""
    out: List[str] = []
    for piece in str(raw or "").split(","):
        text = piece.strip()
        if text and text not in out:
            out.append(text)
    return out


@router.post("/api/tensorboard/start")
async def api_tensorboard_start(
    body: TensorBoardStartRequest,
    request: Request,) -> Response:
    """启动或复用 TensorBoard，返回可新标签打开的 URL。"""
    from service.tensorboard_service import (
        collect_tasks_for_scope,
        public_tensorboard_url,
        start_tensorboard,
    )

    scope = (body.scope or "all").strip().lower()
    if scope not in ("all", "task"):
        raise RolloutInsightException(
            RolloutInsightException.INVALID_ARGUMENT,
            "scope 须为 all 或 task",
        )

    runs, skipped = collect_tasks_for_scope(scope, body.task_id)
    if not runs:
        msg = (
            f"任务 #{body.task_id} 未找到可用的 tbs/ 或 tf_logs/ 目录"
            if scope == "task"
            else "库内任务均未找到可用的 tbs/ 或 tf_logs/ 目录"
        )
        raise RolloutInsightException(
            RolloutInsightException.INVALID_ARGUMENT, msg
        )

    host = request.headers.get("host")
    try:
        data = start_tensorboard(runs, request_host=host)
    except ValueError as exc:
        raise RolloutInsightException(
            RolloutInsightException.INVALID_ARGUMENT, str(exc)
        ) from exc
    except RuntimeError as exc:
        raise RolloutInsightException(
            RolloutInsightException.SYSTEM_ERROR, str(exc)
        ) from exc

    data["skipped"] = skipped
    data["scope"] = scope
    # 确保 url 始终带当前请求 host
    if not data.get("url"):
        data["url"] = public_tensorboard_url(host, int(data.get("port") or 6006))
    logger.info(
        "TensorBoard start scope={} runs={} skipped={} reused={}",
        scope,
        len(data.get("runs") or []),
        len(skipped),
        data.get("reused"),
    )
    return Response.success(msg="ok", data=data)


@router.get("/api/tensorboard/status")
async def api_tensorboard_status(request: Request) -> Response:
    from service.tensorboard_service import (
        public_tensorboard_url,
        tensorboard_status,
    )

    status = tensorboard_status()
    if status.get("running") and status.get("port"):
        status["url"] = public_tensorboard_url(
            request.headers.get("host"), int(status["port"])
        )
    else:
        status["url"] = None
    return Response.success(msg="ok", data=status)


@router.post("/api/tensorboard/stop")
async def api_tensorboard_stop() -> Response:
    from service.tensorboard_service import stop_tensorboard

    data = stop_tensorboard()
    return Response.success(msg="ok", data=data)
