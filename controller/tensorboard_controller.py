"""TensorBoard 启停与多任务曲线对比入口。

- GET  /api/tensorboard                 对比页（新标签打开 TB，不嵌入）
- POST /api/tensorboard/start           启动/复用 TB（scope=all|task）
- GET  /api/tensorboard/status
- POST /api/tensorboard/stop
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel, Field
from starlette.templating import Jinja2Templates

from data.response import Response
from data.rollout_insight_exception import RolloutInsightException
from utils.auth_session import require_login
from utils.config_loader import config

router = APIRouter(dependencies=[Depends(require_login)])
templates = Jinja2Templates(directory=str(config.templates_dir))


class TensorBoardStartRequest(BaseModel):
    scope: str = Field(default="all", description="all | task")
    task_id: Optional[int] = None


@router.get("/api/tensorboard", include_in_schema=False)
async def page_tensorboard_compare(request: Request) -> HTMLResponse:
    """渲染多任务曲线对比页。"""
    from service.tensorboard_service import tensorboard_status

    status = tensorboard_status()
    return templates.TemplateResponse(
        request,
        "tensorboard_compare.html",
        {
            "page_title": "训练曲线对比（TensorBoard）",
            "tb_status": status,
            "tensorboard_port": int(config.get_int("tensorboard_port", 6006)),
        },
    )


@router.post("/api/tensorboard/start")
async def api_tensorboard_start(
    body: TensorBoardStartRequest,
    request: Request,
) -> Response:
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
