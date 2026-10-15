#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""rollout-insight FastAPI 入口（v0.1.0 骨架版）。

- 监听端口由环境变量 ROLLOUT_INSIGHT_PORT 控制，默认 39998（避免与 automate 39999 冲突）
- 启动时自动 include controller/ 下所有 router
- 全局捕获 RolloutInsightException，转为统一 Response.error
- 根路径 / 由 controller/task_controller.py 的 page_task_list 渲染 task_list.html
"""
import importlib
import os
import pkgutil
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from data.response import Response
from data.rollout_insight_exception import RolloutInsightException
from db.db_element import Base, Session, engine
from utils.auth_session import install_session_middleware
from utils.bootstrap import bootstrap_on_startup
from utils.config_loader import config
from utils.log_setup import setup_logging

file = Path(__file__)
sys.path.append(str(file.parent.parent))

# 初始化日志
setup_logging()

description = """
rollout-insight 是 Slime rollout 落盘轨迹分析与检索 Web 平台。

- v0.1.0：基础脚手架、Cursor 规则、分析工具
- v0.2.0：SQLite 任务/Attempt/Segment 持久化 + 多轮对话页
- v0.3.0：南湖调度接入 + Cursor skill 化
"""

app = FastAPI(
    title="rollout-insight",
    description=description,
    version=config.rollout_insight_version,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# v0.5.1：在 include_all_routers() 之前装 SessionMiddleware；
# 这样后面挂载的 business router 拿到的 ``request.session`` 才有 cookie 解析结果。
install_session_middleware(app)


# v0.5.1：全局中间件 —— 把 ``request.state.current_user`` 注入好，方便模板层
# 直接读 ``{{ current_user.username }}``（不依赖 controller 层显式传入）。
# 注：中间件层只做"轻量注入"（不抛 401），实际权限校验仍由各 router 的 Depends(require_login) 负责。
@app.middleware("http")
async def _inject_current_user_middleware(request: Request, call_next):
    """尝试把 session 里的 user_id 解析成 DBUser，挂到 ``request.state.current_user``。

    任何异常都不会中断请求 —— 失败时 ``current_user = None``，由 ``require_login`` 兜底。
    """
    current_user_obj = None
    try:
        from utils.auth_session import current_user as _cu

        # current_user 内部会主动把 user 塞到 request.state.current_user
        current_user_obj = _cu(request)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[middleware] current_user 解析失败（忽略）: {}", exc)
        request.state.current_user = None
    if current_user_obj is None:
        # current_user 内部已经塞过 state；这里兜底 None
        request.state.current_user = getattr(request.state, "current_user", None)
    return await call_next(request)


def include_all_routers() -> None:
    """动态 include controller/ 下所有 router。"""
    package = importlib.import_module("controller")
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"controller.{module_name}")
        if hasattr(module, "router"):
            app.include_router(getattr(module, "router"))
            logger.info("已注册 router: controller.{}", module_name)


include_all_routers()

# 启动时建表（v0.1.0 暂为空模型；v0.2 起正式使用）
Base.metadata.create_all(bind=engine)

# v0.5.0 启动期 bootstrap：
#   - DELETE FROM ri_segment_turn_table（p17_a：丢弃旧 turn 行）
#   - ensure_admin_seed（若 user 表空则插入 admin/admin123）
try:
    bootstrap_summary = bootstrap_on_startup()
    logger.info("[main] v0.5.0 bootstrap summary: {}", bootstrap_summary)
except Exception as exc:  # noqa: BLE001
    logger.exception("[main] bootstrap_on_startup 失败: {}", exc)


@app.exception_handler(RolloutInsightException)
async def rollout_exception_handler(request: Request, exc: RolloutInsightException):
    """业务异常：返回统一 Response.error 结构。"""
    logger.warning("RolloutInsightException: {}", exc)
    return JSONResponse(
        status_code=200,
        content=Response.error(msg=exc.message, code=exc.code).model_dump(),
    )


# v0.5.1：把 ``HTTPException(detail={code, msg})`` 转成统一 Response.error 结构
# （默认 FastAPI 给的是 ``{"detail": {...}}``，不方便客户端解析）
from fastapi import HTTPException as _HTTPException


@app.exception_handler(_HTTPException)
async def http_exception_handler(request: Request, exc: _HTTPException):
    """全局 HTTPException 处理器。

    规则：
    - exc.detail 是 ``{code, msg}`` 字典 → 用 ``code`` 作为 HTTP 状态码，``msg`` 作为 msg
    - exc.detail 是字符串 → 用 exc.status_code + exc.detail
    - exc.headers 有 Location → 透传（用于 302 重定向）
    """
    detail = exc.detail
    headers = exc.headers or None
    status_code = exc.status_code

    # 如果 detail 是 dict 且含 code 字段，把 code 作为业务码写到 body，但 HTTP 状态码仍用 status_code
    if isinstance(detail, dict):
        code = detail.get("code", status_code)
        msg = detail.get("msg") or detail.get("detail") or str(detail)
    elif isinstance(detail, str):
        code = status_code
        msg = detail
    else:
        code = status_code
        msg = str(detail)

    body = Response.error(msg=str(msg), code=int(code)).model_dump()
    # 业务码 401 等保留 HTTP 状态码对齐；redirection（302）保留
    return JSONResponse(status_code=status_code, content=body, headers=headers)


@app.get("/healthz", include_in_schema=False)
async def healthz():
    """健康检查。"""
    return Response.success(
        msg="ok",
        data={"version": config.rollout_insight_version},
    )


# 静态文件
static_dir = Path(__file__).parent / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    port = int(os.environ.get("ROLLOUT_INSIGHT_PORT", "39998"))
    logger.info("rollout-insight 启动中，端口={}", port)
    uvicorn.run(app="main:app", host="0.0.0.0", port=port, reload=False)
