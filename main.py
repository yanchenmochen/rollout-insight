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


@app.exception_handler(RolloutInsightException)
async def rollout_exception_handler(request: Request, exc: RolloutInsightException):
    """业务异常：返回统一 Response.error 结构。"""
    logger.warning("RolloutInsightException: {}", exc)
    return JSONResponse(
        status_code=200,
        content=Response.error(msg=exc.message, code=exc.code).model_dump(),
    )


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
