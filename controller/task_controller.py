"""任务相关路由（v0.1.0 占位）。

v0.1.0 仅提供：
- GET /api/tasks：返回任务列表页占位信息
后续阶段会接入：查询、筛选、详情、暂停、终止、删除等接口。
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from data.response import Response
from utils.config_loader import config

router = APIRouter()


@router.get("/api/tasks")
async def list_tasks():
    """任务列表入口（占位）。

    v0.1.0 仅返回版本号与提示语；v0.2 起会接入任务表。
    """
    return Response.success(
        msg="rollout-insight v0.1.0 骨架就绪",
        data={
            "version": config.rollout_insight_version,
            "title": config.rollout_insight_title,
            "hint": "任务表尚未初始化（v0.2 阶段落地），当前仅占位。",
            "tasks": [],
        },
    )
