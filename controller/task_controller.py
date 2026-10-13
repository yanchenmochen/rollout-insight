"""任务相关路由（v0.2.0 阶段 2C：Jinja 模板 + 控制器）。

提供：
- GET  /                       渲染 task_list.html
- GET  /api/tasks              JSON 列表（带分页 / 过滤）
- GET  /api/tasks/create       渲染 task_create.html
- POST /api/tasks              创建任务（form-encoded），写库后 302 跳详情
- GET  /api/tasks/{id}         渲染 task_detail.html
- POST /api/tasks/{id}/rescan  触发 RolloutScanService.rescan_task
- POST /api/tasks/{id}/delete  删除任务

设计：
- 业务错误用 RolloutInsightException 抛出，由 main.py 的统一 handler 转 JSON
- 参数错误用 Response.error(code=50001, msg=...) 直接返回
- 所有对 db/service 的导入在函数内**延迟**进行，避免 2A/2B worker
  还没提交时被 import 链拉断
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from loguru import logger
from starlette.templating import Jinja2Templates

from data.response import Response
from data.rollout_insight_exception import RolloutInsightException
from utils.config_loader import config

router = APIRouter()

# 模板对象复用 main 启动时相同的目录
templates = Jinja2Templates(directory=str(config.templates_dir))


# ---------------------------------------------------------------------------
# 延迟导入辅助：避免循环/未就绪依赖把 controller 拉崩
# ---------------------------------------------------------------------------

def _safe_import_attr(module_name: str, attr: str):
    """延迟导入 module_name.attr；导入失败返回 None，不抛错。"""
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return getattr(mod, attr, None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("延迟导入 {}.{} 失败: {}", module_name, attr, exc)
        return None


def _task_service_module():
    return _safe_import_attr("service.task_service", "TaskService") or _safe_import_attr(
        "service", "task_service"
    )


def _scan_service():
    return _safe_import_attr("service.rollout_scan_service", "RolloutScanService")


def _session_factory():
    """获取 SQLAlchemy Session 工厂（延迟）。"""
    return _safe_import_attr("db.db_element", "Session")


def _task_model():
    """获取 Task ORM 模型类。"""
    return _safe_import_attr("db.models", "Task") or _safe_import_attr(
        "db", "Task"
    )


def _rollout_model():
    return _safe_import_attr("db.models", "Rollout") or _safe_import_attr("db", "Rollout")


def _attempt_model():
    return _safe_import_attr("db.models", "Attempt") or _safe_import_attr("db", "Attempt")


def _segment_model():
    return _safe_import_attr("db.models", "Segment") or _safe_import_attr("db", "Segment")


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------

def _task_to_dict(task: Any) -> Dict[str, Any]:
    """把 ORM Task 转成模板/JSON 友好的 dict。"""
    if task is None:
        return {}
    return {
        "id": getattr(task, "id", None),
        "task_name": getattr(task, "task_name", None),
        "status": getattr(task, "status", None),
        "source_type": getattr(task, "source_type", None),
        "nanhu_job_id": getattr(task, "nanhu_job_id", None),
        "rollout_root": getattr(task, "rollout_root", None),
        "tokenizer_dir": getattr(task, "tokenizer_dir", None),
        "recursive": getattr(task, "recursive", None),
        "description": getattr(task, "description", None),
        "rollout_count": getattr(task, "rollout_count", None),
        "attempt_count": getattr(task, "attempt_count", None),
        "segment_count": getattr(task, "segment_count", None),
        "mean_reward": getattr(task, "mean_reward", None),
        "error_message": getattr(task, "error_message", None),
        "create_time": _format_dt(getattr(task, "create_time", None)),
        "update_time": _format_dt(getattr(task, "update_time", None)),
        "last_scanned_at": _format_dt(getattr(task, "last_scanned_at", None)),
    }


def _rollout_to_dict(r: Any) -> Dict[str, Any]:
    if r is None:
        return {}
    return {
        "id": getattr(r, "id", None),
        "rollout_id": getattr(r, "rollout_id", None),
        "n_samples": getattr(r, "n_samples", None),
        "n_attempts": getattr(r, "n_attempts", None),
        "mean_reward": getattr(r, "mean_reward", None),
        "scanned_at": _format_dt(getattr(r, "scanned_at", None)),
        "pt_path": getattr(r, "pt_path", None),
    }


def _attempt_to_dict(a: Any) -> Dict[str, Any]:
    if a is None:
        return {}
    return {
        "id": getattr(a, "id", None),
        "rollout_id": getattr(a, "rollout_id", None),
        "group_index": getattr(a, "group_index", None),
        "instance_id": getattr(a, "instance_id", None),
        "segment_count": getattr(a, "segment_count", None),
        "create_time": _format_dt(getattr(a, "create_time", None)),
    }


def _segment_to_dict(s: Any) -> Dict[str, Any]:
    if s is None:
        return {}
    return {
        "id": getattr(s, "id", None),
        "attempt_id": getattr(s, "attempt_id", None),
        "rollout_id": getattr(s, "rollout_id", None),
        "group_index": getattr(s, "group_index", None),
        "instance_id": getattr(s, "instance_id", None),
        "segment_index": getattr(s, "segment_index", None),
        "segment_count": getattr(s, "segment_count", None),
        "sandbox_id": getattr(s, "sandbox_id", None),
        "status": getattr(s, "status", None),
        "reward": getattr(s, "reward", None),
        "agent_turns": getattr(s, "agent_turns", None),
        "tool_calls": getattr(s, "tool_calls", None),
        "response_length": getattr(s, "response_length", None),
        "diff_line_count": getattr(s, "diff_line_count", None),
    }


def _format_dt(value: Any) -> Optional[str]:
    """把 datetime 转 ISO 字符串；空值/None 返回 None。"""
    if value is None:
        return None
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


# 模板引用的 task 字段全集（保证空占位也有全部 key，模板 `task.foo` 不会爆）
_TASK_PLACEHOLDER_KEYS: tuple = (
    "id", "task_name", "status", "source_type", "nanhu_job_id",
    "rollout_root", "tokenizer_dir", "recursive", "description",
    "rollout_count", "attempt_count", "segment_count", "mean_reward",
    "error_message", "create_time", "update_time", "last_scanned_at",
)


def _empty_task_dict(task_id: Optional[int] = None) -> Dict[str, Any]:
    return {k: None for k in _TASK_PLACEHOLDER_KEYS} | {
        "id": task_id,
        "task_name": "(未找到)",
        "status": "UNKNOWN",
    }


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@router.get("/", include_in_schema=False)
async def page_task_list(request: Request) -> HTMLResponse:
    """根路径 → task_list.html（覆盖 main.py 里的 RedirectResponse 行为）。

    main.py 已经有 `@app.get("/")` 走 RedirectResponse；这里 router 的同路径
    会因为 include 顺序在后生效。不过为防御性，我们仍然接受 GET / 的渲染请求。
    """
    return templates.TemplateResponse(
        request,
        "task_list.html",
        {"page_title": "Rollout 任务列表"},
    )


@router.get("/api/tasks", response_model=None)
async def api_list_tasks(
    request: Request,
    task_name: Optional[str] = Query(default=None, description="模糊匹配任务名"),
    status: Optional[str] = Query(default=None, description="按状态过滤"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    page: Optional[int] = Query(default=None, ge=1, description="页码（1-based）"),
    page_size: Optional[int] = Query(default=None, ge=1, le=500),
):
    """JSON 列表接口；同时供 task_list.html 的 JS 拉数据。"""
    # 兼容 page/page_size
    if page is not None:
        offset = (page - 1) * (page_size or 20)
    effective_size = page_size or limit

    Task = _task_model()
    if Task is None:
        # 2A 还没落地模型：返回空数据 + 友好提示
        logger.warning("db.models.Task 不可用，返回空列表占位")
        return Response.success(
            msg="ok（Task 模型尚未就绪）",
            data={
                "tasks": [],
                "total": 0,
                "limit": effective_size,
                "offset": offset,
                "page": page,
                "page_size": page_size,
            },
        )

    Session = _session_factory()
    if Session is None:
        raise RolloutInsightException(
            RolloutInsightException.DB_ERROR, "Session 工厂不可用"
        )

    try:
        with Session() as session:
            q = session.query(Task)
            if task_name:
                like = f"%{task_name}%"
                q = q.filter(Task.task_name.ilike(like))
            if status:
                q = q.filter(Task.status == status)

            total = q.count()
            rows: List[Any] = (
                q.order_by(Task.id.desc())
                .offset(offset)
                .limit(effective_size)
                .all()
            )
            tasks = [_task_to_dict(r) for r in rows]
        return Response.success(
            data={
                "tasks": tasks,
                "total": total,
                "limit": effective_size,
                "offset": offset,
                "page": page,
                "page_size": page_size,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("查询任务列表失败: {}", exc)
        raise RolloutInsightException(
            RolloutInsightException.DB_ERROR, f"查询任务失败: {exc}"
        ) from exc


@router.get("/api/tasks/create", include_in_schema=False)
async def page_task_create(request: Request) -> HTMLResponse:
    """渲染 task_create.html。"""
    return templates.TemplateResponse(
        request,
        "task_create.html",
        {"page_title": "新建 Rollout 任务"},
    )


@router.post("/api/tasks")
async def api_create_task(
    request: Request,
    task_name: str = Form(...),
    description: str = Form(default=""),
    rollout_root: str = Form(...),
    tokenizer_dir: str = Form(default=""),
    nanhu_job_id: str = Form(default=""),
    source_type: str = Form(default="local_dump"),
    recursive: str = Form(default="0"),
) -> Response:
    """创建任务（form-encoded）。写库后由前端跳到 /api/tasks/{id}。

    注：根据用户任务描述，本路由需要「写库后 302 跳 /api/tasks/{id}」；
    我们同时返回 JSON（data.id）和可选的 location 头，方便前端跳转。
    真正 302 跳转由前端在收到 code=0 响应后根据 data.id 完成。
    """
    if not task_name or not task_name.strip():
        return Response.error(code=50001, msg="任务名不能为空")
    if not rollout_root or not rollout_root.strip():
        return Response.error(code=50001, msg="rollout 根路径不能为空")

    Task = _task_model()
    Session = _session_factory()
    if Task is None or Session is None:
        raise RolloutInsightException(
            RolloutInsightException.TASK_CREATE_FAILED,
            "任务模型尚未就绪（2A 阶段未完成）",
        )

    is_recursive = recursive in ("1", "true", "True", "on", "yes")

    try:
        with Session() as session:
            obj = Task(
                task_name=task_name.strip(),
                description=description or None,
                rollout_root=rollout_root.strip(),
                tokenizer_dir=(tokenizer_dir or "").strip() or None,
                nanhu_job_id=(nanhu_job_id or "").strip() or None,
                source_type=source_type or "local_dump",
                recursive=is_recursive,
                status="PENDING",
            )
            # 模型可能有 default 值字段（rollout_count/mean_reward/create_time 等）
            session.add(obj)
            session.commit()
            session.refresh(obj)
            new_id = getattr(obj, "id", None)
            logger.info("已创建任务 id={} name={}", new_id, task_name)

        # 触发一次初始扫描（如果扫描服务已就绪）；失败不影响创建结果
        try:
            scan = _scan_service()
            if scan is not None and new_id is not None:
                scan_inst = scan() if callable(scan) else scan
                if hasattr(scan_inst, "rescan_task"):
                    # 后台异步触发；不阻塞响应
                    import threading
                    threading.Thread(
                        target=scan_inst.rescan_task,
                        args=(new_id,),
                        daemon=True,
                        name=f"rescan-{new_id}",
                    ).start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("初始扫描触发失败（不影响创建）: {}", exc)

        return Response.success(
            msg="创建成功",
            data={
                "id": new_id,
                "task_id": new_id,
                "redirect": f"/api/tasks/{new_id}" if new_id else "/api/tasks",
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("创建任务失败: {}", exc)
        raise RolloutInsightException(
            RolloutInsightException.TASK_CREATE_FAILED, f"创建任务失败: {exc}"
        ) from exc


@router.get("/api/tasks/{task_id}", include_in_schema=False)
async def page_task_detail(task_id: int, request: Request) -> HTMLResponse:
    """渲染 task_detail.html；带 rollouts/attempts/segments 三张子表。"""
    Task = _task_model()
    Rollout = _rollout_model()
    Attempt = _attempt_model()
    Segment = _segment_model()
    Session = _session_factory()

    if Task is None or Session is None:
        return templates.TemplateResponse(
            request,
            "task_detail.html",
            {
                "page_title": f"任务 #{task_id}",
                "task": _empty_task_dict(task_id),
                "rollouts": [],
                "attempts": [],
                "segments": [],
                "warning": "任务模型尚未就绪（2A 阶段未完成）",
            },
            status_code=200,
        )

    try:
        with Session() as session:
            obj = session.query(Task).filter(Task.id == task_id).first()
            if obj is None:
                # 渲染一个"未找到"占位
                placeholder = _empty_task_dict(task_id)
                placeholder["status"] = "NOT_FOUND"
                return templates.TemplateResponse(
                    request,
                    "task_detail.html",
                    {
                        "page_title": f"任务 #{task_id}",
                        "task": placeholder,
                        "rollouts": [],
                        "attempts": [],
                        "segments": [],
                        "warning": f"任务 #{task_id} 不存在",
                    },
                    status_code=200,
                )

            rollouts: List[Dict[str, Any]] = []
            attempts: List[Dict[str, Any]] = []
            segments: List[Dict[str, Any]] = []

            if Rollout is not None:
                for r in (
                    session.query(Rollout)
                    .filter(Rollout.task_id == task_id)
                    .order_by(Rollout.id.asc())
                    .all()
                ):
                    rollouts.append(_rollout_to_dict(r))
            if Attempt is not None:
                # 通过 rollout_id 关联到本任务
                if Rollout is not None:
                    rollout_ids = [r["id"] for r in rollouts if r.get("id") is not None]
                    if rollout_ids:
                        for a in (
                            session.query(Attempt)
                            .filter(Attempt.rollout_id.in_(rollout_ids))
                            .order_by(Attempt.id.asc())
                            .limit(500)
                            .all()
                        ):
                            attempts.append(_attempt_to_dict(a))
            if Segment is not None:
                attempt_ids = [a["id"] for a in attempts if a.get("id") is not None]
                if attempt_ids:
                    for s in (
                        session.query(Segment)
                        .filter(Segment.attempt_id.in_(attempt_ids))
                        .order_by(Segment.attempt_id.asc(), Segment.segment_index.asc())
                        .limit(1000)
                        .all()
                    ):
                        segments.append(_segment_to_dict(s))

        return templates.TemplateResponse(
            request,
            "task_detail.html",
            {
                "page_title": f"任务 #{task_id} - {obj.task_name}",
                "task": _task_to_dict(obj),
                "rollouts": rollouts,
                "attempts": attempts,
                "segments": segments,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("加载任务详情失败: {}", exc)
        raise RolloutInsightException(
            RolloutInsightException.DB_ERROR, f"加载任务失败: {exc}"
        ) from exc


@router.post("/api/tasks/{task_id}/rescan")
async def api_rescan_task(task_id: int) -> Response:
    """触发 RolloutScanService.rescan_task。"""
    scan_cls = _scan_service()
    if scan_cls is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "扫描服务尚未就绪（2B 阶段未完成）",
        )
    try:
        inst = scan_cls() if callable(scan_cls) else scan_cls
        if not hasattr(inst, "rescan_task"):
            raise RolloutInsightException(
                RolloutInsightException.NOT_IMPLEMENTED,
                "RolloutScanService.rescan_task 不可用",
            )
        result = inst.rescan_task(task_id)
        # rescan_task 可能返回 dict / int / str；尽量规整
        if isinstance(result, dict):
            payload = result
        else:
            payload = {"task_id": task_id, "result": result}
        return Response.success(msg="重扫完成", data=payload)
    except RolloutInsightException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("重扫任务 {} 失败: {}", task_id, exc)
        raise RolloutInsightException(
            RolloutInsightException.TASK_ERROR, f"重扫失败: {exc}"
        ) from exc


@router.post("/api/tasks/{task_id}/delete")
async def api_delete_task(task_id: int) -> Response:
    """删除任务（级联删除其 rollout/attempt/segment）。"""
    Task = _task_model()
    Rollout = _rollout_model()
    Attempt = _attempt_model()
    Segment = _segment_model()
    Session = _session_factory()
    if Task is None or Session is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "任务模型尚未就绪（2A 阶段未完成）",
        )

    try:
        with Session() as session:
            obj = session.query(Task).filter(Task.id == task_id).first()
            if obj is None:
                raise RolloutInsightException(
                    RolloutInsightException.TASK_NOT_FOUND, f"任务 #{task_id} 不存在"
                )

            # 防御性：按子表顺序级联删除（即使 ORM 已配 cascade，这里也兜底）
            if Rollout is not None and Attempt is not None and Segment is not None:
                rollout_ids = [
                    rid for (rid,) in session.query(Rollout.id).filter(Rollout.task_id == task_id).all()
                ]
                if rollout_ids:
                    attempt_ids = [
                        aid for (aid,) in session.query(Attempt.id).filter(Attempt.rollout_id.in_(rollout_ids)).all()
                    ]
                    if attempt_ids:
                        session.query(Segment).filter(Segment.attempt_id.in_(attempt_ids)).delete(synchronize_session=False)
                    session.query(Attempt).filter(Attempt.rollout_id.in_(rollout_ids)).delete(synchronize_session=False)
                    session.query(Rollout).filter(Rollout.task_id == task_id).delete(synchronize_session=False)
            elif Rollout is not None:
                session.query(Rollout).filter(Rollout.task_id == task_id).delete(synchronize_session=False)

            session.delete(obj)
            session.commit()
            logger.info("已删除任务 id={}", task_id)
        return Response.success(msg="已删除", data={"task_id": task_id})
    except RolloutInsightException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("删除任务失败: {}", exc)
        raise RolloutInsightException(
            RolloutInsightException.DB_ERROR, f"删除失败: {exc}"
        ) from exc
