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

from fastapi import APIRouter, BackgroundTasks, Form, Query, Request
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
        "id": getattr(task, "id", None) or getattr(task, "primary_id", None),
        "primary_id": getattr(task, "primary_id", None) or getattr(task, "id", None),
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
        "id": getattr(r, "primary_id", None) or getattr(r, "id", None),
        "primary_id": getattr(r, "primary_id", None) or getattr(r, "id", None),
        "rollout_id": getattr(r, "rollout_id", None),
        "n_samples": getattr(r, "n_samples", None),
        "n_attempts": getattr(r, "n_attempts", None),
        "mean_reward": getattr(r, "mean_reward", None),
        "scanned_at": _format_dt(getattr(r, "scanned_at", None)),
        "pt_path": getattr(r, "pt_path", None),
        "task_id": getattr(r, "task_id", None),
    }


def _attempt_to_dict(a: Any) -> Dict[str, Any]:
    if a is None:
        return {}
    return {
        "id": getattr(a, "primary_id", None) or getattr(a, "id", None),
        "primary_id": getattr(a, "primary_id", None) or getattr(a, "id", None),
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
        "id": getattr(s, "primary_id", None) or getattr(s, "id", None),
        "primary_id": getattr(s, "primary_id", None) or getattr(s, "id", None),
        "attempt_id": getattr(s, "attempt_id", None),
        "rollout_id": getattr(s, "rollout_id", None),
        "group_index": getattr(s, "group_index", None),
        "instance_id": None,  # attempt 表里；为 null 时由 caller 补充
        "segment_index": getattr(s, "segment_index", None),
        "segment_count": getattr(s, "segment_count", None),
        "sandbox_id": getattr(s, "sandbox_id", None),
        "status": getattr(s, "status", None),
        "reward": getattr(s, "reward", None),
        "agent_turns": None,
        "tool_calls": None,
        "response_length": getattr(s, "response_length", None),
        "diff_line_count": None,
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


def _looks_like_browser(request: Request) -> bool:
    """基于 User-Agent 粗判：是否是浏览器。

    `Accept: */*` 常见于 curl（curl 默认带 `Accept: */*`），但 curl 默认还会带
    `User-Agent: curl/...`，所以看 UA 含 'mozilla' / 'chrome' / 'safari' 等
    才视作浏览器。避免 curl 直接访问 /api/tasks 拿到 HTML 模板。
    """
    ua = (request.headers.get("user-agent") or "").lower()
    if not ua:
        return False
    # 显式排除常见 CLI 工具
    if ua.startswith("curl/") or ua.startswith("wget/") or ua.startswith("python-requests"):
        return False
    return any(tok in ua for tok in ("mozilla", "chrome", "safari", "firefox", "edge", "opera"))


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


def _list_tasks_for_display(
    task_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """共享查询逻辑：从 DB 取 task 列表（含分页/过滤），返回 dict 给 JSON 或模板。

    返回字段：
        {"tasks": List[Dict], "total": int, "limit": int, "offset": int,
         "task_name": str|None, "status": str|None, "warning": str|None}
    """
    result: Dict[str, Any] = {
        "tasks": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
        "task_name": task_name,
        "status": status,
        "warning": None,
    }

    Task = _task_model()
    if Task is None:
        result["warning"] = "任务模型尚未就绪（2A 阶段未完成）"
        logger.warning("db.models.Task 不可用，返回空列表占位")
        return result

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

            result["total"] = q.count()
            rows: List[Any] = (
                q.order_by(Task.primary_id.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            result["tasks"] = [_task_to_dict(r) for r in rows]
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("查询任务列表失败: {}", exc)
        raise RolloutInsightException(
            RolloutInsightException.DB_ERROR, f"查询任务失败: {exc}"
        ) from exc


# 已知 status 值（供模板下拉框渲染）
_TASK_STATUS_OPTIONS: tuple = (
    "PENDING",
    "SCANNING",
    "READY",
    "FAILED",
    "PENDING_SCAN",
)


@router.get("/", include_in_schema=False)
async def page_task_list(
    request: Request,
    task_name: Optional[str] = Query(default=None, description="模糊匹配任务名"),
    status: Optional[str] = Query(default=None, description="按状态过滤"),
) -> HTMLResponse:
    """根路径 → task_list.html（直接注入任务列表到模板上下文）。

    浏览器/curl 带 Accept: text/html 访问 / 时直接渲染带表格的 HTML。
    走 GET 表单过滤：query string `?status=READY&task_name=foo` 触发 server-side
    查询，不再依赖前端 fetch。
    """
    data = _list_tasks_for_display(task_name=task_name, status=status, limit=200)
    return templates.TemplateResponse(
        request,
        "task_list.html",
        {
            "page_title": "Rollout 任务列表",
            "tasks": data["tasks"],
            "total": data["total"],
            "filter_task_name": data["task_name"] or "",
            "filter_status": data["status"] or "",
            "status_options": _TASK_STATUS_OPTIONS,
            "warning": data["warning"],
        },
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
    """任务列表端点。

    行为分流（基于 Accept 头）：
    - `Accept: text/html`（或 `text/*`、`*/*` + 浏览器 UA 提示）→ 返回 task_list.html，
      任务列表注入模板上下文，浏览器直接看到表格。
    - 其他（含 `Accept: application/json`）→ 返回 JSON（curl / API 客户端）。

    这样：
    - `curl -H 'Accept: text/html' /api/tasks` 和访问 / 一样看到表格
    - `curl -H 'Accept: application/json' /api/tasks` 仍返回 JSON（不破坏现有客户端）
    """
    # 兼容 page/page_size
    if page is not None:
        offset = (page - 1) * (page_size or 20)
    effective_size = page_size or limit

    data = _list_tasks_for_display(
        task_name=task_name, status=status, limit=effective_size, offset=offset
    )

    # Accept 头分流：浏览器（text/html）→ HTML 模板；其他 → JSON
    accept = (request.headers.get("accept") or "").lower()
    wants_html = (
        "text/html" in accept
        or "text/*" in accept
        or (accept == "*/*" and _looks_like_browser(request))
    )

    if wants_html:
        return templates.TemplateResponse(
            request,
            "task_list.html",
            {
                "page_title": "Rollout 任务列表",
                "tasks": data["tasks"],
                "total": data["total"],
                "filter_task_name": data["task_name"] or "",
                "filter_status": data["status"] or "",
                "status_options": _TASK_STATUS_OPTIONS,
                "warning": data["warning"],
            },
        )

    return Response.success(
        data={
            "tasks": data["tasks"],
            "total": data["total"],
            "limit": effective_size,
            "offset": offset,
            "page": page,
            "page_size": page_size,
        },
    )


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
            new_id = getattr(obj, "primary_id", None) or getattr(obj, "id", None)
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
            obj = session.query(Task).filter(Task.primary_id == task_id).first()
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
                    .order_by(Rollout.primary_id.asc())
                    .all()
                ):
                    rollouts.append(_rollout_to_dict(r))
            if Attempt is not None:
                # 通过 (task_id, rollout_id 业务键) 关联到本任务
                if Rollout is not None:
                    rollout_biz_ids = [
                        r["rollout_id"] for r in rollouts if r.get("rollout_id") is not None
                    ]
                    if rollout_biz_ids:
                        for a in (
                            session.query(Attempt)
                            .filter(
                                Attempt.task_id == task_id,
                                Attempt.rollout_id.in_(rollout_biz_ids),
                            )
                            .order_by(Attempt.primary_id.asc())
                            .limit(500)
                            .all()
                        ):
                            attempts.append(_attempt_to_dict(a))
            if Segment is not None:
                attempt_ids = [a["primary_id"] for a in attempts if a.get("primary_id") is not None]
                if attempt_ids and Attempt is not None:
                    # 批量查 attempt 表，用于补 segment dict 的 instance_id / tool_calls 等
                    attempt_objs = (
                        session.query(Attempt)
                        .filter(Attempt.primary_id.in_(attempt_ids))
                        .all()
                    )
                    attempt_map = {a.primary_id: a for a in attempt_objs}
                    for s in (
                        session.query(Segment)
                        .filter(Segment.attempt_id.in_(attempt_ids))
                        .order_by(Segment.attempt_id.asc(), Segment.segment_index.asc())
                        .limit(1000)
                        .all()
                    ):
                        seg_attempt = attempt_map.get(getattr(s, "attempt_id", None))
                        # 借用 segment_controller._segment_to_dict：让 segment 模板拿得到
                        # instance_id / agent_turns / tool_calls（这些字段在 attempt 表）
                        from controller.segment_controller import (
                            _segment_to_dict as _seg_to_dict,
                        )
                        seg_dict = _seg_to_dict(s, seg_attempt)
                        # 再补 task_id 和 pt_path（用于页面跳转）
                        if seg_attempt is not None:
                            seg_dict["rollout_id"] = getattr(seg_attempt, "rollout_id", None)
                            # 通过 (task_id, rollout_id 业务键) 找 rollout
                            if Rollout is not None and seg_dict["rollout_id"] is not None:
                                ro = (
                                    session.query(Rollout)
                                    .filter(
                                        Rollout.task_id == task_id,
                                        Rollout.rollout_id == seg_dict["rollout_id"],
                                    )
                                    .first()
                                )
                                if ro is not None:
                                    seg_dict["pt_path"] = getattr(ro, "pt_path", None)
                        segments.append(seg_dict)
                else:
                    # 无 attempts 时（极端情况）只列 segment 主键
                    for s in (
                        session.query(Segment)
                        .filter(Segment.attempt_id.in_(attempt_ids))
                        .order_by(Segment.attempt_id.asc(), Segment.segment_index.asc())
                        .limit(1000)
                        .all()
                    ):
                        from controller.segment_controller import (
                            _segment_to_dict as _seg_to_dict,
                        )
                        segments.append(_seg_to_dict(s))

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
async def api_rescan_task(
    task_id: int,
    background: BackgroundTasks = None,  # 由 FastAPI 依赖注入
) -> Response:
    """触发 RolloutScanService.rescan_task（异步后台跑，请求立即返回）。

    异步原因：单 task 含 30~50 个 .pt 同步扫会持续 1~3 分钟，期间阻塞 server 线程
    让所有其他请求都排队。改成 BackgroundTasks 后请求立即返回，scan 在后台跑。
    """
    Task = _task_model()
    if Task is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "任务模型尚未就绪（2A 阶段未完成）",
        )
    # 校验 task 存在性（先同步快路径，避免后台跑空）
    Session = _session_factory()
    if Session is None:
        raise RolloutInsightException(
            RolloutInsightException.DB_ERROR, "Session 工厂不可用"
        )
    with Session() as session:
        obj = session.query(Task).filter(Task.primary_id == task_id).first()
        if obj is None:
            raise RolloutInsightException(
                RolloutInsightException.TASK_NOT_FOUND, f"任务 #{task_id} 不存在"
            )

    scan_cls = _scan_service()
    if scan_cls is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "扫描服务尚未就绪（2B 阶段未完成）",
        )
    # 把扫描塞到后台；请求立即返回
    if background is not None:
        background.add_task(_run_rescan_background, scan_cls, task_id)
    return Response.success(
        msg="已提交后台重扫",
        data={
            "task_id": task_id,
            "status": "PENDING_SCAN",
            "hint": "用 GET /api/tasks/{id} 查看进度（status=READY/FAILED/SCANNING）",
        },
    )


def _run_rescan_background(scan_cls, task_id: int) -> None:
    """后台线程：执行 rescan，仅记日志，不抛回响应。"""
    try:
        inst = scan_cls() if callable(scan_cls) else scan_cls
        inst.rescan_task(task_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("后台重扫 task_id={} 失败: {}", task_id, exc)


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
            obj = session.query(Task).filter(Task.primary_id == task_id).first()
            if obj is None:
                raise RolloutInsightException(
                    RolloutInsightException.TASK_NOT_FOUND, f"任务 #{task_id} 不存在"
                )

            # 防御性：按子表顺序级联删除（即使 ORM 已配 cascade，这里也兜底）
            if Rollout is not None and Attempt is not None and Segment is not None:
                rollout_ids = [
                    rid for (rid,) in session.query(Rollout.primary_id).filter(Rollout.task_id == task_id).all()
                ]
                if rollout_ids:
                    attempt_ids = [
                        aid for (aid,) in session.query(Attempt.primary_id).filter(Attempt.rollout_id.in_(rollout_ids)).all()
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
