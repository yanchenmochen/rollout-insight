"""Rollout 控制器（v0.4.0 阶段：T2 rollout 详情页 + JSON 端点）。

提供：
- GET  /api/tasks/{task_id}/rollouts/{rollout_id}            渲染 rollout_detail.html
- GET  /api/rollouts/{rollout_id}                            JSON 端点
- POST /api/rollouts/{rollout_id}/parse                      触发 T3 重解析（force=true）

约定：
- rollout_id 是 ``ri_rollout_table.primary_id``（不是业务键 ``rollout_id``）
- 业务错误用 RolloutInsightException 抛出
- 所有对 db/service 的导入在函数内**延迟**进行
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from starlette.templating import Jinja2Templates

from data.response import Response
from data.rollout_insight_exception import RolloutInsightException
from utils.config_loader import config

router = APIRouter()
templates = Jinja2Templates(directory=str(config.templates_dir))


# ---------------------------------------------------------------------------
# 延迟导入辅助
# ---------------------------------------------------------------------------


def _safe_import_attr(module_name: str, attr: str):
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return getattr(mod, attr, None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("延迟导入 {}.{} 失败: {}", module_name, attr, exc)
        return None


def _session_factory():
    return _safe_import_attr("db.db_element", "Session")


def _rollout_model():
    return _safe_import_attr("db.models", "Rollout") or _safe_import_attr("db", "Rollout")


def _task_model():
    return _safe_import_attr("db.models", "Task") or _safe_import_attr("db", "Task")


def _attempt_model():
    return _safe_import_attr("db.models", "Attempt") or _safe_import_attr("db", "Attempt")


def _segment_model():
    return _safe_import_attr("db.models", "Segment") or _safe_import_attr("db", "Segment")


def _rollout_parse_service():
    return _safe_import_attr("service.rollout_parse_service", "parse_rollout") or _safe_import_attr(
        "service.rollout_parse_service", "RolloutParseService"
    )


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------


def _rollout_to_dict(r: Any) -> Dict[str, Any]:
    if r is None:
        return {}
    return {
        "id": getattr(r, "primary_id", None) or getattr(r, "id", None),
        "primary_id": getattr(r, "primary_id", None) or getattr(r, "id", None),
        "task_id": getattr(r, "task_id", None),
        "rollout_id": getattr(r, "rollout_id", None),
        "pt_path": getattr(r, "pt_path", None),
        "pt_size_bytes": getattr(r, "pt_size_bytes", None),
        "pt_mtime": getattr(r, "pt_mtime", None),
        "mean_reward": getattr(r, "mean_reward", None),
        "n_samples": getattr(r, "n_samples", None),
        "n_attempts": getattr(r, "n_attempts", None),
        "instance_count": getattr(r, "instance_count", None),
        "scanned_at": _format_dt(getattr(r, "scanned_at", None)),
        "create_time": _format_dt(getattr(r, "create_time", None)),
    }


def _task_to_dict(t: Any) -> Dict[str, Any]:
    if t is None:
        return {}
    return {
        "id": getattr(t, "primary_id", None) or getattr(t, "id", None),
        "primary_id": getattr(t, "primary_id", None) or getattr(t, "id", None),
        "task_name": getattr(t, "task_name", None),
        "status": getattr(t, "status", None),
    }


def _format_dt(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# 聚合查询
# ---------------------------------------------------------------------------


def _load_rollout_full(rollout_pk: int) -> Dict[str, Any]:
    """拉一个 rollout 的完整信息：rollout + 任务 + attempts + sibling segments。

    Returns:
        {
            "rollout": dict,
            "task": dict | None,
            "attempts": [
                {
                    "primary_id": int,
                    "group_index": int | None,
                    "attempt_id": int,
                    "rollout_id": int,
                    "sandbox_id": str,
                    "instance_id": str,
                    "status": str,
                    "reward": float | None,
                    "segment_count": int,
                    "mean_reward": float,
                    "segments": [segment_dict, ...],   # 每个 attempt 下 segments
                },
                ...
            ],
        }
    """
    Rollout = _rollout_model()
    Task = _task_model()
    Attempt = _attempt_model()
    Segment = _segment_model()
    Session = _session_factory()
    if Rollout is None or Session is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "Rollout 模型尚未就绪",
        )

    out: Dict[str, Any] = {
        "rollout": {},
        "task": None,
        "attempts": [],
    }

    with Session() as session:
        rollout = (
            session.query(Rollout).filter(Rollout.primary_id == rollout_pk).first()
        )
        if rollout is None:
            raise RolloutInsightException(
                RolloutInsightException.TASK_NOT_FOUND,
                f"rollout #{rollout_pk} 不存在",
            )
        out["rollout"] = _rollout_to_dict(rollout)
        # task
        task = None
        if Task is not None and getattr(rollout, "task_id", None) is not None:
            task = session.query(Task).filter(Task.primary_id == rollout.task_id).first()
            out["task"] = _task_to_dict(task)
        # attempts
        if Attempt is not None:
            attempts = (
                session.query(Attempt)
                .filter(
                    Attempt.task_id == rollout.task_id,
                    Attempt.rollout_id == rollout.rollout_id,
                )
                .order_by(Attempt.group_index.asc(), Attempt.attempt_id.asc())
                .all()
            )
        else:
            attempts = []
        attempt_pks = [a.primary_id for a in attempts]
        # segments by attempt
        seg_map: Dict[int, List[Any]] = {}
        if Segment is not None and attempt_pks:
            segs = (
                session.query(Segment)
                .filter(Segment.attempt_id.in_(attempt_pks))
                .order_by(
                    Segment.attempt_id.asc(),
                    Segment.segment_index.asc(),
                )
                .all()
            )
            for s in segs:
                seg_map.setdefault(int(s.attempt_id), []).append(s)

        for a in attempts:
            segs = seg_map.get(int(a.primary_id), [])
            rewards = [
                float(s.reward) for s in segs if isinstance(getattr(s, "reward", None), (int, float))
            ]
            seg_dicts: List[Dict[str, Any]] = []
            for s in segs:
                seg_dicts.append(
                    {
                        "primary_id": int(s.primary_id),
                        "segment_index": getattr(s, "segment_index", None),
                        "segment_count": getattr(s, "segment_count", None),
                        "status": getattr(s, "status", None),
                        "reward": getattr(s, "reward", None),
                        "sandbox_id": getattr(s, "sandbox_id", None) or "",
                    }
                )
            out["attempts"].append(
                {
                    "primary_id": int(a.primary_id),
                    "group_index": getattr(a, "group_index", None),
                    "attempt_id": getattr(a, "attempt_id", None),
                    "rollout_id": getattr(a, "rollout_id", None),
                    "sandbox_id": getattr(a, "sandbox_id", None) or "",
                    "instance_id": getattr(a, "instance_id", None) or "",
                    "status": getattr(a, "status", None) or "",
                    "reward": getattr(a, "reward", None),
                    "segment_count": len(seg_dicts),
                    "mean_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
                    "segments": seg_dicts,
                }
            )
    return out


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


@router.get("/api/tasks/{task_id}/rollouts/{rollout_id}", include_in_schema=False)
async def page_rollout_detail(
    task_id: int, rollout_id: int, request: Request
) -> HTMLResponse:
    """渲染 rollout_detail.html。rollout_id 是 ``ri_rollout_table.primary_id``。"""
    Rollout = _rollout_model()
    Task = _task_model()
    Session = _session_factory()
    warning: Optional[str] = None
    rollout: Dict[str, Any] = {}
    task: Optional[Dict[str, Any]] = None
    attempts: List[Dict[str, Any]] = []

    if Rollout is None or Session is None:
        warning = "Rollout 模型尚未就绪"
    else:
        try:
            with Session() as session:
                obj = (
                    session.query(Rollout)
                    .filter(Rollout.primary_id == rollout_id)
                    .first()
                )
                if obj is None:
                    warning = f"rollout #{rollout_id} 不存在"
                else:
                    rollout = _rollout_to_dict(obj)
                    if Task is not None and getattr(obj, "task_id", None) is not None:
                        t_obj = (
                            session.query(Task)
                            .filter(Task.primary_id == obj.task_id)
                            .first()
                        )
                        task = _task_to_dict(t_obj)
                    attempts = _load_rollout_full(rollout_id).get("attempts", [])
        except Exception as exc:  # noqa: BLE001
            logger.exception("加载 rollout 详情失败 rollout_id={}: {}", rollout_id, exc)
            warning = f"加载失败: {exc}"

    return templates.TemplateResponse(
        request,
        "rollout_detail.html",
        {
            "page_title": f"Rollout #{rollout_id} - 任务 #{task_id}",
            "rollout": rollout,
            "task": task,
            "attempts": attempts,
            "warning": warning,
        },
    )


@router.get("/api/rollouts/{rollout_id}", response_model=None)
async def api_get_rollout_json(rollout_id: int) -> Response:
    """JSON 端点：返回 rollout 概览 + attempts 列表。"""
    try:
        data = _load_rollout_full(rollout_id)
    except RolloutInsightException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("api_get_rollout_json 失败 rollout_id={}: {}", rollout_id, exc)
        raise RolloutInsightException(
            RolloutInsightException.DB_ERROR, f"加载 rollout 失败: {exc}"
        ) from exc

    return Response.success(
        data={
            "pt_path": data["rollout"].get("pt_path"),
            "mean_reward": data["rollout"].get("mean_reward"),
            "n_samples": data["rollout"].get("n_samples"),
            "n_attempts": data["rollout"].get("n_attempts"),
            "instance_count": data["rollout"].get("instance_count"),
            "scanned_at": data["rollout"].get("scanned_at"),
            "attempts": [
                {
                    "primary_id": a["primary_id"],
                    "group_index": a["group_index"],
                    "segment_count": a["segment_count"],
                    "mean_reward": a["mean_reward"],
                    "sandbox_id": a["sandbox_id"],
                    "instance_id": a["instance_id"],
                    "status": a["status"],
                    "first_segment_id": (a["segments"][0]["primary_id"] if a["segments"] else None),
                }
                for a in data["attempts"]
            ],
        }
    )


@router.post("/api/rollouts/{rollout_id}/parse")
async def api_parse_rollout(
    rollout_id: int,
    background: BackgroundTasks = None,
    force: bool = Query(default=True, description="force=True 强制重解析"),
) -> Response:
    """触发 T3 重解析（默认 force=True）。"""
    Rollout = _rollout_model()
    Session = _session_factory()
    if Rollout is None or Session is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED, "Rollout 模型尚未就绪"
        )
    pt_path: Optional[str] = None
    with Session() as session:
        obj = (
            session.query(Rollout).filter(Rollout.primary_id == rollout_id).first()
        )
        if obj is None:
            raise RolloutInsightException(
                RolloutInsightException.TASK_NOT_FOUND, f"rollout #{rollout_id} 不存在"
            )
        pt_path = getattr(obj, "pt_path", None)
    if not pt_path:
        raise RolloutInsightException(
            RolloutInsightException.PARSE_DUMP_NOT_FOUND,
            f"rollout #{rollout_id} 没有 pt_path",
        )

    parse_fn = _rollout_parse_service()
    if parse_fn is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "rollout_parse_service 尚未就绪（T3 阶段未完成）",
        )

    if background is not None:
        background.add_task(_run_parse_background, parse_fn, pt_path, force)
        return Response.success(
            msg="已提交后台解析",
            data={
                "rollout_id": rollout_id,
                "pt_path": pt_path,
                "force": force,
                "status": "PENDING_PARSE",
            },
        )
    # 无 background：同步跑
    return _do_parse_sync(parse_fn, pt_path, rollout_id, force)


def _do_parse_sync(parse_fn, pt_path: str, rollout_id: int, force: bool) -> Response:
    try:
        if callable(parse_fn) and not isinstance(parse_fn, type):
            result = parse_fn(pt_path, force=force)
        else:
            inst = parse_fn() if callable(parse_fn) else parse_fn
            result = inst.parse_rollout(pt_path, force=force)
    except RolloutInsightException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("解析 rollout 失败: {}", exc)
        raise RolloutInsightException(
            RolloutInsightException.PARSE_ERROR, f"解析失败: {exc}"
        ) from exc
    return Response.success(
        msg="解析完成",
        data={
            "rollout_id": rollout_id,
            "pt_path": pt_path,
            "n_attempts": result.get("n_attempts"),
            "n_segments": result.get("n_segments"),
            "persisted_turns": result.get("persisted_turns"),
        },
    )


def _run_parse_background(parse_fn, pt_path: str, force: bool) -> None:
    """后台线程：执行 parse_rollout，仅记日志。"""
    try:
        if callable(parse_fn) and not isinstance(parse_fn, type):
            parse_fn(pt_path, force=force)
        else:
            inst = parse_fn() if callable(parse_fn) else parse_fn
            inst.parse_rollout(pt_path, force=force)
    except Exception as exc:  # noqa: BLE001
        logger.exception("后台 parse_rollout 失败 pt={}: {}", pt_path, exc)
