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

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from starlette.templating import Jinja2Templates

from data.response import Response
from data.rollout_insight_exception import RolloutInsightException
from utils.auth_session import require_login
from utils.config_loader import config

# v0.5.1：业务路由统一要求登录
router = APIRouter(dependencies=[Depends(require_login)])
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
    return _safe_import_attr("db", "Rollout") or _safe_import_attr("db.models", "Rollout")


def _task_model():
    return _safe_import_attr("db", "Task") or _safe_import_attr("db.models", "Task")


def _attempt_model():
    return _safe_import_attr("db", "Attempt") or _safe_import_attr("db.models", "Attempt")


def _segment_model():
    return _safe_import_attr("db", "Segment") or _safe_import_attr("db.models", "Segment")


def _rollout_parse_service():
    return _safe_import_attr("service.rollout_parse_service", "parse_rollout") or _safe_import_attr(
        "service.rollout_parse_service", "RolloutParseService"
    )


def _attempt_token_stats_fn():
    """v0.5.0 T2：取 service 层的 attempt 聚合函数（延迟导入）。"""
    return _safe_import_attr(
        "service.rollout_parse_service", "parse_attempt_token_stats"
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

        # v0.5.0 T2：批量聚合 attempt 摘要（避免每行单独 query）
        stats_fn = _attempt_token_stats_fn()
        stats_cache: Dict[int, Dict[str, Any]] = {}
        if stats_fn is not None:
            for a in attempts:
                try:
                    s = stats_fn(int(a.primary_id)) or {}
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "_load_rollout_full: attempt_pk={} 聚合失败: {}",
                        a.primary_id,
                        exc,
                    )
                    s = {}
                stats_cache[int(a.primary_id)] = s

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
            stats = stats_cache.get(int(a.primary_id), {}) or {}
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
                    # v0.5.0 T2：attempt 维度聚合
                    "n_tool_calls": int(stats.get("n_tool_calls", 0) or 0),
                    "n_diff_lines": int(stats.get("n_diff_lines", 0) or 0),
                    "total_resp_len": int(stats.get("total_resp_len", 0) or 0),
                    "total_mask_sum": int(stats.get("total_mask_sum", 0) or 0),
                    "total_trainable_tokens": int(
                        stats.get("total_trainable_tokens", 0) or 0
                    ),
                    "reward_bucket": stats.get("reward_bucket") or "none",
                    "agent_turns": int(getattr(a, "agent_turns", 0) or 0),
                }
            )
    return out


_GROUP_BRIEF_CACHE: Dict[str, Dict[int, Dict[str, str]]] = {}


def _prompt_text_of(sample: Dict[str, Any]) -> str:
    prompt = sample.get("prompt") or ""
    if isinstance(prompt, list):
        parts: List[str] = []
        for item in prompt:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    if isinstance(prompt, dict):
        return str(prompt.get("content") or prompt.get("text") or "").strip()
    return str(prompt or "").strip()


def _group_briefs(pt_path: Optional[str]) -> Dict[int, Dict[str, str]]:
    """每个 group 取第一条 sample 的题目名和 prompt。结果按 pt 路径缓存。"""
    if not pt_path:
        return {}
    cached = _GROUP_BRIEF_CACHE.get(pt_path)
    if cached is not None:
        return cached
    briefs: Dict[int, Dict[str, str]] = {}
    try:
        from utils.rollout_pt_parser import load_pt

        dump = load_pt(pt_path)
        for sample in dump.get("samples") or []:
            if not isinstance(sample, dict):
                continue
            gi = sample.get("group_index")
            if not isinstance(gi, int) or gi in briefs:
                continue
            md = sample.get("metadata") or {}
            title = str(md.get("instance_id") or sample.get("label") or "").strip()
            briefs[gi] = {
                "title": title,
                "prompt": _prompt_text_of(sample),
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取 group prompt 失败 pt={}: {}", pt_path, exc)
        briefs = {}
    _GROUP_BRIEF_CACHE[pt_path] = briefs
    return briefs


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


@router.get("/api/tasks/{task_id}/rollouts/{rollout_id}", include_in_schema=False)
async def page_rollout_detail(
    task_id: int, rollout_id: int, request: Request, group: Optional[int] = None
) -> HTMLResponse:
    """渲染 rollout_detail.html。rollout_id 是 ``ri_rollout_table.primary_id``。

    v0.5.0 T4 新增 ``?group=N`` 过滤：第一层手风琴按 group 折完后，
    当 URL 带 group 参数时，只渲染该 group 的 attempts。
    """
    Rollout = _rollout_model()
    Task = _task_model()
    Session = _session_factory()
    warning: Optional[str] = None
    rollout: Dict[str, Any] = {}
    task: Optional[Dict[str, Any]] = None
    attempts: List[Dict[str, Any]] = []
    filtered_attempts: List[Dict[str, Any]] = []
    distinct_groups: List[int] = []
    group_briefs: Dict[int, Dict[str, str]] = {}

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
                    distinct_groups = sorted(
                        {
                            int(a["group_index"])
                            for a in attempts
                            if a.get("group_index") is not None
                        }
                    )
                    if group is not None:
                        filtered_attempts = [
                            a for a in attempts if a.get("group_index") == int(group)
                        ]
                        if not filtered_attempts:
                            warning = (
                                f"rollout #{rollout_id} 中未找到 group={group} 的 attempt"
                            )
                    else:
                        filtered_attempts = attempts
                    group_briefs = _group_briefs(rollout.get("pt_path"))
                    for a in attempts:
                        gi = a.get("group_index")
                        brief = group_briefs.get(gi) if isinstance(gi, int) else None
                        if brief and not brief.get("title"):
                            brief["title"] = a.get("instance_id") or ""
                        elif isinstance(gi, int) and gi not in group_briefs and a.get("instance_id"):
                            group_briefs[gi] = {
                                "title": a.get("instance_id") or "",
                                "prompt": "",
                            }
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
            "attempts": filtered_attempts,
            "all_attempts": attempts,
            "distinct_groups": distinct_groups,
            "current_group": group,
            "group_briefs": group_briefs,
            "warning": warning,
        },
    )


@router.get("/api/rollouts/{rollout_id}", response_model=None)
async def api_get_rollout_json(
    rollout_id: int, group: Optional[int] = Query(default=None, description="按 group_index 过滤 attempts")
) -> Response:
    """JSON 端点：返回 rollout 概览 + attempts 列表。

    v0.5.0 T4 新增 ``?group=N`` 过滤：仅返回该 group 的 attempts。
    """
    try:
        data = _load_rollout_full(rollout_id)
    except RolloutInsightException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("api_get_rollout_json 失败 rollout_id={}: {}", rollout_id, exc)
        raise RolloutInsightException(
            RolloutInsightException.DB_ERROR, f"加载 rollout 失败: {exc}"
        ) from exc

    all_attempts = data["attempts"]
    if group is not None:
        filtered = [a for a in all_attempts if a.get("group_index") == int(group)]
    else:
        filtered = all_attempts

    return Response.success(
        data={
            "pt_path": data["rollout"].get("pt_path"),
            "mean_reward": data["rollout"].get("mean_reward"),
            "n_samples": data["rollout"].get("n_samples"),
            "n_attempts": data["rollout"].get("n_attempts"),
            "instance_count": data["rollout"].get("instance_count"),
            "scanned_at": data["rollout"].get("scanned_at"),
            "group_filter": group,
            "distinct_groups": sorted(
                {
                    int(a["group_index"])
                    for a in all_attempts
                    if a.get("group_index") is not None
                }
            ),
            "attempts": [
                {
                    "primary_id": a["primary_id"],
                    "group_index": a["group_index"],
                    "attempt_id": a.get("attempt_id"),
                    "segment_count": a["segment_count"],
                    "mean_reward": a["mean_reward"],
                    "sandbox_id": a["sandbox_id"],
                    "instance_id": a["instance_id"],
                    "status": a["status"],
                    "first_segment_id": (a["segments"][0]["primary_id"] if a["segments"] else None),
                    # v0.5.0 T2
                    "n_tool_calls": a.get("n_tool_calls", 0),
                    "n_diff_lines": a.get("n_diff_lines", 0),
                    "total_resp_len": a.get("total_resp_len", 0),
                    "total_mask_sum": a.get("total_mask_sum", 0),
                    "total_trainable_tokens": a.get("total_trainable_tokens", 0),
                    "reward_bucket": a.get("reward_bucket", "none"),
                }
                for a in filtered
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
