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
            attempt_status = str(getattr(a, "status", None) or "").split(".")[-1].lower()
            attempt_aborted = attempt_status in _ABORT_STATUS_SET
            rewards: List[float] = []
            seg_dicts: List[Dict[str, Any]] = []
            for s in segs:
                seg_status = str(getattr(s, "status", None) or "").split(".")[-1].lower()
                seg_aborted = seg_status in _ABORT_STATUS_SET
                if seg_aborted:
                    attempt_aborted = True
                raw_reward = getattr(s, "reward", None)
                # abort 上常残留 reward=0.0，展示与分桶都当作无奖励
                display_reward = None if seg_aborted else raw_reward
                if (
                    not seg_aborted
                    and isinstance(display_reward, (int, float))
                ):
                    rewards.append(float(display_reward))
                seg_dicts.append(
                    {
                        "primary_id": int(s.primary_id),
                        "segment_index": getattr(s, "segment_index", None),
                        "segment_count": getattr(s, "segment_count", None),
                        "status": getattr(s, "status", None),
                        "reward": display_reward,
                        "sandbox_id": getattr(s, "sandbox_id", None) or "",
                    }
                )
            stats = stats_cache.get(int(a.primary_id), {}) or {}
            bucket = stats.get("reward_bucket") or "none"
            if attempt_aborted:
                bucket = "aborted"
            out["attempts"].append(
                {
                    "primary_id": int(a.primary_id),
                    "group_index": getattr(a, "group_index", None),
                    "attempt_id": getattr(a, "attempt_id", None),
                    "rollout_id": getattr(a, "rollout_id", None),
                    "sandbox_id": getattr(a, "sandbox_id", None) or "",
                    "instance_id": getattr(a, "instance_id", None) or "",
                    "status": getattr(a, "status", None) or "",
                    "reward": None if attempt_aborted else getattr(a, "reward", None),
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
                    "reward_bucket": bucket,
                    "agent_turns": int(getattr(a, "agent_turns", 0) or 0),
                }
            )
    return out


def _empty_failure_clusters() -> Dict[str, Any]:
    from service.task_annotation_service import empty_failure_clusters

    return empty_failure_clusters()


def _load_pt_annotations(pt_path: Optional[str]) -> Dict[str, Any]:
    """薄封装：注解加载已下沉到 ``service.task_annotation_service``。"""
    from service.task_annotation_service import load_pt_annotations

    return load_pt_annotations(pt_path)


def _group_briefs(pt_path: Optional[str]) -> Dict[int, Dict[str, str]]:
    """每个 group 取第一条 sample 的题目名和 prompt。结果按 pt 路径缓存。"""
    return _load_pt_annotations(pt_path).get("group_briefs") or {}

def _attempt_segment_count(attempt: Dict[str, Any]) -> int:
    """单个 attempt 的 segment 数；优先用 segment_count，否则回退 len(segments)。"""
    sc = attempt.get("segment_count")
    if isinstance(sc, int):
        return sc
    segs = attempt.get("segments")
    if isinstance(segs, list):
        return len(segs)
    try:
        return int(sc or 0)
    except (TypeError, ValueError):
        return 0


_ABORT_STATUS_SET = frozenset(
    {"aborted", "abort", "cancelled", "canceled", "invalid", "failed_infra"}
)


def _group_outcome(
    all_zero: int,
    all_one: int,
    mixed: int,
    none: int,
    aborted: int,
    n_attempts: int,
) -> str:
    """组级结果标签（与 attempt 级 bucket 正交）。

    - has_abort：组内存在 abort / 不可测 attempt（优先于其它标签，UI 标红）
    - all_pass：全部 attempt 都是全1
    - all_fail：全部 attempt 都是全0
    - split：组内既有全0又有全1（跨 attempt 分化；UI 标绿）
    - seg_mixed：存在段内混合 attempt（单次 attempt 内 segment reward 不一致）
    - empty：全无数字 reward
    - other：其余组合（如仅全1+无）
    """
    if n_attempts <= 0:
        return "empty"
    if aborted > 0:
        return "has_abort"
    if none == n_attempts:
        return "empty"
    if all_one == n_attempts:
        return "all_pass"
    if all_zero == n_attempts:
        return "all_fail"
    if all_zero > 0 and all_one > 0:
        return "split"
    if mixed > 0:
        return "seg_mixed"
    return "other"


def summarize_groups(
    attempts: List[Dict[str, Any]],
    group_briefs: Optional[Dict[int, Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """按 group_index 聚合概览行。

    attempt 级计数（满足 ``全0 + 全1 + 段内混合 + 无奖励 + abort = n_attempts``）：
    - all_zero / all_one / mixed / none / aborted —— 来自每个 attempt 的 ``reward_bucket``
      （mixed = **单次 attempt 内** segment reward 不一致；aborted ≠ 全0）

    组级：
    - outcome —— ``has_abort|all_pass|all_fail|split|seg_mixed|empty|other``
    """
    briefs = group_briefs or {}
    by_gi: Dict[int, List[Dict[str, Any]]] = {}
    for a in attempts or []:
        gi = a.get("group_index")
        if gi is None:
            continue
        try:
            key = int(gi)
        except (TypeError, ValueError):
            continue
        by_gi.setdefault(key, []).append(a)

    rows: List[Dict[str, Any]] = []
    for gi in sorted(by_gi):
        items = by_gi[gi]
        azero = aone = amixed = anone = aborted = 0
        n_segments = 0
        for a in items:
            b = str(a.get("reward_bucket") or "none").lower()
            status = str(a.get("status") or "").split(".")[-1].lower()
            if b == "aborted" or status in _ABORT_STATUS_SET:
                aborted += 1
            elif b == "all_zero":
                azero += 1
            elif b == "all_one":
                aone += 1
            elif b == "mixed":
                amixed += 1
            else:
                anone += 1
            n_segments += _attempt_segment_count(a)
        brief = briefs.get(gi) or {}
        title = str(brief.get("title") or "").strip()
        if not title:
            # 回退 attempt.instance_id
            for a in items:
                inst = str(a.get("instance_id") or "").strip()
                if inst:
                    title = inst
                    break
        n_attempts = len(items)
        rows.append(
            {
                "group_index": gi,
                "title": title,
                "n_attempts": n_attempts,
                "n_segments": n_segments,
                "all_zero": azero,
                "all_one": aone,
                "mixed": amixed,
                "none": anone,
                "aborted": aborted,
                "outcome": _group_outcome(
                    azero, aone, amixed, anone, aborted, n_attempts
                ),
            }
        )
    return rows


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
    group_summaries: List[Dict[str, Any]] = []
    abort_clusters: Dict[str, Any] = {
        "total": 0,
        "n_groups_with_abort": 0,
        "by_reason": [],
        "by_group": {},
    }
    abort_clusters_view: Dict[str, Any] = abort_clusters
    failure_clusters: Dict[str, Any] = _empty_failure_clusters()
    failure_clusters_view: Dict[str, Any] = failure_clusters

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
                    annotations = _load_pt_annotations(rollout.get("pt_path"))
                    group_briefs = annotations.get("group_briefs") or {}
                    abort_clusters = annotations.get("abort_clusters") or abort_clusters
                    failure_clusters = (
                        annotations.get("failure_clusters") or failure_clusters
                    )
                    outcome_index = annotations.get("attempt_outcomes") or {}
                    for a in attempts:
                        gi = a.get("group_index")
                        aid = a.get("attempt_id")
                        if isinstance(gi, int) and aid is not None:
                            try:
                                info = outcome_index.get((int(gi), int(aid))) or {}
                            except (TypeError, ValueError):
                                info = {}
                            if info:
                                a["abort_reason"] = info.get("abort_reason")
                                a["outcome_category"] = info.get("outcome_category")
                                a["outcome_reason"] = info.get("outcome_reason")
                                a["outcome_label"] = info.get("outcome_label")
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
                    group_summaries = summarize_groups(attempts, group_briefs)
                    # 对齐任务页 rollout 列表：每组补 通过/abort/超时/正常失败/有效/mean_reward
                    try:
                        from db.db_attempt_service import DBAttemptService

                        biz_rid = rollout.get("rollout_id")
                        tid = rollout.get("task_id")
                        if tid is not None and biz_rid is not None:
                            g_metrics = DBAttemptService.summarize_reward_metrics_by_group(
                                int(tid), int(biz_rid)
                            )
                            for gs in group_summaries:
                                gi = gs.get("group_index")
                                m = g_metrics.get(int(gi), {}) if gi is not None else {}
                                gs["n_passed"] = int(m.get("n_passed", 0))
                                gs["n_aborted"] = int(m.get("n_aborted", 0))
                                gs["n_timeout"] = int(m.get("n_timeout", 0))
                                gs["n_failed"] = int(m.get("n_failed", 0))
                                gs["n_other"] = int(m.get("n_other", 0))
                                gs["is_valid"] = bool(m.get("is_valid", False))
                                gs["mean_reward"] = m.get("mean_reward")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "聚合 group reward 指标失败 rollout_pk={}: {}",
                            rollout_id,
                            exc,
                        )
                    # 组过滤时：柱状图只展示该组失败分布
                    if group is not None:
                        gi = int(group)
                        g_info = (abort_clusters.get("by_group") or {}).get(gi) or {
                            "total": 0,
                            "by_reason": [],
                        }
                        abort_clusters_view = {
                            "total": int(g_info.get("total") or 0),
                            "n_groups_with_abort": 1 if g_info.get("total") else 0,
                            "by_reason": g_info.get("by_reason") or [],
                            "by_group": {gi: g_info},
                            "scope": "group",
                            "group_index": gi,
                        }
                        f_info = (failure_clusters.get("by_group") or {}).get(gi) or {
                            "n_attempts": 0,
                            "n_pass": 0,
                            "n_abort": 0,
                            "n_timeout": 0,
                            "n_fail": 0,
                            "n_other": 0,
                            "n_failure": 0,
                            "by_category": [],
                            "by_reason": [],
                        }
                        failure_clusters_view = {
                            **f_info,
                            "n_groups_with_abort": 1
                            if int(f_info.get("n_abort") or 0)
                            else 0,
                            "n_groups_with_failure": 1
                            if int(f_info.get("n_failure") or 0)
                            else 0,
                            "by_group": {gi: f_info},
                            "group_trend": [
                                {
                                    "group_index": gi,
                                    "n_attempts": int(f_info.get("n_attempts") or 0),
                                    "n_pass": int(f_info.get("n_pass") or 0),
                                    "n_abort": int(f_info.get("n_abort") or 0),
                                    "n_timeout": int(f_info.get("n_timeout") or 0),
                                    "n_fail": int(f_info.get("n_fail") or 0),
                                    "n_other": int(f_info.get("n_other") or 0),
                                    "n_failure": int(f_info.get("n_failure") or 0),
                                    "fail_rate_pct": round(
                                        100.0
                                        * int(f_info.get("n_failure") or 0)
                                        / max(int(f_info.get("n_attempts") or 0), 1),
                                        1,
                                    ),
                                }
                            ],
                            "scope": "group",
                            "group_index": gi,
                        }
                    else:
                        abort_clusters_view = dict(abort_clusters)
                        abort_clusters_view["scope"] = "rollout"
                        failure_clusters_view = dict(failure_clusters)
                        failure_clusters_view["scope"] = "rollout"
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
            "group_summaries": group_summaries,
            "abort_clusters": abort_clusters_view,
            "failure_clusters": failure_clusters_view,
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
            "group_summaries": summarize_groups(all_attempts),
            "abort_clusters": _load_pt_annotations(
                data["rollout"].get("pt_path")
            ).get("abort_clusters"),
            "failure_clusters": _load_pt_annotations(
                data["rollout"].get("pt_path")
            ).get("failure_clusters"),
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
