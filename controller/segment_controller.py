"""Segment 控制器（v0.2.0 阶段 2C）。

提供：
- GET  /api/segments/{id}                       渲染 segment_transcript.html
- GET  /api/segments/search                    JSON 列表（详细逻辑由 2D 写）
- GET  /api/segments/{id}/transcript.json      调 SegmentService.ensure_transcript 后返回 JSON

设计：
- 业务错误用 RolloutInsightException 抛出
- 参数错误用 Response.error(code=50001, msg=...) 直接返回
- 所有对 db/service 的导入在函数内**延迟**进行，避免 2A/2B worker 还没提交时崩
"""
from __future__ import annotations

import json
import re
from html import escape as _html_escape
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Request
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


def _segment_model():
    return _safe_import_attr("db", "Segment") or _safe_import_attr("db.models", "Segment")


def _attempt_model():
    return _safe_import_attr("db", "Attempt") or _safe_import_attr("db.models", "Attempt")


def _rollout_model():
    return _safe_import_attr("db", "Rollout") or _safe_import_attr("db.models", "Rollout")


def _task_model():
    return _safe_import_attr("db", "Task") or _safe_import_attr("db.models", "Task")


def _ensure_transcript_fn():
    """直接拿 2D 的 ensure_transcript 函数。"""
    return _safe_import_attr("service.segment_transcript_service", "ensure_transcript")


# ---------------------------------------------------------------------------
# Turn 渲染辅助
# ---------------------------------------------------------------------------

_MARKER_START_RE = re.compile(r"<\|im_start\|>")
_MARKER_END_RE = re.compile(r"<\|im_end\|>")
_TOOL_CALL_RE = re.compile(r"<\|tool_call(?:[^>]*)?>", re.IGNORECASE)
_TOOL_CALL_END_RE = re.compile(r"</tool_call>", re.IGNORECASE)
_TOOL_RESULT_RE = re.compile(r"<\|tool_response(?:[^>]*)?>")
_TOOL_RESULT_END_RE = re.compile(r"</tool_response>", re.IGNORECASE)


def _split_by_im_start(text: str) -> List[Dict[str, str]]:
    """把整段 transcript 文本按 <|im_start|>role\\n...<|im_end|> 切分成 turn。

    返回 [{role, body}, ...]；前导（首个 im_start 之前的）当作 role=unknown。
    """
    if not text:
        return []

    parts = _MARKER_START_RE.split(text)
    # parts[0] = 第一个 im_start 之前的内容（可能为空）
    turns: List[Dict[str, str]] = []
    if parts and parts[0].strip():
        turns.append({"role": "unknown", "body": parts[0]})

    for chunk in parts[1:]:
        # chunk 形如 "role\\nxxxxx<|im_end|>xxxx<|im_start|>..."
        if not chunk:
            continue
        # 取第一行作为 role
        newline_idx = chunk.find("\n")
        if newline_idx < 0:
            role = chunk.strip()
            body = ""
        else:
            role = chunk[:newline_idx].strip()
            body = chunk[newline_idx + 1 :]
        # 截断到下一个 im_end
        end_idx = body.find("<|im_end|>")
        if end_idx >= 0:
            body = body[:end_idx]
        # role 归一化
        role_lc = role.lower().strip()
        if role_lc in ("system",):
            label = "system"
        elif role_lc in ("user", "human"):
            label = "user"
        elif role_lc in ("assistant", "gpt", "model"):
            label = "assistant"
        elif role_lc in ("tool", "tool_response", "toolresponse"):
            label = "tool"
        else:
            label = role_lc or "unknown"
        turns.append({"role": role, "body": body, "role_label": label})
    return turns


def _highlight_html(body: str) -> str:
    """对 turn body 做轻量高亮，保留 \\n / 空格 / 缩进。

    - <|im_start|> / <|im_end|> 标黄/标绿
    - <|tool_call ...> / </tool_call> 标蓝
    - <|tool_response ...> / </tool_response> 标橙
    - 其余转义为 HTML 文本
    """
    if not body:
        return ""

    # 先做转义
    s = _html_escape(body, quote=False)
    # im_start
    s = _MARKER_START_RE.sub(
        lambda m: '<span class="marker-start">&lt;|im_start|&gt;</span>', s
    )
    s = _MARKER_END_RE.sub(
        lambda m: '<span class="marker-end">&lt;|im_end|&gt;</span>', s
    )
    # tool_call（成对）
    s = _TOOL_CALL_RE.sub(
        lambda m: '<span class="tool-call">&lt;|tool_call&gt;</span>', s
    )
    s = _TOOL_CALL_END_RE.sub(
        lambda m: '<span class="tool-call">&lt;/tool_call&gt;</span>', s
    )
    # tool_response
    s = _TOOL_RESULT_RE.sub(
        lambda m: '<span class="tool-result">&lt;|tool_response&gt;</span>', s
    )
    s = _TOOL_RESULT_END_RE.sub(
        lambda m: '<span class="tool-result">&lt;/tool_response&gt;</span>', s
    )
    return s


def _build_turns_html(transcript_text: str) -> List[Dict[str, Any]]:
    """把 transcript 文本切成 turn，并预渲染高亮 HTML。"""
    raw_turns = _split_by_im_start(transcript_text)
    out: List[Dict[str, Any]] = []
    for t in raw_turns:
        body = t.get("body", "")
        out.append({
            "role": t.get("role", ""),
            "role_label": t.get("role_label", "unknown"),
            "role_class": t.get("role_label", "unknown"),
            "html": _highlight_html(body),
            "char_count": len(body),
        })
    return out


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------

def _segment_to_dict(s: Any, attempt: Any = None) -> Dict[str, Any]:
    """Segment ORM → 模板/JSON 友好字典。

    Args:
        s: DBSegment ORM 对象
        attempt: 可选，对应的 DBAttempt（用于补 instance_id / agent_turns /
                 tool_calls / diff_line_count / pt_path 等 segment 表没有的字段）
    """
    if s is None:
        return {}
    seg_dict = {
        "id": getattr(s, "primary_id", None) or getattr(s, "id", None),
        "primary_id": getattr(s, "primary_id", None) or getattr(s, "id", None),
        "attempt_id": getattr(s, "attempt_id", None),
        "rollout_id": getattr(s, "rollout_id", None),
        "group_index": getattr(s, "group_index", None),
        "instance_id": getattr(attempt, "instance_id", None) if attempt else None,
        "segment_index": getattr(s, "segment_index", None),
        "segment_count": getattr(s, "segment_count", None),
        "sandbox_id": getattr(s, "sandbox_id", None),
        "status": _status_str(getattr(s, "status", None)),
        "reward": getattr(s, "reward", None),
        "agent_turns": getattr(attempt, "agent_turns", None) if attempt else None,
        "tool_calls": getattr(attempt, "tool_calls", None) if attempt else None,
        "diff_line_count": getattr(attempt, "diff_line_count", None) if attempt else None,
        "response_length": getattr(s, "response_length", None),
        "pt_path": None,
    }
    return seg_dict


def _status_str(value: Any) -> Optional[str]:
    """把 status 枚举转字符串。"""
    if value is None:
        return None
    s = str(value)
    if "." in s:
        return s.split(".")[-1].lower()
    return s.lower()


# segment 模板字段全集（保证占位 dict 有所有 key，避免模板 `segment.X is not none` 误判）
_SEGMENT_PLACEHOLDER_KEYS: tuple = (
    "id", "attempt_id", "rollout_id", "group_index", "instance_id",
    "segment_index", "segment_count", "sandbox_id", "status",
    "reward", "agent_turns", "tool_calls", "response_length",
    "diff_line_count", "pt_path",
)


def _empty_segment_dict(segment_id: Optional[int] = None) -> Dict[str, Any]:
    return {k: None for k in _SEGMENT_PLACEHOLDER_KEYS} | {
        "id": segment_id,
        "status": "UNKNOWN",
    }


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@router.get("/api/segments/search", response_model=None)
async def api_segments_search(
    request: Request,
    task_id: Optional[int] = Query(default=None, description="按任务过滤"),
    min_reward: Optional[float] = Query(default=None, description="reward 下限"),
    keyword: Optional[str] = Query(default=None, description="关键字（命中 segment_transcript.response_text）"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
):
    """Segment 检索：调 SegmentSearchService.search_segments。

    支持 (task_id, min_reward, keyword) 三维过滤，keyword 命中后会回写
    ri_segment_transcript_table 的 full_text_fetched=1，下次秒回。
    """
    try:
        from service.segment_search_service import search_segments
    except Exception as exc:  # noqa: BLE001
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            f"segment_search_service 不可用: {exc}",
        ) from exc
    try:
        offset = (page - 1) * page_size
        result = search_segments(
            task_id=task_id,
            min_reward=float(min_reward or 0.0),
            keyword=keyword or "",
            limit=page_size,
            offset=offset,
        )
        return Response.success(
            data={
                "segments": result.get("items", []),
                "total": result.get("total", 0),
                "page": page,
                "page_size": page_size,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("segment 检索失败: {}", exc)
        raise RolloutInsightException(
            RolloutInsightException.DB_ERROR, f"segment 检索失败: {exc}"
        ) from exc


@router.get("/api/segments/{segment_id}", include_in_schema=False)
async def page_segment_transcript(segment_id: int, request: Request) -> HTMLResponse:
    """渲染 segment_transcript.html（v0.4.0 T5：注入 turns 列表 + 兄弟 segment 导航）。"""
    Segment = _segment_model()
    Attempt = _attempt_model()
    Rollout = _rollout_model()
    Task = _task_model()
    Session = _session_factory()

    warning: Optional[str] = None
    transcript_text: str = ""
    turns: List[Dict[str, Any]] = []
    seg: Dict[str, Any] = _empty_segment_dict(segment_id)
    task_id: Optional[int] = None
    task_name: Optional[str] = None
    rollout_biz_id: Optional[int] = None
    rollout_pk: Optional[int] = None
    sibling_prev_id: Optional[int] = None
    sibling_next_id: Optional[int] = None
    turn_stats: Dict[str, Any] = {
        "turn_count": 0,
        "total_chars": 0,
        "token_estimate": 0,
        "time_cost": None,
    }

    if Segment is None or Session is None:
        warning = "Segment 模型尚未就绪（2A 阶段未完成）"
    else:
        try:
            with Session() as session:
                obj = session.query(Segment).filter(Segment.primary_id == segment_id).first()
                if obj is None:
                    warning = f"segment #{segment_id} 不存在"
                else:
                    # 找 attempt 和 rollout（用于补 instance_id / pt_path / task_id）
                    attempt_obj = None
                    rollout_obj = None
                    if (
                        Attempt is not None
                        and getattr(obj, "attempt_id", None) is not None
                    ):
                        attempt_obj = session.query(Attempt).filter(Attempt.primary_id == obj.attempt_id).first()
                    if (
                        attempt_obj is not None
                        and Rollout is not None
                        and getattr(attempt_obj, "rollout_id", None) is not None
                    ):
                        rollout_obj = (
                            session.query(Rollout)
                            .filter(Rollout.primary_id == attempt_obj.rollout_id)
                            .first()
                        )
                    seg = _segment_to_dict(obj, attempt_obj)
                    if rollout_obj is not None:
                        seg["rollout_id"] = rollout_obj.primary_id
                        seg["pt_path"] = getattr(rollout_obj, "pt_path", None)
                        rollout_pk = getattr(rollout_obj, "primary_id", None)
                        rollout_biz_id = getattr(rollout_obj, "rollout_id", None)
                        if getattr(rollout_obj, "task_id", None) is not None:
                            task_id = rollout_obj.task_id
                    if task_id is not None and Task is not None:
                        t_obj = session.query(Task).filter(Task.primary_id == task_id).first()
                        if t_obj is not None:
                            task_name = getattr(t_obj, "task_name", None)

                    # v0.4.0 T5：先从 ri_segment_turn_table 读 turns
                    turns = _load_turns_for_segment(segment_id, session)
                    if not turns:
                        # 没有 turns：触发一次 ensure_transcript（懒加载 .pt）
                        # 然后调 T3 split_response_to_turns 落库到 turn 表
                        transcript_text = _ensure_turns_for_segment(segment_id, obj, seg)
                        if transcript_text:
                            turns = _load_turns_for_segment(segment_id, session)
                        if not turns:
                            # 兜底：从 raw transcript 切分（仅用于显示，不落库）
                            try:
                                ensure_fn = _ensure_transcript_fn()
                                if callable(ensure_fn):
                                    r = ensure_fn(segment_id)
                                    if isinstance(r, dict):
                                        transcript_text = (
                                            r.get("transcript")
                                            or r.get("text")
                                            or r.get("response_text")
                                            or r.get("prompt_text")
                                            or ""
                                        )
                            except Exception as exc:  # noqa: BLE001
                                logger.warning("ensure_transcript 失败 segment_id={}: {}", segment_id, exc)
                            if not transcript_text:
                                transcript_text = _load_transcript_text(obj, seg)
                            if transcript_text:
                                turns = _build_turns_html(transcript_text)
                            else:
                                warning = (
                                    "未找到 transcript 文本（可能为 2D 阶段尚未补 ensure_transcript，"
                                    "或未提供 tokenizer 目录）"
                                )

                    # 统计 turn 数据 + 兄弟 segment 导航
                    if turns:
                        turn_stats["turn_count"] = len(turns)
                        for t in turns:
                            turn_stats["total_chars"] += int(t.get("body_char_count") or t.get("char_count") or 0)
                            turn_stats["token_estimate"] += int(t.get("token_estimate") or 0)
                    turn_stats["time_cost"] = _extract_time_cost(seg, obj)

                    # 兄弟 segment（按 segment_index 排序的 prev/next）
                    if attempt_obj is not None:
                        sibling_prev_id, sibling_next_id = _sibling_nav(
                            session, attempt_obj.primary_id, obj.segment_index
                        )
        except Exception as exc:  # noqa: BLE001
            logger.exception("加载 segment #{} 失败: {}", segment_id, exc)
            warning = f"加载失败：{exc}"

    return templates.TemplateResponse(
        request,
        "segment_transcript.html",
        {
            "page_title": f"Segment #{segment_id}",
            "segment": seg,
            "turns": turns,
            "task_id": task_id,
            "task_name": task_name,
            "rollout_biz_id": rollout_biz_id,
            "rollout_pk": rollout_pk,
            "sibling_prev_id": sibling_prev_id,
            "sibling_next_id": sibling_next_id,
            "turn_stats": turn_stats,
            "warning": warning,
        },
    )


# ---------------------------------------------------------------------------
# T5 辅助
# ---------------------------------------------------------------------------


def _load_turns_for_segment(segment_id: int, session: Any) -> List[Dict[str, Any]]:
    """从 ``ri_segment_turn_table`` 读 turns；若空返回 []。"""
    try:
        from db.db_segment_turn import DBSegmentTurn
    except Exception as exc:  # noqa: BLE001
        logger.warning("_load_turns_for_segment: DBSegmentTurn 不可用: {}", exc)
        return []
    try:
        rows = (
            session.query(DBSegmentTurn)
            .filter(DBSegmentTurn.segment_id == segment_id)
            .order_by(DBSegmentTurn.turn_index.asc())
            .all()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("_load_turns_for_segment: 查询失败: {}", exc)
        return []
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            tool_calls = _safe_json_load(getattr(r, "tool_call_names_json", "[]"), default=[])
        except Exception:
            tool_calls = []
        if not isinstance(tool_calls, list):
            tool_calls = []
        body_text = r.body_text or ""
        out.append(
            {
                "primary_id": r.primary_id,
                "turn_index": r.turn_index,
                "role": r.role or "unknown",
                "role_label": r.role or "unknown",
                "role_class": r.role or "unknown",
                "body_text": body_text,
                "body_char_count": int(r.body_char_count or 0),
                "has_tool_call": bool(r.has_tool_call),
                "has_think": bool(r.has_think),
                "tool_call_names": [str(x) for x in tool_calls if x],
                "token_estimate": int(r.token_estimate or 0),
                "html": _highlight_html(body_text),
            }
        )
    return out


def _ensure_turns_for_segment(segment_id: int, seg_obj: Any, seg_dict: Dict[str, Any]) -> str:
    """懒加载：调 T3 split_response_to_turns，把 turn 写到 ``ri_segment_turn_table``。

    Returns:
        response_text（写入前的内容），用于诊断。
    """
    try:
        from db.db_segment_turn_service import DBSegmentTurnService
    except Exception as exc:  # noqa: BLE001
        logger.warning("_ensure_turns_for_segment: DBSegmentTurnService 不可用: {}", exc)
        return ""
    # 先 ensure_transcript 让 transcript 行落库
    response_text = ""
    try:
        ensure_fn = _ensure_transcript_fn()
        if callable(ensure_fn):
            r = ensure_fn(segment_id)
            if isinstance(r, dict):
                response_text = (
                    r.get("transcript")
                    or r.get("text")
                    or r.get("response_text")
                    or ""
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure_transcript 失败 segment_id={}: {}", segment_id, exc)
    if not response_text:
        response_text = _load_transcript_text(seg_obj, seg_dict)
    if not response_text:
        return ""

    try:
        from utils.rollout_pt_parser import split_response_to_turns
    except Exception as exc:  # noqa: BLE001
        logger.warning("import split_response_to_turns 失败: {}", exc)
        return response_text
    turns = split_response_to_turns(response_text)
    if not turns:
        return response_text
    try:
        DBSegmentTurnService.upsert_segment_turns(
            segment_id=int(segment_id), turns=turns, force=True
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("upsert_segment_turns 失败 segment_id={}: {}", segment_id, exc)
    return response_text


def _extract_time_cost(seg_dict: Dict[str, Any], seg_obj: Any) -> Optional[float]:
    """从 attempt / segment 元数据抽 time_cost（秒）。"""
    for candidate in (seg_dict, seg_obj):
        pt = None
        if hasattr(candidate, "phase_time"):
            pt = getattr(candidate, "phase_time", None)
        elif isinstance(candidate, dict):
            pt = candidate.get("phase_time")
        if isinstance(pt, dict):
            for key in ("agent_run", "rollout_total", "total", "agent"):
                v = pt.get(key)
                if isinstance(v, (int, float)):
                    return float(v)
            total = sum(v for v in pt.values() if isinstance(v, (int, float)))
            if total > 0:
                return float(total)
    return None


def _sibling_nav(session: Any, attempt_id: int, current_index: Any) -> tuple[Optional[int], Optional[int]]:
    """返回 (prev_segment_id, next_segment_id)，按 segment_index 排序。"""
    try:
        from db.db_segment import DBSegment
    except Exception:
        return None, None
    prev_id: Optional[int] = None
    next_id: Optional[int] = None
    if current_index is None:
        return None, None
    try:
        prev_row = (
            session.query(DBSegment)
            .filter(
                DBSegment.attempt_id == attempt_id,
                DBSegment.segment_index < current_index,
            )
            .order_by(DBSegment.segment_index.desc())
            .first()
        )
        next_row = (
            session.query(DBSegment)
            .filter(
                DBSegment.attempt_id == attempt_id,
                DBSegment.segment_index > current_index,
            )
            .order_by(DBSegment.segment_index.asc())
            .first()
        )
        if prev_row is not None:
            prev_id = int(prev_row.primary_id)
        if next_row is not None:
            next_id = int(next_row.primary_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("_sibling_nav 失败: {}", exc)
    return prev_id, next_id


def _safe_json_load(raw: Any, default: Any) -> Any:
    import json as _json

    if raw is None:
        return default
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return _json.loads(raw)
    except (TypeError, ValueError):
        return default


@router.get("/api/segments/{segment_id}/transcript.json")
async def api_segment_transcript_json(segment_id: int) -> Response:
    """调 SegmentService.ensure_transcript 后返回 JSON。

    返回结构：
        {
          "id": segment_id,
          "transcript": "<整段 transcript 文本>",
          "turns": [{"role": "...", "body": "..."}],
          "meta": {...}
        }
    """
    Segment = _segment_model()
    Attempt = _attempt_model()
    Rollout = _rollout_model()
    Session = _session_factory()
    if Segment is None or Session is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "Segment 模型尚未就绪（2A 阶段未完成）",
        )

    with Session() as session:
        obj = session.query(Segment).filter(Segment.primary_id == segment_id).first()
        if obj is None:
            raise RolloutInsightException(
                RolloutInsightException.PARSE_DUMP_NOT_FOUND,
                f"segment #{segment_id} 不存在",
            )
        # 补 attempt 字段（instance_id / agent_turns / tool_calls / pt_path 等）
        attempt_obj = None
        if Attempt is not None and getattr(obj, "attempt_id", None) is not None:
            attempt_obj = (
                session.query(Attempt)
                .filter(Attempt.primary_id == obj.attempt_id)
                .first()
            )
        seg = _segment_to_dict(obj, attempt_obj)
        if attempt_obj is not None and Rollout is not None:
            seg["rollout_id"] = getattr(attempt_obj, "rollout_id", None)
            ro = (
                session.query(Rollout)
                .filter(
                    Rollout.task_id == attempt_obj.task_id,
                    Rollout.rollout_id == attempt_obj.rollout_id,
                )
                .first()
            )
            if ro is not None:
                seg["pt_path"] = getattr(ro, "pt_path", None)

    # 尝试走 SegmentTranscriptService.ensure_transcript（2D 阶段补齐）
    transcript_text: Optional[str] = None
    used_service = False
    try:
        ensure_fn = _ensure_transcript_fn()
        if callable(ensure_fn):
            result = ensure_fn(segment_id)
            # 约定：返回 dict {prompt_text, response_text, turns_json, ...} 或 str
            if isinstance(result, dict):
                transcript_text = (
                    result.get("transcript")
                    or result.get("text")
                    or result.get("response_text")
                    or result.get("prompt_text")
                    or ""
                )
            elif isinstance(result, str):
                transcript_text = result
            used_service = bool(transcript_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure_transcript 调用失败: {}", exc)

    # 兜底：直接读 segment 上的 transcript / response 字段
    if not transcript_text:
        transcript_text = _load_transcript_text(obj, seg)

    turns = _split_by_im_start(transcript_text or "")
    return Response.success(
        msg="ok" if transcript_text else "transcript 为空",
        data={
            "id": segment_id,
            "transcript": transcript_text or "",
            "turns": turns,
            "meta": seg,
            "source": "service" if used_service else "fallback",
        },
    )


@router.get("/api/segments/{segment_id}/turns.json", response_model=None)
async def api_segment_turns_json(
    segment_id: int,
    pt_path: Optional[str] = None,
) -> Response:
    """按需返回某个 segment 的 turn 列表（供 rollout 详情页懒加载）。

    v0.5.0 T4：优先用 ``pt_path`` + tokenizer 切分（``split_response_to_turns_with_tokenizer``）；
    若 tokens 拿不到或 tokenizer 不可用，回退到 ``ri_segment_turn_table``；
    再回退到 ``ensure_transcript`` + 文本切分。

    如果 query string 带 ``?pt_path=...``，会优先用该 .pt 的 tokens 重新切分；
    否则从 ``ensure_transcript`` 拿的 sample tokens 切。
    """
    Segment = _segment_model()
    Attempt = _attempt_model()
    Rollout = _rollout_model()
    Session = _session_factory()
    if Segment is None or Session is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED, "Segment 模型尚未就绪"
        )

    # 拿 seg + 找 pt_path
    pt_path_resolved: Optional[str] = pt_path or None
    try:
        with Session() as session:
            seg_obj = (
                session.query(Segment)
                .filter(Segment.primary_id == segment_id)
                .first()
            )
            if seg_obj is None:
                raise RolloutInsightException(
                    RolloutInsightException.PARSE_DUMP_NOT_FOUND,
                    f"segment #{segment_id} 不存在",
                )
            if not pt_path_resolved and Attempt is not None and getattr(seg_obj, "attempt_id", None):
                att = (
                    session.query(Attempt)
                    .filter(Attempt.primary_id == seg_obj.attempt_id)
                    .first()
                )
                if att is not None and Rollout is not None:
                    ro = (
                        session.query(Rollout)
                        .filter(
                            Rollout.task_id == att.task_id,
                            Rollout.rollout_id == att.rollout_id,
                        )
                        .order_by(Rollout.scanned_at.desc())
                        .first()
                    )
                    if ro is not None:
                        pt_path_resolved = getattr(ro, "pt_path", None)
    except RolloutInsightException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("解析 segment / pt_path 失败 segment_id={}: {}", segment_id, exc)

    turns: List[Dict[str, Any]] = []
    source = "turn_table"

    # 0) v0.5.0 T1 优先：tokens + tokenizer 切分（带 pt_path 解析）
    if pt_path_resolved:
        try:
            from utils.rollout_pt_parser import (
                load_pt,
                split_response_to_turns_with_tokenizer,
                _discover_tokenizer_path,
                _load_tokenizer,
            )
            dump = load_pt(pt_path_resolved)
            samples = dump.get("samples") or []
            # 按 (group_index, segment_index) 反查 sample
            tokens_for_sample: Optional[List[int]] = None
            sample_idx_found: Optional[int] = None
            try:
                with Session() as session:
                    seg_obj = (
                        session.query(Segment)
                        .filter(Segment.primary_id == segment_id)
                        .first()
                    )
                    target_si = getattr(seg_obj, "segment_index", None)
                    target_gi: Optional[int] = None
                    if Attempt is not None and getattr(seg_obj, "attempt_id", None):
                        att = (
                            session.query(Attempt)
                            .filter(Attempt.primary_id == seg_obj.attempt_id)
                            .first()
                        )
                        target_gi = getattr(att, "group_index", None)
                # 找 sample_idx
                for i, s in enumerate(samples):
                    md = s.get("metadata") or {}
                    if (
                        s.get("group_index") == target_gi
                        and md.get("segment_index") == target_si
                    ):
                        toks = s.get("tokens") or []
                        if isinstance(toks, list) and toks:
                            tokens_for_sample = [int(t) for t in toks]
                            sample_idx_found = i
                            break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "反查 sample 失败 segment_id={}: {}", segment_id, exc
                )

            tok = None
            try:
                tok_path = _discover_tokenizer_path(pt_path_resolved)
                if tok_path is not None:
                    tok = _load_tokenizer(tok_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "加载 tokenizer 失败 segment_id={}: {}", segment_id, exc
                )

            if tokens_for_sample is not None:
                parsed, parsed_source = split_response_to_turns_with_tokenizer(
                    response_text="",
                    tokens=tokens_for_sample,
                    tokenizer=tok,
                    dump_path=pt_path_resolved,
                )
                if parsed:
                    turns = parsed
                    source = f"tokens:{parsed_source}"
                    # 写回 turn 表（覆盖）
                    try:
                        from db.db_segment_turn_service import DBSegmentTurnService
                        DBSegmentTurnService.upsert_segment_turns(
                            segment_id=int(segment_id),
                            turns=parsed,
                            force=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "token 切分写回 turn 表失败 segment_id={}: {}",
                            segment_id,
                            exc,
                        )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "token 切分失败 segment_id={}: {}（回退到 turn 表）",
                segment_id,
                exc,
            )

    # 1) 兜底：从 turn 表读
    if not turns:
        try:
            from db.db_segment_turn_service import DBSegmentTurnService

            turns = DBSegmentTurnService.list_turns(int(segment_id))
            if turns:
                source = "turn_table"
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_turns 失败 segment_id={}: {}", segment_id, exc)

    # 2) 兜底：兜兜：ensure_transcript + 文本切分
    if not turns:
        transcript_text = ""
        try:
            ensure_fn = _ensure_transcript_fn()
            if callable(ensure_fn):
                r = ensure_fn(segment_id)
                if isinstance(r, dict):
                    transcript_text = (
                        r.get("transcript")
                        or r.get("text")
                        or r.get("response_text")
                        or ""
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ensure_transcript 失败 segment_id={}: {}", segment_id, exc)
        if not transcript_text:
            try:
                with Session() as session:
                    seg_obj = (
                        session.query(Segment)
                        .filter(Segment.primary_id == segment_id)
                        .first()
                    )
                    if seg_obj is not None:
                        transcript_text = _load_transcript_text(seg_obj, {})
            except Exception:
                transcript_text = ""

        if transcript_text:
            try:
                from db.db_segment_turn_service import DBSegmentTurnService
                from utils.rollout_pt_parser import split_response_to_turns

                parsed = split_response_to_turns(transcript_text)
                if parsed:
                    DBSegmentTurnService.upsert_segment_turns(
                        segment_id=int(segment_id), turns=parsed, force=True
                    )
                    turns = DBSegmentTurnService.list_turns(int(segment_id))
                    source = "parsed_on_demand"
            except Exception as exc:  # noqa: BLE001
                logger.warning("懒切分 turn 失败 segment_id={}: {}", segment_id, exc)

    # 渲染时附带预计算的高亮 HTML & 摘要，便于前端直接渲染
    items: List[Dict[str, Any]] = []
    for t in turns:
        body = t.get("body_text") or ""
        summary = _make_turn_summary(body)
        items.append(
            {
                "primary_id": t.get("primary_id"),
                "turn_index": t.get("turn_index"),
                "role": t.get("role") or "unknown",
                "role_label": t.get("role") or "unknown",
                "has_tool_call": bool(t.get("has_tool_call")),
                "has_think": bool(t.get("has_think")),
                "tool_call_names": t.get("tool_call_names") or [],
                "body_char_count": int(t.get("body_char_count") or len(body)),
                "token_estimate": int(t.get("token_estimate") or 0),
                "prompt_summary": t.get("prompt_summary") or summary,
                "html": _highlight_html(body),
                "body_text": body,
            }
        )

    return Response.success(
        data={
            "segment_id": int(segment_id),
            "turn_count": len(items),
            "turns": items,
            "source": source,
        }
    )


def _make_turn_summary(body: str, limit: int = 120) -> str:
    """根据 body 生成一行摘要（用于折叠态展示）。

    - 去除首尾空白
    - 把换行/多个空白压成一个空格
    - 超过 limit 截断并加 …
    """
    if not body:
        return ""
    flat = " ".join(body.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


def _load_transcript_text(seg_obj: Any, seg_dict: Dict[str, Any]) -> str:
    """从 segment ORM/字段里尝试取出 transcript 文本。

    兼容以下字段名（按优先级）：
    - transcript
    - transcript_text
    - response（多轮 transcript 文本）
    - prompt（最少给个 fallback）
    """
    for key in ("transcript", "transcript_text", "response"):
        v = getattr(seg_obj, key, None)
        if isinstance(v, str) and v.strip():
            return v
    # fallback: prompt
    prompt = getattr(seg_obj, "prompt", None)
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    return ""
