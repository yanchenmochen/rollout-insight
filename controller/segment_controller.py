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
    return _safe_import_attr("db.models", "Segment") or _safe_import_attr("db", "Segment")


def _attempt_model():
    return _safe_import_attr("db.models", "Attempt") or _safe_import_attr("db", "Attempt")


def _rollout_model():
    return _safe_import_attr("db.models", "Rollout") or _safe_import_attr("db", "Rollout")


def _task_model():
    return _safe_import_attr("db.models", "Task") or _safe_import_attr("db", "Task")


def _segment_service():
    return _safe_import_attr("service.segment_service", "SegmentService")


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
        "status": _status_str(getattr(s, "status", None)),
        "reward": getattr(s, "reward", None),
        "agent_turns": getattr(s, "agent_turns", None),
        "tool_calls": getattr(s, "tool_calls", None),
        "response_length": getattr(s, "response_length", None),
        "diff_line_count": getattr(s, "diff_line_count", None),
        "pt_path": getattr(s, "pt_path", None),
    }


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
    keyword: Optional[str] = Query(default=None, description="instance_id / sandbox_id 模糊匹配"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
):
    """Segment 检索：详细检索/聚合逻辑由 2D 阶段补齐。

    当前实现：
    - 必须有 task_id（2D 阶段才会放开）
    - 返回基本字段（id / attempt / instance_id / reward / turns / tool_calls / status / segment 序号）
    """
    if task_id is None:
        return Response.error(
            code=50001,
            msg="当前阶段 2C 仅支持 task_id 过滤的 segment 检索；详细检索由 2D 补齐",
        )

    Segment = _segment_model()
    Attempt = _attempt_model()
    Rollout = _rollout_model()
    Session = _session_factory()
    if Segment is None or Session is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "Segment 模型尚未就绪（2A 阶段未完成）",
        )

    try:
        with Session() as session:
            q = session.query(Segment)
            if Attempt is not None and Rollout is not None:
                q = q.join(Attempt, Segment.attempt_id == Attempt.id)
                q = q.join(Rollout, Attempt.rollout_id == Rollout.id)
                q = q.filter(Rollout.task_id == task_id)
            if min_reward is not None:
                q = q.filter(Segment.reward >= min_reward)
            if keyword:
                like = f"%{keyword}%"
                # instance_id / sandbox_id 任一命中
                q = q.filter(
                    (Segment.instance_id.ilike(like)) | (Segment.sandbox_id.ilike(like))
                )

            total = q.count()
            rows = (
                q.order_by(Segment.id.asc())
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )
            seg_list = [_segment_to_dict(s) for s in rows]
        return Response.success(
            data={
                "segments": seg_list,
                "total": total,
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
    """渲染 segment_transcript.html。"""
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

    if Segment is None or Session is None:
        warning = "Segment 模型尚未就绪（2A 阶段未完成）"
    else:
        try:
            with Session() as session:
                obj = session.query(Segment).filter(Segment.id == segment_id).first()
                if obj is None:
                    warning = f"segment #{segment_id} 不存在"
                else:
                    seg = _segment_to_dict(obj)
                    # 找 task_id（segment → attempt → rollout → task）
                    if (
                        Attempt is not None
                        and Rollout is not None
                        and Task is not None
                        and getattr(obj, "attempt_id", None) is not None
                    ):
                        attempt_obj = session.query(Attempt).filter(Attempt.id == obj.attempt_id).first()
                        if attempt_obj is not None and getattr(attempt_obj, "rollout_id", None) is not None:
                            rollout_obj = (
                                session.query(Rollout)
                                .filter(Rollout.id == attempt_obj.rollout_id)
                                .first()
                            )
                            if rollout_obj is not None and getattr(rollout_obj, "task_id", None) is not None:
                                task_id = rollout_obj.task_id
                                seg["rollout_id"] = rollout_obj.id

                    # 尝试加载 transcript 文本
                    transcript_text = _load_transcript_text(obj, seg)
                    if transcript_text:
                        turns = _build_turns_html(transcript_text)
                    else:
                        warning = (
                            "未找到 transcript 文本（可能为 2D 阶段尚未补 ensure_transcript，"
                            "或未提供 tokenizer 目录）"
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
            "warning": warning,
        },
    )


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
    Session = _session_factory()
    if Segment is None or Session is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "Segment 模型尚未就绪（2A 阶段未完成）",
        )

    with Session() as session:
        obj = session.query(Segment).filter(Segment.id == segment_id).first()
        if obj is None:
            raise RolloutInsightException(
                RolloutInsightException.PARSE_DUMP_NOT_FOUND,
                f"segment #{segment_id} 不存在",
            )
        seg = _segment_to_dict(obj)

    # 尝试走 SegmentService.ensure_transcript（2D 阶段补齐）
    transcript_text: Optional[str] = None
    used_service = False
    try:
        svc_cls = _segment_service()
        if svc_cls is not None:
            svc_inst = svc_cls() if callable(svc_cls) else svc_cls
            if hasattr(svc_inst, "ensure_transcript"):
                result = svc_inst.ensure_transcript(segment_id)
                # 约定：返回 dict {transcript: str, ...} 或 str
                if isinstance(result, dict):
                    transcript_text = result.get("transcript") or result.get("text")
                elif isinstance(result, str):
                    transcript_text = result
                used_service = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("SegmentService.ensure_transcript 调用失败: {}", exc)

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
