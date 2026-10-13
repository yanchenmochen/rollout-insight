#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight segment transcript 懒加载 service。

提供：
- ensure_transcript：第一次访问某个 segment 时解析 .pt 落盘并落库；之后走缓存
- _parse_transcript_md：把 view_trajectory 渲染出的 Markdown 切成 turns / tool_calls
- _extract_rollout_sample_index：从 summarize_dump 行表中按 (group, attempt, segment)
  匹配到 sample 行号

该 service 只关心 service/ 内的事务和工具封装，不动 db/ 下模型定义。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from db.db_attempt import DBAttempt
from db.db_element import Session
from db.db_rollout import DBRollout
from db.db_segment import DBSegment
from db.db_segment_transcript_service import DBSegmentTranscriptService
from data.rollout_insight_exception import RolloutInsightException
from utils.analyze_dump import render_transcript, summarize_dump


__all__ = [
    "ensure_transcript",
    "_parse_transcript_md",
    "_extract_rollout_sample_index",
]


# ---------- 文本切分常量 ----------
# 工具调用块的字面 `<tool_call>` 与 `</tool_call>` (注意是普通字符，源码里直接拼接)
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
# turns 行: `### turn N · {role}` 或 token 形式 `### turn N · role · tokens [a:b)`
_TURN_HEADER_RE = re.compile(r"^###\s+turn\s+\d+\s+·\s+(\S+)(?:\s+·\s+tokens\s+\[\d+:\d+\))?", re.MULTILINE)
# 单个 tool_call 块
_TOOL_CALL_RE = re.compile(
    re.escape(_TOOL_CALL_OPEN) + r"(.*?)" + re.escape(_TOOL_CALL_CLOSE),
    re.DOTALL,
)


def _parse_transcript_md(md_text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """把 view_trajectory 渲染出的 transcript Markdown 切成 turns / tool_calls。

    turns 形如 [{"role": "assistant", "content": "...", "mask": "TRAINABLE|MASKED"}]。
    tool_calls 形如 [{"name": "...", "arguments": {...}, "raw": "<tool_call>..."}]。
    解析失败或不命中时返回空列表。
    """
    turns: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []

    if not md_text:
        return turns, tool_calls

    # 找到所有 `### turn N · role` 头，按位置切分
    matches = list(_TURN_HEADER_RE.finditer(md_text))
    if not matches:
        return turns, tool_calls

    # 跳过 `### turn N · role · tokens [a:b)` 形式中的 `tokens ...` 信息
    for idx, match in enumerate(matches):
        role = (match.group(1) or "").strip().lower() or "unknown"
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(md_text)
        body = md_text[body_start:body_end].rstrip()

        # 解析下一行 mask: `mask: PROMPT/TRAINABLE [...] [x:y) n tok | ...`
        mask_value = _parse_mask_label(body.splitlines()[0]) if body else None

        # 从 body 中抽出所有 <tool_call>...</tool_call> 块
        for tc_idx, tc_match in enumerate(_TOOL_CALL_RE.finditer(body)):
            raw_block = tc_match.group(0)
            inner = tc_match.group(1)
            tool_calls.append(
                {
                    "name": _extract_tool_name(inner),
                    "arguments": _parse_tool_arguments(inner),
                    "raw": raw_block.strip(),
                    "turn_role": role,
                    "order": tc_idx,
                }
            )

        turns.append(
            {
                "role": role,
                "content": body.strip(),
                "mask": mask_value,
            }
        )

    return turns, tool_calls


_MASK_LINE_RE = re.compile(
    r"^mask:\s*(.*)$",
    re.MULTILINE,
)
_MASK_SEG_RE = re.compile(
    r"(?P<location>PROMPT|RESPONSE)/(?P<loss>TRAINABLE|MASKED)",
)


def _parse_mask_label(first_line: str) -> Optional[str]:
    """从 `mask: PROMPT/TRAINABLE [...] ...` 行抽出 mask 标签。

    返回 `TRAINABLE` / `MASKED` / `MIXED` / None。
    """
    if not first_line or not first_line.startswith("mask:"):
        return None
    labels = set(_MASK_SEG_RE.findall(first_line))
    if not labels:
        return None
    losses = {loss for _, loss in labels}
    if losses == {"TRAINABLE"}:
        return "TRAINABLE"
    if losses == {"MASKED"}:
        return "MASKED"
    return "MIXED"


# tool_call 块内部常见结构:
#   <tool_call>opensandbox_terminal<arg_key>command</arg_key><arg_value>ls</arg_value>
#                  <arg_key>security_risk</arg_key><arg_value>LOW</arg_value></tool_call>
_ARG_PAIR_RE = re.compile(
    r"<arg_key>(?P<key>[^<]+)</arg_key>\s*<arg_value>(?P<value>.*?)</arg_value>",
    re.DOTALL,
)


def _extract_tool_name(inner: str) -> str:
    """提取工具调用名（紧跟 `<tool_call>` 之后的第一个 token）。"""
    head = inner.lstrip()
    if not head:
        return ""
    # 第一个非空白 token
    name_match = re.match(r"([A-Za-z_][\w]*)", head)
    if not name_match:
        return ""
    return name_match.group(1)


def _parse_tool_arguments(inner: str) -> Dict[str, str]:
    """把 `<arg_key>k</arg_key><arg_value>v</arg_value>` 拆成 dict。"""
    pairs: Dict[str, str] = {}
    for match in _ARG_PAIR_RE.finditer(inner):
        key = (match.group("key") or "").strip()
        value = (match.group("value") or "").strip()
        if key:
            pairs[key] = value
    return pairs


# ---------- rollout sample 行号定位 ----------


def _extract_rollout_sample_index(
    rollout_pt_path: str,
    group_index: Optional[int],
    attempt_id: Optional[int],
    segment_index: Optional[int],
) -> int:
    """根据 (group_index, attempt_id, segment_index) 在 .pt sample 行表中定位行号。

    先调 summarize_dump 拿到所有 (idx, group_index, rollout_id, segment_index)，
    再线性匹配第一个精确三元组相等的行号。找不到抛 PARSE_SAMPLE_OUT_OF_RANGE。
    """
    if not rollout_pt_path:
        raise RolloutInsightException(
            RolloutInsightException.INVALID_ARGUMENT, "rollout_pt_path 不能为空"
        )
    rows = summarize_dump(rollout_pt_path)
    for row in rows:
        if (
            row.get("group_index") == group_index
            and row.get("rollout_id") == attempt_id
            and row.get("segment_index") == segment_index
        ):
            return int(row["idx"])
    raise RolloutInsightException(
        RolloutInsightException.PARSE_SAMPLE_OUT_OF_RANGE,
        f"未找到 group={group_index} attempt={attempt_id} segment={segment_index} 对应的 sample 行",
    )


# ---------- 主入口：ensure_transcript ----------


def ensure_transcript(
    segment_id: int,
    tokenizer_dir: str = "",
    force: bool = False,
) -> Dict[str, Any]:
    """懒加载 segment 的 transcript。

    1. 若 ri_segment_transcript_table 中该 segment 已 full_text_fetched=1 且未 force → 直接返回；
    2. 否则联表查出 pt_path / sample_idx → 渲染 Markdown → 解析 turns / tool_calls →
       upsert_transcript 写回库。

    返回 dict:
        {
            "segment_id": int,
            "turns_json": list,
            "tool_calls_json": list,
            "response_chars": int,
            "full_text_fetched": int (0 or 1),
            "prompt_text": str,
            "response_text": str,
        }
    """
    if segment_id is None:
        raise RolloutInsightException(
            RolloutInsightException.INVALID_ARGUMENT, "segment_id 不能为空"
        )

    existing = DBSegmentTranscriptService.get_by_segment_id(segment_id)
    if existing is not None and not force and existing.full_text_fetched == 1:
        return _transcript_to_dict(existing)

    # 联表查 segment → attempt → rollout，找 pt_path / sample_idx
    pt_path, attempt_meta, segment_index_value = _resolve_pt_path(segment_id)

    sample_index = _extract_rollout_sample_index(
        pt_path,
        group_index=attempt_meta.get("group_index"),
        attempt_id=attempt_meta.get("attempt_id"),
        segment_index=segment_index_value,
    )

    # 渲染 markdown
    try:
        md_text = render_transcript(
            pt_path,
            sample_index,
            tokenizer_dir=tokenizer_dir or None,
            include_prompt=True,
        )
    except Exception as exc:
        logger.exception("render_transcript 失败: {}", exc)
        raise RolloutInsightException(
            RolloutInsightException.PARSE_TRANSCRIPT_FAILED,
            f"segment_id={segment_id} 渲染 transcript 失败: {exc}",
        ) from exc

    prompt_text, response_text = _split_prompt_response(md_text)
    turns, tool_calls = _parse_transcript_md(md_text)
    response_chars = len(response_text)

    logger.info(
        "ensure_transcript: segment_id={} sample_idx={} turns={} tool_calls={} response_chars={}",
        segment_id,
        sample_index,
        len(turns),
        len(tool_calls),
        response_chars,
    )

    row = DBSegmentTranscriptService.upsert_transcript(
        segment_id=segment_id,
        prompt_text=prompt_text,
        response_text=response_text,
        turns=turns,
        tool_calls=tool_calls,
        tokenizer_path_used=tokenizer_dir or "",
        response_chars=response_chars,
        mark_as_fetched=True,
    )
    return _transcript_to_dict(row)


def _resolve_pt_path(
    segment_id: int,
) -> Tuple[str, Dict[str, Any], Optional[int]]:
    """联表查 segment → attempt → rollout，回传 (pt_path, attempt_meta, segment_index)。
    """
    with Session() as session:
        seg = (
            session.query(DBSegment)
            .filter(DBSegment.primary_id == segment_id)
            .first()
        )
        if seg is None:
            raise RolloutInsightException(
                RolloutInsightException.TASK_NOT_FOUND,
                f"segment_id={segment_id} 不存在",
            )
        attempt = (
            session.query(DBAttempt)
            .filter(DBAttempt.primary_id == seg.attempt_id)
            .first()
        )
        if attempt is None:
            raise RolloutInsightException(
                RolloutInsightException.PARSE_ERROR,
                f"segment_id={segment_id} 关联的 attempt {seg.attempt_id} 缺失",
            )
        rollout = (
            session.query(DBRollout)
            .filter(DBRollout.task_id == attempt.task_id)
            .filter(DBRollout.rollout_id == attempt.rollout_id)
            .order_by(DBRollout.scanned_at.desc())
            .first()
        )
        if rollout is None or not rollout.pt_path:
            raise RolloutInsightException(
                RolloutInsightException.PARSE_DUMP_NOT_FOUND,
                f"segment_id={segment_id} 找不到对应 .pt 文件",
            )
        attempt_meta = {
            "task_id": attempt.task_id,
            "rollout_id": attempt.rollout_id,
            "group_index": attempt.group_index,
            "attempt_id": attempt.attempt_id,
        }
        return rollout.pt_path, attempt_meta, seg.segment_index


def _split_prompt_response(md_text: str) -> Tuple[str, str]:
    """把渲染出的 md 拆成 prompt_text 与 response_text 两段。

    view_trajectory 的渲染 header 形如：
        ----- DATASET PROMPT (full token sequence unavailable) -----
        <prompt body>
        ----- LEGACY TRANSCRIPT (response only) -----
        <transcript>
    或 token 化时无 DATASET PROMPT 块（整段都是 transcript）。
    """
    if not md_text:
        return "", ""
    prompt_marker = "----- DATASET PROMPT"
    transcript_marker = "----- LEGACY TRANSCRIPT"
    token_marker = "----- COMPLETE STORED TRAINING SEQUENCE -----"
    # 找出 prompt 段
    prompt_idx = md_text.find(prompt_marker)
    transcript_idx = md_text.find(transcript_marker)
    token_idx = md_text.find(token_marker)

    if prompt_idx >= 0 and transcript_idx > prompt_idx:
        prompt_body = md_text[prompt_idx + len(prompt_marker):transcript_idx]
        # 去掉分隔横线
        prompt_body = re.sub(r"^-+\s*", "", prompt_body, count=2).strip()
        response_body = md_text[transcript_idx + len(transcript_marker):].strip()
        return prompt_body, response_body

    if token_idx >= 0:
        # 整段视为 transcript（response）
        return "", md_text[token_idx + len(token_marker):].strip()

    # 兜底：把整个 md 视作 response
    return "", md_text.strip()


def _transcript_to_dict(row: Any) -> Dict[str, Any]:
    """把 ORM row 转成 dict；turns_json / tool_calls_json 解析回 list。"""
    turns_payload = _safe_json(row.turns_json, default=[])
    tool_calls_payload = _safe_json(row.tool_calls_json, default=[])
    return {
        "segment_id": int(row.segment_id),
        "turns_json": turns_payload,
        "tool_calls_json": tool_calls_payload,
        "response_chars": int(row.response_chars or 0),
        "full_text_fetched": int(row.full_text_fetched or 0),
        "prompt_text": row.prompt_text or "",
        "response_text": row.response_text or "",
        "tokenizer_path_used": row.tokenizer_path_used or "",
    }


def _safe_json(raw: Any, default: Any) -> Any:
    """安全解析 json 字符串；解析失败时返回 default。"""
    if raw is None:
        return default
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default