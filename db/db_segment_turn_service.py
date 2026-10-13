#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight v0.4.0 DBSegmentTurnService —— ri_segment_turn_table 的 CRUD。

提供：
- ``upsert_segment_turns(segment_id, turns, force=False)``：先删旧再批量插
- ``list_turns(segment_id)``：按 turn_index 升序返回
- ``delete_by_segment(segment_id)``：清空某 segment 的 turn 行
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from db.db_element import Session
from db.db_segment import DBSegment
from db.db_segment_turn import DBSegmentTurn
from data.rollout_insight_exception import RolloutInsightException


__all__ = ["DBSegmentTurnService"]


# body_text 单 turn 上限 100KB（防止 OOM / 误存整文 response）
_BODY_TEXT_MAX_BYTES = 100 * 1024


def _dump_json(value: Any) -> str:
    if value is None:
        return "[]"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return "[]"


def _safe_load_json(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


class DBSegmentTurnService:
    """turn 表的读写服务。"""

    @staticmethod
    def upsert_segment_turns(
        segment_id: int,
        turns: List[Dict[str, Any]],
        force: bool = False,
    ) -> int:
        """按 (segment_id, turn_index) upsert turn 行。

        Args:
            segment_id: 关联的 ``ri_segment_table.primary_id``
            turns: turn 列表，每项含 ``turn_index/role/role_label/body/has_tool_call/has_think/tool_call_names/im_start_marker/token_estimate``
            force: True 时先 delete_by_segment 再插

        Returns:
            写入的行数
        """
        if segment_id is None:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "segment_id 不能为空"
            )
        if not turns:
            return 0

        # 校验 segment 存在
        with Session() as session:
            seg_exists = (
                session.query(DBSegment)
                .filter(DBSegment.primary_id == segment_id)
                .first()
            )
            if seg_exists is None:
                raise RolloutInsightException(
                    RolloutInsightException.TASK_NOT_FOUND,
                    f"segment_id={segment_id} 不存在，无法写 turn",
                )

        if force:
            DBSegmentTurnService.delete_by_segment(segment_id)

        rows: List[DBSegmentTurn] = []
        for i, t in enumerate(turns):
            turn_index = t.get("turn_index") or (i + 1)
            role = (t.get("role_label") or t.get("role") or "unknown")
            if not isinstance(role, str):
                role = str(role)
            role = role.lower()[:16]
            body = t.get("body") or ""
            if not isinstance(body, str):
                body = str(body)
            # 截断超大 body
            if len(body.encode("utf-8")) > _BODY_TEXT_MAX_BYTES:
                body = body[: _BODY_TEXT_MAX_BYTES // 2]
                logger.warning(
                    "upsert_segment_turns: segment_id={} turn_index={} body 截断至 {} bytes",
                    segment_id,
                    turn_index,
                    len(body.encode("utf-8")),
                )
            tool_call_names = t.get("tool_call_names") or []
            if not isinstance(tool_call_names, list):
                tool_call_names = []
            im_start_marker = t.get("im_start_marker") or "<|im_start|>"
            if not isinstance(im_start_marker, str):
                im_start_marker = "<|im_start|>"
            im_start_marker = im_start_marker[:16]
            token_estimate = int(t.get("token_estimate") or 0)
            prompt_summary = t.get("prompt_summary") or ""
            if not isinstance(prompt_summary, str):
                prompt_summary = str(prompt_summary)
            if len(prompt_summary) > 512:
                prompt_summary = prompt_summary[:512]

            rows.append(
                DBSegmentTurn(
                    segment_id=int(segment_id),
                    turn_index=int(turn_index),
                    role=role,
                    body_text=body,
                    body_char_count=len(body),
                    has_tool_call=bool(t.get("has_tool_call")),
                    has_think=bool(t.get("has_think")),
                    tool_call_names_json=_dump_json(tool_call_names),
                    im_start_marker=im_start_marker,
                    token_estimate=token_estimate,
                    prompt_summary=prompt_summary,
                )
            )

        try:
            with Session() as session:
                session.add_all(rows)
                session.commit()
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "upsert_segment_turns: 批量插入失败 segment_id={} err={}", segment_id, exc
            )
            raise RolloutInsightException(
                RolloutInsightException.DB_ERROR,
                f"写入 turn 失败: {exc}",
            ) from exc

        return len(rows)

    @staticmethod
    def list_turns(segment_id: int) -> List[Dict[str, Any]]:
        """按 turn_index 升序返回 turn 字典列表。"""
        if segment_id is None:
            return []
        with Session() as session:
            rows = (
                session.query(DBSegmentTurn)
                .filter(DBSegmentTurn.segment_id == segment_id)
                .order_by(DBSegmentTurn.turn_index.asc())
                .all()
            )
            return [r.to_dict() for r in rows]

    @staticmethod
    def delete_by_segment(segment_id: int) -> int:
        """删除某 segment 的所有 turn，返回受影响行数。"""
        if segment_id is None:
            return 0
        with Session() as session:
            count = (
                session.query(DBSegmentTurn)
                .filter(DBSegmentTurn.segment_id == segment_id)
                .delete(synchronize_session=False)
            )
            session.commit()
            return int(count or 0)

    @staticmethod
    def list_orm_turns(segment_id: int) -> List[DBSegmentTurn]:
        """返回 ORM 对象列表（不转 dict），用于批量聚合统计。"""
        if segment_id is None:
            return []
        with Session() as session:
            return (
                session.query(DBSegmentTurn)
                .filter(DBSegmentTurn.segment_id == segment_id)
                .order_by(DBSegmentTurn.turn_index.asc())
                .all()
            )

    @staticmethod
    def get_tool_call_names(row: DBSegmentTurn) -> List[str]:
        """从 ORM 行抽出 tool_call_names 列表。"""
        if row is None:
            return []
        payload = _safe_load_json(row.tool_call_names_json, default=[])
        if not isinstance(payload, list):
            return []
        return [str(x) for x in payload if x]
