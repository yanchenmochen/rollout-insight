#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight DBSegmentTranscriptService —— ri_segment_transcript_table 的 CRUD。

提供：
- get_by_segment_id：懒加载 segment 的 transcript 文本
- upsert_transcript：懒加载后写入；扫描阶段可主动调 mark_fetched
- mark_fetched：仅在已存在行上把 full_text_fetched 置 1
"""
import json
from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import and_

from db.db_element import Session
from db.db_segment import DBSegment
from db.db_segment_transcript import DBSegmentTranscript
from data.rollout_insight_exception import RolloutInsightException


class DBSegmentTranscriptService:
    """transcript 表的读写服务。"""

    @staticmethod
    def _dump_json(value: Any) -> str:
        if value is None:
            return "[]"
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return "[]"

    @staticmethod
    def get_by_segment_id(segment_id: int) -> Optional[DBSegmentTranscript]:
        """按 segment_id 查 transcript；不存在返回 None。"""
        if segment_id is None:
            return None
        with Session() as session:
            return (
                session.query(DBSegmentTranscript)
                .filter(DBSegmentTranscript.segment_id == segment_id)
                .first()
            )

    @staticmethod
    def upsert_transcript(
        segment_id: int,
        prompt_text: str = "",
        response_text: str = "",
        turns: Optional[List[Any]] = None,
        tool_calls: Optional[List[Any]] = None,
        diff_text: Optional[str] = None,
        tokenizer_path_used: str = "",
        response_chars: Optional[int] = None,
        mark_as_fetched: bool = True,
    ) -> DBSegmentTranscript:
        """写入或更新一条 transcript。

        若 (segment_id) 行不存在 → INSERT；
        若存在 → 更新 prompt_text / response_text / *_json / 计数，并按 mark_as_fetched
        标记 full_text_fetched。
        """
        if segment_id is None:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "segment_id 不能为空"
            )
        turns_payload = DBSegmentTranscriptService._dump_json(turns)
        tool_calls_payload = DBSegmentTranscriptService._dump_json(tool_calls)
        if response_chars is None:
            response_chars = len(response_text or "")
        with Session() as session:
            existing: Optional[DBSegmentTranscript] = (
                session.query(DBSegmentTranscript)
                .filter(DBSegmentTranscript.segment_id == segment_id)
                .first()
            )
            if existing is None:
                # 验证 segment 存在，避免悬挂外键
                seg_exists = (
                    session.query(DBSegment)
                    .filter(DBSegment.primary_id == segment_id)
                    .first()
                )
                if seg_exists is None:
                    raise RolloutInsightException(
                        RolloutInsightException.TASK_NOT_FOUND,
                        f"segment_id={segment_id} 不存在，无法创建 transcript",
                    )
                row = DBSegmentTranscript(
                    segment_id=segment_id,
                    prompt_text=prompt_text or "",
                    response_text=response_text or "",
                    turns_json=turns_payload,
                    tool_calls_json=tool_calls_payload,
                    diff_text=diff_text,
                    tokenizer_path_used=tokenizer_path_used or "",
                    full_text_fetched=1 if mark_as_fetched else 0,
                    response_chars=response_chars,
                )
                session.add(row)
                session.commit()
                session.refresh(row)
                return row

            existing.prompt_text = prompt_text or ""
            existing.response_text = response_text or ""
            existing.turns_json = turns_payload
            existing.tool_calls_json = tool_calls_payload
            existing.diff_text = diff_text
            existing.tokenizer_path_used = tokenizer_path_used or ""
            existing.response_chars = response_chars
            # mark_as_fetched 显式决定 fetched 状态：True→1，False→0
            existing.full_text_fetched = 1 if mark_as_fetched else 0
            existing.update_time = datetime.now()
            session.commit()
            session.refresh(existing)
            return existing

    @staticmethod
    def mark_fetched(segment_id: int, tokenizer_path_used: str = "") -> bool:
        """把一条 transcript 标记为 full_text_fetched=1；行不存在则返回 False。"""
        if segment_id is None:
            return False
        with Session() as session:
            row = (
                session.query(DBSegmentTranscript)
                .filter(DBSegmentTranscript.segment_id == segment_id)
                .first()
            )
            if row is None:
                return False
            row.full_text_fetched = 1
            if tokenizer_path_used:
                row.tokenizer_path_used = tokenizer_path_used
            row.update_time = datetime.now()
            session.commit()
            return True

    @staticmethod
    def search_by_keyword(
        keyword: str,
        task_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DBSegmentTranscript]:
        """按 response_text 关键词查询（已 fetched 的 transcript 命中）。

        关键词通过 LIKE '%kw%'；命中即写回 full_text_fetched=1（不在此处写，仅返回）。
        """
        kw = (keyword or "").strip()
        if not kw:
            return []
        with Session() as session:
            query = session.query(DBSegmentTranscript).filter(
                and_(
                    DBSegmentTranscript.full_text_fetched == 1,
                    DBSegmentTranscript.response_text.like(f"%{kw}%"),
                )
            )
            if task_id is not None:
                # JOIN ri_segment_table → ri_attempt_table 过滤 task_id
                from db.db_attempt import DBAttempt

                query = query.join(
                    DBSegment, DBSegment.primary_id == DBSegmentTranscript.segment_id
                ).join(
                    DBAttempt, DBAttempt.primary_id == DBSegment.attempt_id
                ).filter(
                    DBAttempt.task_id == task_id
                )
            return (
                query.order_by(DBSegmentTranscript.update_time.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

    @staticmethod
    def delete_by_segment_id(segment_id: int) -> bool:
        """按 segment_id 删 transcript；不存在返回 False。"""
        if segment_id is None:
            return False
        with Session() as session:
            row = (
                session.query(DBSegmentTranscript)
                .filter(DBSegmentTranscript.segment_id == segment_id)
                .first()
            )
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True
