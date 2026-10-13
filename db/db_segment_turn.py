#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight v0.4.0 segment turn 表 ORM（ri_segment_turn_table）。

把 ``ri_segment_transcript_table.turns_json`` 的 JSON 数组规范化：每个 turn 一行，
便于 T5 三级树状视图 + T6 token 分布统计。

注意（T4 决策）：
- 不存整文 ``response_text``（避免 1.5GB 膨胀）
- 每个 turn 存 ``body_text``（≤ 100KB），整文 response 通过
  ``ri_segment_transcript_table`` 在访问时懒加载
- 索引：``(segment_id, turn_index)`` UNIQUE；``body_text`` 不建索引（避免 TEXT 膨胀）
"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from db.db_element import Base


class DBSegmentTurn(Base):
    """segment 内按 turn 切分的明细行。"""

    __tablename__ = "ri_segment_turn_table"
    __table_args__ = (
        UniqueConstraint(
            "segment_id",
            "turn_index",
            name="uq_segment_turn_segment_turn",
        ),
        # 普通索引：方便按 segment_id 拉全部 turn
        Index("idx_segment_turn_segment_id", "segment_id"),
    )

    primary_id = Column(Integer, primary_key=True)
    segment_id = Column(
        Integer,
        ForeignKey("ri_segment_table.primary_id"),
        nullable=False,
    )
    turn_index = Column(Integer, nullable=False)
    role = Column(String(16), nullable=False, default="unknown")
    body_text = Column(Text, nullable=False, default="")
    body_char_count = Column(Integer, nullable=False, default=0)
    has_tool_call = Column(Boolean, nullable=False, default=False)
    has_think = Column(Boolean, nullable=False, default=False)
    tool_call_names_json = Column(Text, nullable=False, default="[]")
    im_start_marker = Column(String(16), nullable=False, default="")
    token_estimate = Column(Integer, nullable=False, default=0)
    prompt_summary = Column(Text(512), nullable=False, default="")
    create_time = Column(DateTime, nullable=False, default=datetime.now)

    def __repr__(self) -> str:
        return (
            f"<DBSegmentTurn(primary_id={self.primary_id}, "
            f"segment_id={self.segment_id}, turn_index={self.turn_index}, "
            f"role={self.role!r})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        import json as _json

        try:
            tool_calls_payload = _json.loads(self.tool_call_names_json or "[]")
        except (TypeError, ValueError):
            tool_calls_payload = []
        if not isinstance(tool_calls_payload, list):
            tool_calls_payload = []
        return {
            "primary_id": self.primary_id,
            "segment_id": self.segment_id,
            "turn_index": self.turn_index,
            "role": self.role or "unknown",
            "body_text": self.body_text or "",
            "body_char_count": int(self.body_char_count or 0),
            "has_tool_call": bool(self.has_tool_call),
            "has_think": bool(self.has_think),
            "tool_call_names": tool_calls_payload,
            "im_start_marker": self.im_start_marker or "",
            "token_estimate": int(self.token_estimate or 0),
            "prompt_summary": self.prompt_summary or "",
            "create_time": self._dt_iso(self.create_time),
        }

    @staticmethod
    def _dt_iso(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None
