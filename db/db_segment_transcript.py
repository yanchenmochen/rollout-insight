#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight segment transcript 表 ORM（ri_segment_transcript_table）。

与 ri_segment_table 1:1：segment_id 即 FK 主键；首次访问 transcript 时解析
.pt 并落库（full_text_fetched=1），后续检索直接 SELECT 即可。
"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

from db.db_element import Base


class DBSegmentTranscript(Base):
    """lazy-loaded transcript 文本与原文指针。"""

    __tablename__ = "ri_segment_transcript_table"

    primary_id = Column(Integer, primary_key=True)
    # 1:1 关系：segment_id 在该表唯一
    segment_id = Column(
        Integer,
        ForeignKey("ri_segment_table.primary_id"),
        nullable=False,
        unique=True,
    )
    prompt_text = Column(Text, nullable=False, default="")
    response_text = Column(Text, nullable=False, default="")
    turns_json = Column(Text, nullable=False, default="[]")
    tool_calls_json = Column(Text, nullable=False, default="[]")
    # 增量保留字段：v0.2+ 才真正填充
    diff_text = Column(Text, nullable=True)
    tokenizer_path_used = Column(String, nullable=False, default="")
    full_text_fetched = Column(Integer, nullable=False, default=0)
    response_chars = Column(Integer, nullable=False, default=0)
    create_time = Column(DateTime, nullable=False, default=datetime.now)
    update_time = Column(
        DateTime, nullable=False, default=datetime.now, onupdate=datetime.now
    )

    def __repr__(self) -> str:
        return (
            f"<DBSegmentTranscript(primary_id={self.primary_id}, "
            f"segment_id={self.segment_id}, "
            f"full_text_fetched={self.full_text_fetched})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_id": self.primary_id,
            "segment_id": self.segment_id,
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "turns_json": self.turns_json,
            "tool_calls_json": self.tool_calls_json,
            "diff_text": self.diff_text,
            "tokenizer_path_used": self.tokenizer_path_used,
            "full_text_fetched": self.full_text_fetched,
            "response_chars": self.response_chars,
            "create_time": self._dt_iso(self.create_time),
            "update_time": self._dt_iso(self.update_time),
        }

    @staticmethod
    def _dt_iso(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None
