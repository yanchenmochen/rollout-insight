#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight segment 表 ORM（ri_segment_table）。

每个 attempt 内一个 sample 一行 segment；segment 是 prompt → response 的最小单元，
懒加载 transcript 时按 segment_id 查回 ri_segment_transcript_table。

注意：plan 文档将 sandbox_id 在字段表里重复列了一行，本实现保留该列以便后续按
sandbox 反查（与 attempt 表对齐）。
"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint

from db.db_element import Base


class DBSegment(Base):
    """attempt 内 (segment_index) 维度的最小单元。"""

    __tablename__ = "ri_segment_table"
    __table_args__ = (
        UniqueConstraint(
            "attempt_id",
            "segment_index",
            name="uq_segment_attempt_segment_index",
        ),
    )

    primary_id = Column(Integer, primary_key=True)
    attempt_id = Column(Integer, ForeignKey("ri_attempt_table.primary_id"), nullable=False)
    segment_index = Column(Integer, nullable=False)
    segment_count = Column(Integer, nullable=False, default=1)
    status = Column(String, nullable=False, default="")
    reward = Column(Float, nullable=True)
    response_length = Column(Integer, nullable=False, default=0)
    trainable_tokens = Column(Integer, nullable=False, default=0)
    mask_sum = Column(Integer, nullable=False, default=0)
    # 与 attempt.sandbox_id 一致；保留便于按 sandbox 反查
    sandbox_id = Column(String, nullable=False, default="")
    create_time = Column(DateTime, nullable=False, default=datetime.now)

    def __repr__(self) -> str:
        return (
            f"<DBSegment(primary_id={self.primary_id}, attempt_id={self.attempt_id}, "
            f"segment_index={self.segment_index})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_id": self.primary_id,
            "attempt_id": self.attempt_id,
            "segment_index": self.segment_index,
            "segment_count": self.segment_count,
            "status": self.status,
            "reward": self.reward,
            "response_length": self.response_length,
            "trainable_tokens": self.trainable_tokens,
            "mask_sum": self.mask_sum,
            "sandbox_id": self.sandbox_id,
            "create_time": self._dt_iso(self.create_time),
        }

    @staticmethod
    def _dt_iso(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None
