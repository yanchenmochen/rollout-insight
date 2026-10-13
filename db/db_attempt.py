#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight attempt 表 ORM（ri_attempt_table）。

每个 (task, rollout, group_index, attempt_id) 一行 attempt；attempt 可能跨多个 segment
（落盘时 group 内多轮重发的轨迹合并为一段；具体拆分由 segment 表承担）。
"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint

from db.db_element import Base


class DBAttempt(Base):
    """rollout 内 (group, attempt) 维度的汇总。"""

    __tablename__ = "ri_attempt_table"
    __table_args__ = (
        UniqueConstraint(
            "task_id",
            "rollout_id",
            "group_index",
            "attempt_id",
            name="uq_attempt_task_rollout_group_attempt",
        ),
    )

    primary_id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey("ri_task_table.primary_id"), nullable=False)
    rollout_id = Column(Integer, nullable=False)
    group_index = Column(Integer, nullable=False)
    attempt_id = Column(Integer, nullable=False)
    sandbox_id = Column(String, nullable=False, default="")
    instance_id = Column(String, nullable=False, default="")
    status = Column(String, nullable=False, default="")
    reward = Column(Float, nullable=True)
    agent_turns = Column(Integer, nullable=False, default=0)
    tool_calls = Column(Integer, nullable=False, default=0)
    diff_line_count = Column(Integer, nullable=False, default=0)
    response_length = Column(Integer, nullable=False, default=0)
    trainable_tokens = Column(Integer, nullable=False, default=0)
    finalized = Column(Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        return (
            f"<DBAttempt(primary_id={self.primary_id}, task_id={self.task_id}, "
            f"rollout_id={self.rollout_id}, group_index={self.group_index}, "
            f"attempt_id={self.attempt_id})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_id": self.primary_id,
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "group_index": self.group_index,
            "attempt_id": self.attempt_id,
            "sandbox_id": self.sandbox_id,
            "instance_id": self.instance_id,
            "status": self.status,
            "reward": self.reward,
            "agent_turns": self.agent_turns,
            "tool_calls": self.tool_calls,
            "diff_line_count": self.diff_line_count,
            "response_length": self.response_length,
            "trainable_tokens": self.trainable_tokens,
            "finalized": self.finalized,
        }
