#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight rollout 主表 ORM（ri_rollout_table）。

一个 rollout 对应一份 .pt 落盘文件：扫描阶段以 (pt_path, pt_size_bytes, pt_mtime)
三件套做指纹 upsert；rollout_id 是 .pt 内部自带的顺序号（来自 slime 落盘）。
"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, UniqueConstraint
from sqlalchemy import ForeignKey

from db.db_element import Base


class DBRollout(Base):
    """rollout 与 .pt 落盘文件一一对应。"""

    __tablename__ = "ri_rollout_table"
    __table_args__ = (
        UniqueConstraint("task_id", "rollout_id", name="uq_rollout_task_rollout_id"),
        Index("idx_rollout_pt_path", "pt_path"),
    )

    primary_id = Column(Integer, primary_key=True)
    # 业务侧 task_id，与 ri_task_table.primary_id 对应；任务删除时级联清理
    task_id = Column(Integer, ForeignKey("ri_task_table.primary_id"), nullable=False)
    # slime 落盘时的 rollout 顺序号
    rollout_id = Column(Integer, nullable=False)
    # pt 文件绝对路径（唯一指纹之一）
    pt_path = Column(String, nullable=False)
    pt_size_bytes = Column(Integer, nullable=False, default=0)
    pt_mtime = Column(Float, nullable=False, default=0.0)
    # 摘要统计（来自 analyze_dump.summarize_dump）
    mean_reward = Column(Float, nullable=False, default=0.0)
    n_samples = Column(Integer, nullable=False, default=0)
    n_attempts = Column(Integer, nullable=False, default=0)
    instance_count = Column(Integer, nullable=False, default=0)
    scanned_at = Column(DateTime, nullable=False, default=datetime.now)
    create_time = Column(DateTime, nullable=False, default=datetime.now)

    def __repr__(self) -> str:
        return (
            f"<DBRollout(primary_id={self.primary_id}, "
            f"task_id={self.task_id}, rollout_id={self.rollout_id}, "
            f"pt_path={self.pt_path!r})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_id": self.primary_id,
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "pt_path": self.pt_path,
            "pt_size_bytes": self.pt_size_bytes,
            "pt_mtime": self.pt_mtime,
            "mean_reward": self.mean_reward,
            "n_samples": self.n_samples,
            "n_attempts": self.n_attempts,
            "instance_count": self.instance_count,
            "scanned_at": self._dt_iso(self.scanned_at),
            "create_time": self._dt_iso(self.create_time),
        }

    @staticmethod
    def _dt_iso(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None
