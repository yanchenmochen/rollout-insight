#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight 南湖作业重启记录表 ORM（ri_nanhu_restart_table）。

每个 nanhu_job_id 维度下按 restart_index 单调递增：每发起一次重启就 +1。
related_task_id 指向被重启的训练任务行（ri_task_table，可空：尚未 link）。
唯一约束 (nanhu_job_id, restart_index) 保证同一个 job 的 restart_index 不重号。
"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint

from db.db_element import Base


class DBNanhuRestart(Base):
    """南湖作业的重启历史。"""

    __tablename__ = "ri_nanhu_restart_table"
    __table_args__ = (
        UniqueConstraint(
            "nanhu_job_id",
            "restart_index",
            name="uq_nanhu_restart_job_index",
        ),
    )

    primary_id = Column(Integer, primary_key=True)
    # FK：ri_nanhu_job_table.job_id（字符串 ID）
    nanhu_job_id = Column(
        String,
        ForeignKey("ri_nanhu_job_table.job_id"),
        nullable=False,
    )
    # 自增重启序号（同一 job 内唯一）
    restart_index = Column(Integer, nullable=False, default=0)
    # 重启发起时间
    restart_at = Column(DateTime, nullable=False, default=datetime.now)
    # 重启状态（REQUESTED / SUBMITTED / FAILED ...）
    restart_status = Column(String, nullable=False, default="REQUESTED")
    # 可选：被关联到的训练任务（ri_task_table.primary_id）；允许为空
    related_task_id = Column(
        Integer,
        ForeignKey("ri_task_table.primary_id"),
        nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f"<DBNanhuRestart(primary_id={self.primary_id}, "
            f"nanhu_job_id={self.nanhu_job_id!r}, "
            f"restart_index={self.restart_index}, "
            f"restart_status={self.restart_status!r})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_id": self.primary_id,
            "nanhu_job_id": self.nanhu_job_id,
            "restart_index": self.restart_index,
            "restart_at": self._dt_iso(self.restart_at),
            "restart_status": self.restart_status,
            "related_task_id": self.related_task_id,
        }

    @staticmethod
    def _dt_iso(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None