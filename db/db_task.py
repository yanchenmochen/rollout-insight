#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight 任务主表 ORM（ri_task_table）。

任务（task）是用户视角下的「一次观察」：给定一个 rollout 根目录或者一次南湖 job，
DBTaskService.create_task 写入一行 ri_task_table，后续扫描/检索都挂在 task 上。
"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, Index, Integer, String, Text
from sqlalchemy.orm.decl_api import DeclarativeBase  # noqa: F401  仅占位，保持风格一致

from db.db_element import Base


class DBTask(Base):
    """rollout-insight 任务主表。"""

    __tablename__ = "ri_task_table"
    __table_args__ = (
        Index("idx_task_status", "status"),
        Index("idx_task_name", "task_name"),
    )

    primary_id = Column(Integer, primary_key=True)
    task_name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=False, default="")
    # 'manual' 表示用户在页面手动创建；'nanhu' 表示由南湖调度触发（v0.3 启用）
    source_type = Column(String, nullable=False, default="manual")
    nanhu_job_id = Column(String, nullable=False, default="")
    # 扫描根路径（页面创建任务时填）；递归扫子目录标志位
    rollout_root = Column(String, nullable=False, default="")
    recursive = Column(Integer, nullable=False, default=1)
    tokenizer_dir = Column(String, nullable=False, default="")
    # 扫描结果摘要
    rollout_count = Column(Integer, nullable=False, default=0)
    attempt_count = Column(Integer, nullable=False, default=0)
    segment_count = Column(Integer, nullable=False, default=0)
    mean_reward = Column(Integer, nullable=False, default=0)
    last_scanned_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=False, default="")
    status = Column(String, nullable=False, default="PENDING")
    created_by = Column(String, nullable=False, default="")
    create_time = Column(DateTime, nullable=False, default=datetime.now)
    update_time = Column(
        DateTime, nullable=False, default=datetime.now, onupdate=datetime.now
    )

    def __repr__(self) -> str:
        return (
            f"<DBTask(primary_id={self.primary_id}, "
            f"task_name={self.task_name!r}, status={self.status!r})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典，便于序列化到模板/JSON。"""
        return {
            "primary_id": self.primary_id,
            "id": self.primary_id,  # 兼容 controller/模板
            "task_name": self.task_name,
            "description": self.description or "",
            "source_type": self.source_type,
            "nanhu_job_id": self.nanhu_job_id or "",
            "rollout_root": self.rollout_root or "",
            "recursive": bool(self.recursive),
            "tokenizer_dir": self.tokenizer_dir or "",
            "rollout_count": self.rollout_count or 0,
            "attempt_count": self.attempt_count or 0,
            "segment_count": self.segment_count or 0,
            "mean_reward": float(self.mean_reward or 0),
            "last_scanned_at": self._dt_iso(self.last_scanned_at),
            "error_message": self.error_message or "",
            "status": self.status,
            "created_by": self.created_by or "",
            "create_time": self._dt_iso(self.create_time),
            "update_time": self._dt_iso(self.update_time),
        }

    @staticmethod
    def _dt_iso(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None
