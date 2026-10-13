#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight 南湖作业主表 ORM（ri_nanhu_job_table）。

每行是南湖调度平台上一个训练 / 推理 job 的本地缓存（首次 list_train_jobs 时落库，
后续 link/unlink/restart 都围绕此表展开）。job_id 是南湖侧字符串 ID（unique）。
"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, Index, Integer, String, Text
from sqlalchemy.orm.decl_api import DeclarativeBase  # noqa: F401  仅占位

from db.db_element import Base


class DBNanhuJob(Base):
    """南湖调度作业本地缓存。"""

    __tablename__ = "ri_nanhu_job_table"
    __table_args__ = (
        Index("idx_nanhu_job_status", "status"),
        Index("idx_nanhu_job_cluster", "cluster"),
    )

    primary_id = Column(Integer, primary_key=True)
    # 南湖侧 job 字符串 ID（j-xxxxxx 之类），unique 且 not null
    job_id = Column(String, nullable=False, unique=True)
    # 所属集群（wuxi / x10000 等）
    cluster = Column(String, nullable=False, default="")
    # 作业名称（用户在创建作业时填的 display name）
    name = Column(String, nullable=False, default="")
    # 作业状态（PENDING/RUNNING/SUCCEEDED/FAILED 等南湖原话首字母大写）
    status = Column(String, nullable=False, default="")
    # 南湖项目 UUID（关联 nanhu project）
    project_uuid = Column(String, nullable=False, default="")
    # 算法 / 镜像 UUID
    algorithm_uuid = Column(String, nullable=False, default="")
    # 训练产物输出目录（可被扫描器用作 dump 根目录）
    output_dir = Column(String, nullable=False, default="")
    # 挂载存储配置（JSON 字符串，复杂结构以 JSON 存）
    mounted_storages_json = Column(Text, nullable=False, default="")
    create_time = Column(DateTime, nullable=False, default=datetime.now)
    update_time = Column(
        DateTime, nullable=False, default=datetime.now, onupdate=datetime.now
    )

    def __repr__(self) -> str:
        return (
            f"<DBNanhuJob(primary_id={self.primary_id}, "
            f"job_id={self.job_id!r}, cluster={self.cluster!r}, "
            f"status={self.status!r})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_id": self.primary_id,
            "job_id": self.job_id,
            "cluster": self.cluster,
            "name": self.name,
            "status": self.status,
            "project_uuid": self.project_uuid,
            "algorithm_uuid": self.algorithm_uuid,
            "output_dir": self.output_dir,
            "mounted_storages_json": self.mounted_storages_json or "",
            "create_time": self._dt_iso(self.create_time),
            "update_time": self._dt_iso(self.update_time),
        }

    @staticmethod
    def _dt_iso(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None