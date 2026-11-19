#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight 训练曲线指标偏好 ORM（ri_train_curve_preference_table）。

两级作用域：
- ``scope='global'``：用户级默认按钮组（``task_id`` 为 NULL），配置一次长期生效。
- ``scope='task'``：某个任务追加的额外展示项（``task_id`` 有值），
  与用户默认按钮组合并后作为该任务的最终展示项。

``keys_json`` 存指标 key 列表（JSON 数组）；``custom_keys_json`` 存用户为该
作用域新增的自定义指标（``custom:<tag>``）。
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint

from db.db_element import Base


class DBTrainCurvePreference(Base):
    """训练曲线指标偏好（用户默认 + 任务级追加）。"""

    __tablename__ = "ri_train_curve_preference_table"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "scope",
            "task_id",
            name="uq_train_curve_pref_user_scope_task",
        ),
        Index("idx_train_curve_pref_user", "user_id"),
    )

    primary_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("ri_user_table.primary_id"), nullable=False)
    # 'global'（用户默认） / 'task'（任务级追加）
    scope = Column(String(16), nullable=False, default="global")
    # scope='task' 时指向具体任务；scope='global' 时为 NULL
    task_id = Column(Integer, nullable=True)
    # 指标 key 列表（JSON 数组文本）
    keys_json = Column(Text, nullable=False, default="[]")
    # 自定义指标 key 列表（JSON 数组文本，形如 custom:<tag>）
    custom_keys_json = Column(Text, nullable=False, default="[]")
    create_time = Column(DateTime, nullable=False, default=datetime.now)
    update_time = Column(
        DateTime, nullable=False, default=datetime.now, onupdate=datetime.now
    )

    def __repr__(self) -> str:
        return (
            f"<DBTrainCurvePreference(primary_id={self.primary_id}, "
            f"user_id={self.user_id}, scope={self.scope!r}, task_id={self.task_id})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_id": self.primary_id,
            "user_id": self.user_id,
            "scope": self.scope or "global",
            "task_id": self.task_id,
            "keys": self._load_list(self.keys_json),
            "custom_keys": self._load_list(self.custom_keys_json),
            "create_time": self._dt_iso(self.create_time),
            "update_time": self._dt_iso(self.update_time),
        }

    @staticmethod
    def _load_list(raw: Any) -> List[str]:
        import json

        if isinstance(raw, list):
            return [str(x) for x in raw]
        try:
            payload = json.loads(raw or "[]")
        except (TypeError, ValueError):
            return []
        if not isinstance(payload, list):
            return []
        return [str(x) for x in payload]

    @staticmethod
    def _dt_iso(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None
