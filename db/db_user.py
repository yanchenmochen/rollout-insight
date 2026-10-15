#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight v0.5.0 用户表 ORM（ri_user_table）。

字段说明（plan T7 / T0）：
- primary_id    PK
- username      UNIQUE
- password_hash bcrypt 哈希（passlib）
- role          admin | user
- is_active     Boolean
- create_time   DateTime
- last_login_at DateTime，可空

v0.5.0 启动期：
- Base.metadata.create_all 自动建表
- ensure_admin_seed 检查表空则插入 admin/admin123
- service 由 Worker B（v0.5.1）实现；本文件只声明 ORM
"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, Column, DateTime, Integer, String

from db.db_element import Base


class DBUser(Base):
    """用户名 + bcrypt + 角色 + 创建/登录时间。"""

    __tablename__ = "ri_user_table"
    # 注意：username UNIQUE 通过 __table_args__ 显式声明，idx_ri_user_username 同步建
    __table_args__ = (
        # 唯一约束 + 索引
        {"sqlite_autoincrement": True},
    )

    primary_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False, default="")
    role = Column(String(16), nullable=False, default="user")
    is_active = Column(Boolean, nullable=False, default=True)
    create_time = Column(DateTime, nullable=False, default=datetime.now)
    last_login_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<DBUser(primary_id={self.primary_id}, username={self.username!r}, "
            f"role={self.role!r}, is_active={self.is_active})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_id": self.primary_id,
            "username": self.username,
            # password_hash 不外露
            "role": self.role or "user",
            "is_active": bool(self.is_active),
            "create_time": self._dt_iso(self.create_time),
            "last_login_at": self._dt_iso(self.last_login_at),
        }

    @staticmethod
    def _dt_iso(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None
