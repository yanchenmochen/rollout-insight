#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight v0.5.0 per-user 集群配置表 ORM（ri_user_cluster_config_table）。

字段说明（plan T7 / T0）：
- primary_id           PK
- user_id              FK → ri_user_table.primary_id
- cluster_key          集群关键字（如 wuxi/x10000）；(user_id, cluster_key) UNIQUE
- display_label        显示名
- api_base_url         集群 API 网关
- user_id_field        南湖 userId
- app_key / app_secret OAuth2 凭据
- auth_token_url / token_http_method / token_request_body / token_enabled
                      OAuth2 token 刷新配置（token_request_body 是裸 JSON 文本）
- create_time / update_time

**不做 owner_user_id 关联到 task/rollout**（p15_b）：所有认证用户都能读写所有
业务数据。

v0.5.0 启动期：
- Base.metadata.create_all 自动建表
- service 由 Worker B（v0.5.1）实现；本文件只声明 ORM
"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
)

from db.db_element import Base


class DBUserClusterConfig(Base):
    """每用户在每个集群下的 OAuth2 网关凭据与刷新参数。"""

    __tablename__ = "ri_user_cluster_config_table"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "cluster_key",
            name="uq_user_cluster_user_cluster_key",
        ),
        Index("idx_user_cluster_user_id", "user_id"),
    )

    primary_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer,
        ForeignKey("ri_user_table.primary_id"),
        nullable=False,
    )
    cluster_key = Column(String(32), nullable=False)
    display_label = Column(String(128), nullable=False, default="")
    api_base_url = Column(String(512), nullable=False, default="")
    user_id_field = Column(String(64), nullable=False, default="")
    app_key = Column(String(128), nullable=False, default="")
    app_secret = Column(String(255), nullable=False, default="")
    auth_token_url = Column(String(512), nullable=False, default="")
    token_http_method = Column(String(8), nullable=False, default="POST")
    token_request_body = Column(Text, nullable=False, default="")
    token_enabled = Column(Boolean, nullable=False, default=True)
    create_time = Column(DateTime, nullable=False, default=datetime.now)
    update_time = Column(
        DateTime, nullable=False, default=datetime.now, onupdate=datetime.now
    )

    def __repr__(self) -> str:
        return (
            f"<DBUserClusterConfig(primary_id={self.primary_id}, user_id={self.user_id}, "
            f"cluster_key={self.cluster_key!r})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        # 默认不外露 app_secret（与 Worker B service 对齐）
        return {
            "primary_id": self.primary_id,
            "user_id": self.user_id,
            "cluster_key": self.cluster_key,
            "display_label": self.display_label or "",
            "api_base_url": self.api_base_url or "",
            "user_id_field": self.user_id_field or "",
            "app_key": self.app_key or "",
            "app_secret": self.app_secret or "",
            "auth_token_url": self.auth_token_url or "",
            "token_http_method": self.token_http_method or "POST",
            "token_request_body": self.token_request_body or "",
            "token_enabled": bool(self.token_enabled),
            "create_time": self._dt_iso(self.create_time),
            "update_time": self._dt_iso(self.update_time),
        }

    @staticmethod
    def _dt_iso(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None
