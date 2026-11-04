#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight v0.5.1 DBUserClusterConfigService —— per-user 集群配置 CRUD。

每行代表一个用户在一个集群下的 OAuth2 网关凭据 + token 刷新参数；key 为
``(user_id, cluster_key)`` UNIQUE。

字段说明（与 Shell-dsv2/automate/db/db_cluster_config.py 对齐，便于后续
v0.5.3+ 把 nanhu-cli 切到读这张表）：
    - cluster_key          集群关键字（如 wuxi/x10000）
    - display_label        显示名
    - api_base_url         集群 API 网关
    - user_id_field        南湖 userId query 参数
    - app_key / app_secret OAuth2 凭据
    - auth_token_url / token_http_method / token_request_body / token_enabled
                          OAuth2 token 刷新参数（token_request_body 是裸 JSON 文本）
    - create_time / update_time

提供方法（v0.5.1）：
- list_by_user(user_id)
- get_by_user_and_key(user_id, cluster_key)
- upsert(user_id, cluster_key, ...)
- delete_by_id(pk, user_id)   仅删自己的
- delete_by_user_and_key(user_id, cluster_key)  便捷入口
- list_need_token_refresh()  给将来的 v0.5.3 token 刷新调度用；v0.5.1 不接入

实现风格参考 db_attempt_service.py / db_nanhu_job_service.py（@staticmethod）。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import and_

from db.db_element import Session
from db.db_user_cluster_config import DBUserClusterConfig


# 可写字段白名单（用于 upsert；避免 caller 塞 schema 之外字段）
_UPSERT_FIELDS: tuple = (
    "display_label",
    "api_base_url",
    "user_id_field",
    "app_key",
    "app_secret",
    "auth_token_url",
    "token_http_method",
    "token_request_body",
    "token_enabled",
)


class DBUserClusterConfigService:
    """``ri_user_cluster_config_table`` 的读写服务。"""

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @staticmethod
    def list_by_user(user_id: int) -> List[DBUserClusterConfig]:
        """列出某用户的所有 cluster 配置，按 ``cluster_key`` 升序。"""
        if user_id is None:
            return []
        try:
            with Session() as session:
                return (
                    session.query(DBUserClusterConfig)
                    .filter(DBUserClusterConfig.user_id == user_id)
                    .order_by(DBUserClusterConfig.cluster_key.asc())
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("DBUserClusterConfigService.list_by_user 失败: {}", exc)
            return []

    @staticmethod
    def get_by_user_and_key(
        user_id: int,
        cluster_key: str,
    ) -> Optional[DBUserClusterConfig]:
        """按 (user_id, cluster_key) 查一行；不存在返回 None。"""
        if user_id is None or not cluster_key:
            return None
        key = (cluster_key or "").strip()
        if not key:
            return None
        try:
            with Session() as session:
                return (
                    session.query(DBUserClusterConfig)
                    .filter(
                        and_(
                            DBUserClusterConfig.user_id == user_id,
                            DBUserClusterConfig.cluster_key == key,
                        )
                    )
                    .first()
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_by_user_and_key 失败: {}", exc)
            return None

    @staticmethod
    def get_by_id(pk: int, user_id: int) -> Optional[DBUserClusterConfig]:
        """按主键查一行（仅限 user_id == 自己）；找不到返回 None。

        多传 user_id 参数是为了防止越权读取（v0.5.1 仅自己读写自己的）。
        """
        if pk is None or user_id is None:
            return None
        try:
            with Session() as session:
                return (
                    session.query(DBUserClusterConfig)
                    .filter(
                        and_(
                            DBUserClusterConfig.primary_id == pk,
                            DBUserClusterConfig.user_id == user_id,
                        )
                    )
                    .first()
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_by_id 失败: {}", exc)
            return None

    @staticmethod
    def list_need_token_refresh() -> List[DBUserClusterConfig]:
        """列出 ``token_enabled == True`` 的所有行（v0.5.3 给 token 刷新调度用）。

        v0.5.1 不接入调度器；这里先把查询 API 暴露，便于调试 / CLI 演示。
        """
        try:
            with Session() as session:
                return (
                    session.query(DBUserClusterConfig)
                    .filter(DBUserClusterConfig.token_enabled.is_(True))
                    .order_by(DBUserClusterConfig.user_id.asc())
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("list_need_token_refresh 失败: {}", exc)
            return []

    @staticmethod
    def list_all() -> List[DBUserClusterConfig]:
        """列出全部用户的集群配置（按 user_id / cluster_key 升序）。"""
        try:
            with Session() as session:
                return (
                    session.query(DBUserClusterConfig)
                    .order_by(
                        DBUserClusterConfig.user_id.asc(),
                        DBUserClusterConfig.cluster_key.asc(),
                    )
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("DBUserClusterConfigService.list_all 失败: {}", exc)
            return []

    @staticmethod
    def resolve_for_cluster(
        cluster: str,
        user_id: Optional[int] = None,
    ) -> Optional[DBUserClusterConfig]:
        """按集群名解析一条可用凭据（须含 app_key + app_secret）。

        匹配顺序：
        1. ``user_id`` 给定时：精确 ``(user_id, cluster_key)``
        2. 该用户下前缀匹配（如 ``x10000`` ↔ ``x10000_v5000``）
        3. 全局精确 / 前缀匹配（后台任务或未传 user_id 时）
        """
        name = (cluster or "").strip()
        if not name:
            return None

        def _has_creds(row: Optional[DBUserClusterConfig]) -> bool:
            return bool(
                row is not None
                and (getattr(row, "app_key", None) or "").strip()
                and (getattr(row, "app_secret", None) or "").strip()
            )

        def _matches(key: str) -> bool:
            k = (key or "").strip().lower()
            c = name.lower()
            if not k:
                return False
            if k == c:
                return True
            if k.startswith(c + "_") or k.startswith(c + "."):
                return True
            if c.startswith(k + "_") or c.startswith(k + "."):
                return True
            return False

        if user_id is not None:
            exact = DBUserClusterConfigService.get_by_user_and_key(user_id, name)
            if _has_creds(exact):
                return exact
            for row in DBUserClusterConfigService.list_by_user(user_id):
                if _matches(row.cluster_key) and _has_creds(row):
                    return row

        # 全局：先精确，再前缀
        candidates = DBUserClusterConfigService.list_all()
        for row in candidates:
            if (row.cluster_key or "").strip().lower() == name.lower() and _has_creds(row):
                return row
        for row in candidates:
            if _matches(row.cluster_key) and _has_creds(row):
                return row
        return None

    # ------------------------------------------------------------------
    # upsert / delete
    # ------------------------------------------------------------------

    @staticmethod
    def upsert(
        user_id: int,
        cluster_key: str,
        display_label: str = "",
        api_base_url: str = "",
        user_id_field: str = "",
        app_key: str = "",
        app_secret: str = "",
        auth_token_url: str = "",
        token_http_method: str = "POST",
        token_request_body: str = "",
        token_enabled: bool = False,
    ) -> Optional[DBUserClusterConfig]:
        """按 ``(user_id, cluster_key)`` upsert 一行。

        不接收 cluster_key / user_id 在 fields 之外；显示校验：cluster_key
        非空；display_label / api_base_url 在新建时必填；update 时允许留空
        （保留原值）。
        """
        key = (cluster_key or "").strip()
        if not key:
            logger.warning("upsert: cluster_key 为空，拒绝写入")
            return None
        if user_id is None:
            logger.warning("upsert: user_id 为空，拒绝写入")
            return None

        # token_http_method 大写归一
        method = (token_http_method or "POST").upper()
        if method not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            method = "POST"

        try:
            with Session() as session:
                existing: Optional[DBUserClusterConfig] = (
                    session.query(DBUserClusterConfig)
                    .filter(
                        and_(
                            DBUserClusterConfig.user_id == user_id,
                            DBUserClusterConfig.cluster_key == key,
                        )
                    )
                    .first()
                )
                if existing is None:
                    # 新建：display_label / api_base_url 必须有
                    if not (display_label or "").strip():
                        logger.warning(
                            "upsert(new): display_label 为空，拒绝写入 cluster_key={}",
                            key,
                        )
                        return None
                    if not (api_base_url or "").strip():
                        logger.warning(
                            "upsert(new): api_base_url 为空，拒绝写入 cluster_key={}",
                            key,
                        )
                        return None
                    row = DBUserClusterConfig(
                        user_id=user_id,
                        cluster_key=key,
                        display_label=(display_label or "").strip(),
                        api_base_url=(api_base_url or "").strip(),
                        user_id_field=(user_id_field or "").strip(),
                        app_key=(app_key or "").strip(),
                        app_secret=(app_secret or "").strip(),
                        auth_token_url=(auth_token_url or "").strip(),
                        token_http_method=method,
                        token_request_body=(token_request_body or "").strip(),
                        token_enabled=bool(token_enabled),
                    )
                    session.add(row)
                    session.commit()
                    session.refresh(row)
                    logger.info(
                        "upsert: 新建 cluster_config pk={} user_id={} cluster_key={}",
                        row.primary_id,
                        user_id,
                        key,
                    )
                    return row

                # 已存在：仅把非空值写回（空值 = "未填" → 不覆盖）
                if display_label.strip():
                    existing.display_label = display_label.strip()
                if api_base_url.strip():
                    existing.api_base_url = api_base_url.strip()
                if user_id_field.strip():
                    existing.user_id_field = user_id_field.strip()
                if app_key.strip():
                    existing.app_key = app_key.strip()
                if app_secret.strip():
                    existing.app_secret = app_secret.strip()
                if auth_token_url.strip():
                    existing.auth_token_url = auth_token_url.strip()
                if token_request_body.strip():
                    existing.token_request_body = token_request_body.strip()
                existing.token_http_method = method
                # token_enabled 是 bool，前端表单可能勾选 false；写回最新值
                existing.token_enabled = bool(token_enabled)
                existing.update_time = datetime.now()
                session.commit()
                session.refresh(existing)
                logger.info(
                    "upsert: 更新 cluster_config pk={} user_id={} cluster_key={}",
                    existing.primary_id,
                    user_id,
                    key,
                )
                return existing
        except Exception as exc:  # noqa: BLE001
            logger.exception("upsert cluster_config 失败: {}", exc)
            return None

    @staticmethod
    def delete_by_id(pk: int, user_id: int) -> bool:
        """按主键删除一行（仅当 user_id == 自己）；找不到返回 False。"""
        if pk is None or user_id is None:
            return False
        try:
            with Session() as session:
                row = (
                    session.query(DBUserClusterConfig)
                    .filter(
                        and_(
                            DBUserClusterConfig.primary_id == pk,
                            DBUserClusterConfig.user_id == user_id,
                        )
                    )
                    .first()
                )
                if row is None:
                    return False
                session.delete(row)
                session.commit()
                logger.info(
                    "delete_by_id: 删除 cluster_config pk={} user_id={}",
                    pk,
                    user_id,
                )
                return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("delete_by_id 失败: {}", exc)
            return False

    @staticmethod
    def delete_by_user_and_key(user_id: int, cluster_key: str) -> bool:
        """按 (user_id, cluster_key) 删除（仅删自己）；便捷入口。"""
        row = DBUserClusterConfigService.get_by_user_and_key(user_id, cluster_key)
        if row is None:
            return False
        return DBUserClusterConfigService.delete_by_id(row.primary_id, user_id)