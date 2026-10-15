#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: DBUserService —— v0.5.0 阶段 1（最小：仅 ``ensure_admin_seed`` + bcrypt 工具）。

v0.5.0 启动期需要：
1. 清空 ``ri_segment_turn_table``（在 utils.bootstrap）
2. 调用本服务的 ``ensure_admin_seed``：若 ``ri_user_table`` 为空则插入 admin
3. WARNING 一次提醒生产环境改密码

Worker B（v0.5.1）会补全：
- authenticate / create_user / get_by_username / list_users /
  update_password / deactivate 等 CRUD

实现说明：
- requirements 里写了 ``passlib[bcrypt]``，但 passlib 1.7.x 与 bcrypt 5.x 的
  ``__about__.__version__`` 检测有兼容问题（passlib 太久没更新，已知
  issue #175）。这里改用 ``bcrypt`` 直接调 ``hashpw`` / ``checkpw``，
  生成 ``$2b$`` 前缀的哈希；passlib 仍保留依赖以备 Worker B 用到。
"""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger


__all__ = [
    "DBUserService",
    "ensure_admin_seed",
    "hash_password",
    "verify_password",
]


# 默认种子账号（仅当 user 表为空时使用；启动日志会 WARNING 提醒）
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"


def _get_bcrypt():
    """延迟导入 bcrypt；失败抛 RuntimeError 让上层感知。"""
    try:
        import bcrypt  # type: ignore

        return bcrypt
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"bcrypt 不可用（v0.5.0 依赖 passlib[bcrypt] / itsdangerous）: {exc}"
        ) from exc


def hash_password(password: str) -> str:
    """用 bcrypt 哈希一个明文密码（生成 ``$2b$`` 前缀）。

    长度 > 72 字节的密码先做 utf-8 截断（bcrypt 协议限制）。
    """
    if not isinstance(password, str) or not password:
        raise ValueError("password 必须是非空字符串")
    bcrypt = _get_bcrypt()
    encoded = password.encode("utf-8")
    if len(encoded) > 72:
        encoded = encoded[:72]
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(encoded, salt).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    """校验明文与 bcrypt 哈希是否匹配。"""
    if not password or not hashed:
        return False
    bcrypt = _get_bcrypt()
    try:
        encoded = password.encode("utf-8")
        if len(encoded) > 72:
            encoded = encoded[:72]
        return bcrypt.checkpw(encoded, hashed.encode("ascii"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("verify_password 失败: {}", exc)
        return False


class DBUserService:
    """v0.5.0 最小实现：仅提供 ``ensure_admin_seed``。

    Worker B（v0.5.1）会扩展这个类：补上 ``create_user / authenticate /
    list_users / update_password / deactivate``。
    """

    @staticmethod
    def ensure_admin_seed(
        username: str = DEFAULT_ADMIN_USERNAME,
        password: str = DEFAULT_ADMIN_PASSWORD,
        log_warning: bool = True,
    ) -> Dict[str, Any]:
        """若 ``ri_user_table`` 为空则插入 admin / bcrypt('admin123') / role='admin'。

        Args:
            username: 默认 admin
            password: 默认 admin123（启动期应在日志 WARNING 一次提醒改密码）
            log_warning: True 时打印一次 WARNING（生产环境提醒）

        Returns:
            {
                "seeded": bool,        # 是否本次写入了 admin 行
                "user_count": int,     # 操作后 user 表行数
                "username": str,
            }
        """
        out: Dict[str, Any] = {
            "seeded": False,
            "user_count": 0,
            "username": username,
        }
        # 延迟导入 ORM，避免循环
        try:
            from db.db_element import Session
            from db.db_user import DBUser
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ensure_admin_seed: db ORM 不可用（v0.5.0 ORM 模块未就绪？）: {}",
                exc,
            )
            return out

        try:
            with Session() as session:
                user_count = (
                    session.query(DBUser).count()
                )
                if user_count > 0:
                    out["user_count"] = int(user_count)
                    return out

                admin = DBUser(
                    username=username,
                    password_hash=hash_password(password),
                    role="admin",
                    is_active=True,
                )
                session.add(admin)
                session.commit()
                session.refresh(admin)
                out["seeded"] = True
                out["user_count"] = 1
                out["username"] = username
                if log_warning:
                    logger.warning(
                        "[bootstrap] 已种 admin 默认账号 (username={}, password={})；"
                        "⚠️  请在生产环境尽快改密码（Worker B v0.5.1 将开放"
                        "users set-password CLI / 登录接口）。",
                        username,
                        password,
                    )
                else:
                    logger.info(
                        "[bootstrap] ensure_admin_seed: 写入 admin 行 primary_id={}",
                        admin.primary_id,
                    )
                return out
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "ensure_admin_seed: DB 操作失败: {}", exc
            )
            return out


def ensure_admin_seed(
    username: str = DEFAULT_ADMIN_USERNAME,
    password: str = DEFAULT_ADMIN_PASSWORD,
    log_warning: bool = True,
) -> Dict[str, Any]:
    """``DBUserService.ensure_admin_seed`` 的函数式快捷入口。"""
    return DBUserService.ensure_admin_seed(
        username=username, password=password, log_warning=log_warning
    )
