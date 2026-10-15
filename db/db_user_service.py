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

from datetime import datetime
from typing import Any, Dict, List, Optional

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
    """v0.5.0 最小实现 + v0.5.1 完整 CRUD。

    提供：
    - ``ensure_admin_seed``（启动期种子 admin）
    - ``create_user``（bcrypt hash + 校验）
    - ``authenticate``（用户名 + 密码 → DBUser 或 None）
    - ``get_by_username`` / ``get_by_id`` / ``list_users``
    - ``update_password`` / ``set_active`` / ``set_last_login``
    - ``count_users`` / ``is_username_taken``
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

    # ------------------------------------------------------------------
    # v0.5.1 CRUD（user list / create / authenticate / update 等）
    # ------------------------------------------------------------------

    @staticmethod
    def _db_user_model():
        try:
            from db.db_user import DBUser

            return DBUser
        except Exception as exc:  # noqa: BLE001
            logger.warning("DBUserService: 导入 DBUser 失败: {}", exc)
            return None

    @staticmethod
    def _session_factory():
        try:
            from db.db_element import Session

            return Session
        except Exception as exc:  # noqa: BLE001
            logger.warning("DBUserService: 导入 Session 失败: {}", exc)
            return None

    @staticmethod
    def count_users() -> int:
        """``ri_user_table`` 行数；DB 不可用时返回 0。"""
        DBUser = DBUserService._db_user_model()
        Session = DBUserService._session_factory()
        if DBUser is None or Session is None:
            return 0
        try:
            with Session() as session:
                return int(session.query(DBUser).count())
        except Exception as exc:  # noqa: BLE001
            logger.exception("count_users 失败: {}", exc)
            return 0

    @staticmethod
    def is_username_taken(username: str) -> bool:
        """用户名是否已存在（区分大小写，DB UNIQUE）。"""
        uname = (username or "").strip()
        if not uname:
            return False
        DBUser = DBUserService._db_user_model()
        Session = DBUserService._session_factory()
        if DBUser is None or Session is None:
            return False
        try:
            with Session() as session:
                return (
                    session.query(DBUser)
                    .filter(DBUser.username == uname)
                    .first()
                    is not None
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("is_username_taken 失败: {}", exc)
            return False

    @staticmethod
    def create_user(
        username: str,
        password: str,
        role: str = "user",
        is_active: bool = True,
    ) -> Optional[Any]:
        """新建一个用户。

        Args:
            username: 必填，非空；最大 64 字符（DB 列宽限制）。
            password: 必填；非空明文密码（内部 bcrypt hash）。
            role: ``admin`` / ``user``；非法时回退到 ``user``。
            is_active: 默认 True。

        Returns:
            ``DBUser`` 实例；失败时返回 ``None``（已记录日志）。

        Raises:
            ValueError: 参数非法（空 username / 空 password）。
        """
        uname = (username or "").strip()
        pwd = password or ""
        if not uname:
            raise ValueError("username 不能为空")
        if not pwd:
            raise ValueError("password 不能为空")
        if role not in ("admin", "user"):
            role = "user"

        DBUser = DBUserService._db_user_model()
        Session = DBUserService._session_factory()
        if DBUser is None or Session is None:
            logger.warning("create_user: DB 不可用，无法创建用户")
            return None
        try:
            with Session() as session:
                # 防御：用户名已存在 → 不抛错，返回 None
                existed = (
                    session.query(DBUser)
                    .filter(DBUser.username == uname)
                    .first()
                )
                if existed is not None:
                    logger.warning(
                        "create_user: 用户名已存在 username={}", uname
                    )
                    return None
                obj = DBUser(
                    username=uname,
                    password_hash=hash_password(pwd),
                    role=role,
                    is_active=is_active,
                )
                session.add(obj)
                session.commit()
                session.refresh(obj)
                logger.info(
                    "create_user: 新建用户 primary_id={} username={} role={}",
                    obj.primary_id,
                    uname,
                    role,
                )
                return obj
        except Exception as exc:  # noqa: BLE001
            logger.exception("create_user 失败: {}", exc)
            return None

    @staticmethod
    def authenticate(username: str, password: str) -> Optional[Any]:
        """校验用户名 + 密码；返回 ``DBUser`` 或 ``None``。

        不区分大小写匹配（用户名 ASCII 比较用 ``func.lower``）。
        失败原因（用户不存在 / 密码错 / 账号禁用）一律返回 None；上层自己
        写日志。
        """
        uname = (username or "").strip()
        pwd = password or ""
        if not uname or not pwd:
            return None
        DBUser = DBUserService._db_user_model()
        Session = DBUserService._session_factory()
        if DBUser is None or Session is None:
            return None
        try:
            from sqlalchemy import func

            with Session() as session:
                row = (
                    session.query(DBUser)
                    .filter(func.lower(DBUser.username) == uname.lower())
                    .first()
                )
                if row is None:
                    return None
                if not row.is_active:
                    logger.info(
                        "authenticate: 账号已禁用 username={}", uname
                    )
                    return None
                if not verify_password(pwd, row.password_hash):
                    return None
                return row
        except Exception as exc:  # noqa: BLE001
            logger.exception("authenticate 失败: {}", exc)
            return None

    @staticmethod
    def get_by_username(username: str) -> Optional[Any]:
        """按用户名精确查一行；不存在返回 ``None``。"""
        uname = (username or "").strip()
        if not uname:
            return None
        DBUser = DBUserService._db_user_model()
        Session = DBUserService._session_factory()
        if DBUser is None or Session is None:
            return None
        try:
            with Session() as session:
                return (
                    session.query(DBUser)
                    .filter(DBUser.username == uname)
                    .first()
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_by_username 失败: {}", exc)
            return None

    @staticmethod
    def get_by_id(pk: int) -> Optional[Any]:
        """按主键查一行；不存在返回 ``None``。"""
        if pk is None:
            return None
        DBUser = DBUserService._db_user_model()
        Session = DBUserService._session_factory()
        if DBUser is None or Session is None:
            return None
        try:
            with Session() as session:
                return (
                    session.query(DBUser)
                    .filter(DBUser.primary_id == pk)
                    .first()
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_by_id 失败: {}", exc)
            return None

    @staticmethod
    def list_users(
        limit: int = 100,
        offset: int = 0,
        role: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> List[Any]:
        """列出用户；可按 role / is_active 过滤。

        按 ``create_time`` 升序（首个 admin 排第一），然后 ``primary_id`` 升序。
        """
        DBUser = DBUserService._db_user_model()
        Session = DBUserService._session_factory()
        if DBUser is None or Session is None:
            return []
        try:
            with Session() as session:
                q = session.query(DBUser)
                if role in ("admin", "user"):
                    q = q.filter(DBUser.role == role)
                if is_active is not None:
                    q = q.filter(DBUser.is_active == is_active)
                return (
                    q.order_by(DBUser.create_time.asc(), DBUser.primary_id.asc())
                    .offset(offset)
                    .limit(limit)
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("list_users 失败: {}", exc)
            return []

    @staticmethod
    def update_password(pk: int, new_password: str) -> bool:
        """改密码（bcrypt hash）；找不到返回 False。

        Raises:
            ValueError: new_password 为空。
        """
        if not new_password:
            raise ValueError("new_password 不能为空")
        DBUser = DBUserService._db_user_model()
        Session = DBUserService._session_factory()
        if DBUser is None or Session is None:
            return False
        try:
            with Session() as session:
                row = (
                    session.query(DBUser)
                    .filter(DBUser.primary_id == pk)
                    .first()
                )
                if row is None:
                    return False
                row.password_hash = hash_password(new_password)
                session.commit()
                logger.info(
                    "update_password: pk={} 已更新", pk
                )
                return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("update_password 失败: {}", exc)
            return False

    @staticmethod
    def set_active(pk: int, is_active: bool) -> bool:
        """激活 / 禁用账号；找不到返回 False。"""
        DBUser = DBUserService._db_user_model()
        Session = DBUserService._session_factory()
        if DBUser is None or Session is None:
            return False
        try:
            with Session() as session:
                row = (
                    session.query(DBUser)
                    .filter(DBUser.primary_id == pk)
                    .first()
                )
                if row is None:
                    return False
                row.is_active = bool(is_active)
                session.commit()
                logger.info(
                    "set_active: pk={} is_active={}", pk, is_active
                )
                return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("set_active 失败: {}", exc)
            return False

    @staticmethod
    def set_last_login(pk: int) -> bool:
        """把 ``last_login_at`` 写为 ``datetime.now()``；找不到返回 False。"""
        if pk is None:
            return False
        DBUser = DBUserService._db_user_model()
        Session = DBUserService._session_factory()
        if DBUser is None or Session is None:
            return False
        try:
            with Session() as session:
                row = (
                    session.query(DBUser)
                    .filter(DBUser.primary_id == pk)
                    .first()
                )
                if row is None:
                    return False
                row.last_login_at = datetime.now()
                session.commit()
                return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("set_last_login 失败: {}", exc)
            return False


def ensure_admin_seed(
    username: str = DEFAULT_ADMIN_USERNAME,
    password: str = DEFAULT_ADMIN_PASSWORD,
    log_warning: bool = True,
) -> Dict[str, Any]:
    """``DBUserService.ensure_admin_seed`` 的函数式快捷入口。"""
    return DBUserService.ensure_admin_seed(
        username=username, password=password, log_warning=log_warning
    )
