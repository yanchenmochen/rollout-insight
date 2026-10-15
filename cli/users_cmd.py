"""CLI 子命令：users（v0.5.1 T12）。

- ``users list``                       表格打印所有用户
- ``users create --username X --password Y [--role admin|user]``
                                       新建用户（默认 role=user）
- ``users set-password --username X --password Y``  改密码
- ``users activate --username X``     启用账号
- ``users deactivate --username X``   禁用账号

实现说明：
- 直接调 ``db.db_user_service.DBUserService``（不走 HTTP），CLI 通常在本机跑
- 写操作默认 dry-run（打印 plan）；加 ``--submit`` 才真正落库
- ``deactivate`` / ``activate`` 永远要 ``--submit``（避免 agent 误触）
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 延迟导入：与 controller 同款
# ---------------------------------------------------------------------------


def _user_service():
    try:
        from db.db_user_service import DBUserService

        return DBUserService
    except Exception as exc:  # noqa: BLE001
        return _MissingService(exc)


class _MissingService:
    """DBUserService 不可用时的占位，所有方法返回 None / []。"""

    def __init__(self, exc: Exception):
        self._exc = exc

    def list_users(self):
        return []

    def get_by_username(self, _username: str):
        return None

    def create_user(self, **_kw):
        return None

    def update_password(self, *_args, **_kw):
        return False

    def set_active(self, *_args, **_kw):
        return False


def _dt_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------


def list_users() -> Dict[str, Any]:
    """列出所有用户（表格 / JSON 共享入口）。"""
    svc = _user_service()
    rows = svc.list_users(limit=500) if svc else []
    items: List[Dict[str, Any]] = []
    for r in rows:
        items.append(
            {
                "primary_id": getattr(r, "primary_id", None),
                "username": getattr(r, "username", ""),
                "role": getattr(r, "role", "user"),
                "is_active": bool(getattr(r, "is_active", True)),
                "create_time": _dt_iso(getattr(r, "create_time", None)),
                "last_login_at": _dt_iso(getattr(r, "last_login_at", None)),
            }
        )
    return {
        "code": 0,
        "msg": "ok",
        "data": {"users": items, "count": len(items)},
    }


def create_user(
    username: str,
    password: str,
    role: str = "user",
    submit: bool = False,
) -> Dict[str, Any]:
    """新建用户。

    Args:
        username: 必填
        password: 必填（明文；落库前 bcrypt hash）
        role: ``admin`` / ``user``，默认 ``user``
        submit: True → 真创建；False → dry-run（仅打印 plan）
    """
    svc = _user_service()
    plan: Dict[str, Any] = {
        "action": "create user",
        "username": username,
        "role": role,
        "submit": submit,
        "applied": False,
    }
    if not submit:
        plan["dry_run"] = True
        plan["note"] = "dry-run：未创建用户；加 --submit 才执行"
        return plan
    if isinstance(svc, _MissingService):
        return {
            "code": 50002,
            "msg": f"DBUserService 不可用: {svc._exc}",
            "data": plan,
        }
    new_user = svc.create_user(username=username, password=password, role=role)
    if new_user is None:
        return {
            "code": 50003,
            "msg": "创建失败（用户名重复或 DB 错误）",
            "data": plan,
        }
    plan["applied"] = True
    plan["primary_id"] = getattr(new_user, "primary_id", None)
    return {"code": 0, "msg": "ok", "data": plan}


def set_password(
    username: str,
    password: str,
    submit: bool = False,
) -> Dict[str, Any]:
    """改密码。"""
    svc = _user_service()
    plan: Dict[str, Any] = {
        "action": "set-password",
        "username": username,
        "submit": submit,
        "applied": False,
    }
    if not submit:
        plan["dry_run"] = True
        plan["note"] = "dry-run：未改密码；加 --submit 才执行"
        return plan
    if isinstance(svc, _MissingService):
        return {
            "code": 50002,
            "msg": f"DBUserService 不可用: {svc._exc}",
            "data": plan,
        }
    user = svc.get_by_username(username)
    if user is None:
        return {"code": 404, "msg": f"用户不存在: {username}", "data": plan}
    ok = svc.update_password(int(user.primary_id), password)
    if not ok:
        return {"code": 50003, "msg": "改密码失败", "data": plan}
    plan["applied"] = True
    plan["primary_id"] = getattr(user, "primary_id", None)
    return {"code": 0, "msg": "ok", "data": plan}


def set_active(
    username: str,
    is_active: bool,
    submit: bool = False,
) -> Dict[str, Any]:
    """激活 / 禁用账号。"""
    svc = _user_service()
    plan: Dict[str, Any] = {
        "action": "activate" if is_active else "deactivate",
        "username": username,
        "is_active": is_active,
        "submit": submit,
        "applied": False,
    }
    if not submit:
        plan["dry_run"] = True
        plan["note"] = "dry-run：未改变状态；加 --submit 才执行"
        return plan
    if isinstance(svc, _MissingService):
        return {
            "code": 50002,
            "msg": f"DBUserService 不可用: {svc._exc}",
            "data": plan,
        }
    user = svc.get_by_username(username)
    if user is None:
        return {"code": 404, "msg": f"用户不存在: {username}", "data": plan}
    ok = svc.set_active(int(user.primary_id), is_active)
    if not ok:
        return {"code": 50003, "msg": "改状态失败", "data": plan}
    plan["applied"] = True
    plan["primary_id"] = getattr(user, "primary_id", None)
    return {"code": 0, "msg": "ok", "data": plan}