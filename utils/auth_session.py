#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight v0.5.1 会话与权限中间件。

提供：
- ``get_session_secret()``        从 config.json 读 ``session_secret``；首次启动
                                  生成 48 字节 url-safe 随机串并落盘 config.json。
- ``install_session_middleware(app)``
                       在 ``app`` 上挂 ``SessionMiddleware``（7 天有效期）。
- ``current_user(request) -> Optional[DBUser]``
                       依赖函数；返回当前请求对应的 ``DBUser`` 或 ``None``。
- ``require_login(request) -> DBUser``
                       依赖函数；未登录 → HTML 路径 302 跳 ``/api/auth/login``，
                       JSON 路径返回 ``{code:401, msg:"login required"}``。
- ``require_admin(request) -> DBUser``
                       依赖函数；非 admin → 403 JSON。
- ``set_login_session(request, user_id)`` / ``clear_session(request)``

注意：
- session 用 ``itsdangerous`` 签名 cookie，签名密钥来自 ``get_session_secret()``。
- 这些是**同步**依赖函数（FastAPI 接受同步函数当依赖），但内部调用
  ``DBUserService.get_by_id`` 是同步 SQLAlchemy Session 调用（不会阻塞 IO）。
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from loguru import logger
from starlette.middleware.sessions import SessionMiddleware

__all__ = [
    "get_session_secret",
    "install_session_middleware",
    "current_user",
    "require_login",
    "require_admin",
    "set_login_session",
    "clear_session",
    "wants_json_response",
]


# 会话 cookie 名（在 SessionMiddleware 里固定为 ``session``，这里只是文档常量）
SESSION_KEY_USER_ID = "user_id"

# require_login / require_admin 的统一返回（避免分散写死）
_LOGIN_REQUIRED_CODE = 401
_LOGIN_REQUIRED_MSG = "login required"
_FORBIDDEN_CODE = 403
_FORBIDDEN_MSG = "admin required"


def get_session_secret(config_path: Optional[str] = None) -> str:
    """从 config.json 读 ``session_secret``；首次启动随机生成并写回。

    Args:
        config_path: 显式 config.json 路径；None 时复用 ``utils.config_loader``
            的解析路径优先级（环境变量 / program_root/config.json）。

    Returns:
        长度 >= 48 字节的 url-safe 随机串。

    Raises:
        FileNotFoundError: 配置文件路径都不存在时（仅在显式传入且文件不存在时抛）。
    """
    from utils.config_loader import _resolve_config_path

    target = _resolve_config_path(config_path)
    p = Path(target)

    secret: str = ""
    if p.is_file():
        try:
            with p.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            secret = (cfg.get("session_secret") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "get_session_secret: 读 config.json 失败 ({}); 将重生成", exc
            )
            secret = ""

    if not secret:
        secret = secrets.token_urlsafe(48)
        try:
            # 写回 config.json（保留已有字段）
            cfg: dict = {}
            if p.is_file():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        cfg = json.load(f)
                except Exception:  # noqa: BLE001
                    cfg = {}
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
            cfg["session_secret"] = secret
            with p.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=4)
            logger.info(
                "get_session_secret: 已生成 session_secret 并落盘 {}", p
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "get_session_secret: 落盘 config.json 失败（仅内存持有）: {}", exc
            )
    return secret


def install_session_middleware(app: Any) -> None:
    """在 ``app`` 上挂 ``SessionMiddleware``（max_age = 7 天）。

    Args:
        app: ``FastAPI`` 实例。
    """
    secret = get_session_secret()
    app.add_middleware(
        SessionMiddleware,
        secret_key=secret,
        max_age=86400 * 7,
        same_site="lax",
        https_only=False,
    )
    logger.info("install_session_middleware: 已挂 SessionMiddleware (max_age=7d)")


# ---------------------------------------------------------------------------
# 辅助：判定请求是否要 JSON（用于 require_login / require_admin 分流）
# ---------------------------------------------------------------------------

def wants_json_response(request: Request) -> bool:
    """基于 Accept 头判断 caller 期望 JSON 而不是 HTML。

    - ``Accept: application/json`` → True
    - ``Accept: text/html`` → False
    - ``Accept: */*`` + 浏览器 UA → False（浏览器想要页面）
    - ``Accept: */*`` + curl/requests UA → True
    """
    accept = (request.headers.get("accept") or "").lower()
    if "application/json" in accept:
        return True
    if "text/html" in accept or "text/*" in accept:
        return False
    # accept = */* 或空
    ua = (request.headers.get("user-agent") or "").lower()
    if ua.startswith("curl/") or ua.startswith("wget/") or ua.startswith(
        "python-requests"
    ) or ua.startswith("httpie/"):
        return True
    if any(tok in ua for tok in ("mozilla", "chrome", "safari", "firefox", "edge", "opera")):
        return False
    # 默认按 JSON 处理（CLI / API 客户端更普遍）
    return True


# ---------------------------------------------------------------------------
# 依赖函数
# ---------------------------------------------------------------------------


def _load_db_user_service():
    """延迟导入 DBUserService；不可用失败返回 None。"""
    try:
        from db.db_user_service import DBUserService

        return DBUserService
    except Exception as exc:  # noqa: BLE001
        logger.warning("_load_db_user_service: 导入失败: {}", exc)
        return None


def current_user(request: Request) -> Optional[Any]:
    """从 ``request.session["user_id"]`` 查 DBUser；没登录返回 None。

    实现说明：
    - 用 ``Depends`` 调用时框架会自动注入 ``Request``；这里直接接收。
    - 查 DB 失败时返回 None（不让权限检查整体崩）。
    """
    user_id = request.session.get(SESSION_KEY_USER_ID)
    if not user_id:
        return None
    svc = _load_db_user_service()
    if svc is None:
        return None
    try:
        user = svc.get_by_id(int(user_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("current_user: DB 查 user_id={} 失败: {}", user_id, exc)
        return None
    if user is None or not user.is_active:
        # session 残留但 DB 已删 / 已禁用 → 清 session
        request.session.pop(SESSION_KEY_USER_ID, None)
        return None
    # 顺便塞到 request.state，模板层可读
    request.state.current_user = user
    return user


def require_login(request: Request) -> Any:
    """依赖：未登录 → 302（HTML）或 401 JSON；否则返回当前 DBUser。

    通过抛 ``HTTPException`` 短路（FastAPI dependency 中不能直接 ``return Response``
    来短路 —— 那只是 dependency 的返回值，会被传给 path operation）。
    """
    user = current_user(request)
    if user is not None:
        return user
    if wants_json_response(request):
        # 401 JSON：保持 ``{code, msg}`` 业务结构
        raise HTTPException(
            status_code=_LOGIN_REQUIRED_CODE,
            detail={
                "code": _LOGIN_REQUIRED_CODE,
                "msg": _LOGIN_REQUIRED_MSG,
            },
        )
    # HTML → 抛 302 重定向到登录页（保留 next=）
    next_path = request.url.path
    if request.url.query:
        next_path = f"{next_path}?{request.url.query}"
    target = f"/api/auth/login?next={next_path}"
    raise HTTPException(
        status_code=status.HTTP_302_FOUND,
        headers={"Location": target},
        detail={"code": 302, "msg": "redirect to login"},
    )


def require_admin(request: Request) -> Any:
    """依赖：未登录 → 同 ``require_login``；非 admin → 403 JSON；否则返回 DBUser。"""
    user = current_user(request)
    if user is None:
        if wants_json_response(request):
            raise HTTPException(
                status_code=_LOGIN_REQUIRED_CODE,
                detail={
                    "code": _LOGIN_REQUIRED_CODE,
                    "msg": _LOGIN_REQUIRED_MSG,
                },
            )
        raise HTTPException(
            status_code=status.HTTP_302_FOUND,
            headers={"Location": "/api/auth/login"},
            detail={"code": 302, "msg": "redirect to login"},
        )
    if (user.role or "") != "admin":
        raise HTTPException(
            status_code=_FORBIDDEN_CODE,
            detail={
                "code": _FORBIDDEN_CODE,
                "msg": _FORBIDDEN_MSG,
            },
        )
    return user


# ---------------------------------------------------------------------------
# session 写入辅助
# ---------------------------------------------------------------------------


def set_login_session(request: Request, user_id: int) -> None:
    """把 user_id 写进 session；FastAPI SessionMiddleware 会在响应时签 cookie。"""
    request.session[SESSION_KEY_USER_ID] = int(user_id)


def clear_session(request: Request) -> None:
    """清掉 user_id（其它自定义 key 保留；如果要全清用 ``request.session.clear()``）。"""
    request.session.pop(SESSION_KEY_USER_ID, None)