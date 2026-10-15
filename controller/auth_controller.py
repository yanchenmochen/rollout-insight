#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight v0.5.1 登录 / 注册 / 登出 / me 控制器。

路由（**不**加 router-level require_login）：
- GET  /api/auth/login      渲染 templates/login.html
- POST /api/auth/login      支持 form + JSON 两种 body；
                            校验 → 写 session → 重定向 / 或返回 JSON
- GET  /api/auth/register   渲染 templates/register.html；
                            仅当 ``ri_user_table`` 行数 == 1（仅 admin）时开放，
                            否则 302 → /api/auth/login
- POST /api/auth/register   创建用户（首位 = admin，其余 = user）→ 自动登录
- POST /api/auth/logout     清 session → 重定向 /
- GET  /api/auth/me         返回当前用户 dict（用于顶栏）

设计：
- 表单提交（浏览器）→ 重定向到 next= 或 /；JSON 提交 → JSON 响应
- 任何依赖都通过 ``Depends`` 显式写，不放 router 级别（避免和 require_login 打架）
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from loguru import logger
from starlette.templating import Jinja2Templates

from data.response import Response
from data.rollout_insight_exception import RolloutInsightException
from utils.auth_session import (
    clear_session,
    current_user,
    set_login_session,
)
from utils.config_loader import config

# 注意：本 router 不加 ``dependencies=[Depends(require_login)]``，
# 否则 login/register/me 会被自己的依赖挡掉。
router = APIRouter()

templates = Jinja2Templates(directory=str(config.templates_dir))


# ---------------------------------------------------------------------------
# 延迟导入辅助（与 task_controller 同款）
# ---------------------------------------------------------------------------


def _safe_import_attr(module_name: str, attr: str):
    try:
        import importlib

        mod = importlib.import_module(module_name)
        return getattr(mod, attr, None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("延迟导入 {}.{} 失败: {}", module_name, attr, exc)
        return None


def _user_service():
    return _safe_import_attr("db.db_user_service", "DBUserService")


# ---------------------------------------------------------------------------
# 路由：login
# ---------------------------------------------------------------------------


@router.get("/api/auth/login", include_in_schema=False)
async def page_login(
    request: Request,
    next: Optional[str] = Query(default=None, description="登录成功后跳转"),
) -> HTMLResponse:
    """渲染 ``templates/login.html``。

    已登录用户访问：直接 302 到 ``next`` 或 ``/api/tasks``，避免重复登录。
    """
    user = current_user(request)
    if user is not None:
        target = next or "/api/tasks"
        return RedirectResponse(url=target, status_code=302)
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "page_title": "登录 - Rollout Insight",
            "next": next or "",
            "register_open": _register_open(),
            "error": None,
            "username": "",
        },
    )


@router.post("/api/auth/login")
async def api_login(
    request: Request,
    next: Optional[str] = Query(default=None, description="登录成功后跳转"),
) -> Response:
    """登录端点（兼容 form + JSON 两种 body）。"""
    svc = _user_service()
    if svc is None:
        return Response.error(
            code=50002, msg="DBUserService 不可用，请确认 db_user_service.py 已就绪"
        )

    body = await _read_body(request)
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""

    if not username or not password:
        return Response.error(code=50001, msg="用户名和密码不能为空")

    user = svc.authenticate(username, password)
    if user is None:
        logger.info("login 失败 username={}", username)
        # HTML 表单提交时希望返回渲染好的页面（带红 banner），而不是 JSON
        accept = (request.headers.get("accept") or "").lower()
        if "text/html" in accept or "text/*" in accept:
            return templates.TemplateResponse(
                request,
                "login.html",
                {
                    "page_title": "登录 - Rollout Insight",
                    "next": next or "",
                    "register_open": _register_open(),
                    "error": "用户名或密码错误",
                    "username": username,
                },
                status_code=200,
            )
        return Response.error(code=401, msg="用户名或密码错误")

    # 写 session + 更新 last_login_at
    set_login_session(request, int(user.primary_id))
    try:
        svc.set_last_login(int(user.primary_id))
    except Exception as exc:  # noqa: BLE001
        logger.debug("set_last_login 失败（不影响登录）: {}", exc)

    target = next or "/api/tasks"
    accept = (request.headers.get("accept") or "").lower()
    wants_html = (
        "text/html" in accept or "text/*" in accept
    ) and not (accept and "json" in accept and "text/html" not in accept)
    if wants_html:
        return RedirectResponse(url=target, status_code=302)

    return Response.success(
        msg="登录成功",
        data={
            "user_id": user.primary_id,
            "username": user.username,
            "role": user.role,
            "redirect": target,
        },
    )


# ---------------------------------------------------------------------------
# 路由：register
# ---------------------------------------------------------------------------


def _register_open() -> bool:
    """注册是否开放：仅当 ``ri_user_table`` 行数 == 1（仅 admin）时返回 True。

    注：>=2 行时不允许通过 /register 页面建新账号（须 admin 在 CLI / UI 添加）。
    """
    svc = _user_service()
    if svc is None:
        return False
    try:
        return svc.count_users() == 1
    except Exception:  # noqa: BLE001
        return False


@router.get("/api/auth/register", include_in_schema=False)
async def page_register(
    request: Request,
    next: Optional[str] = Query(default=None),
) -> HTMLResponse:
    """渲染 ``templates/register.html``。

    不在开放条件（行数 != 1）→ 302 跳 /api/auth/login；
    已登录用户访问 → 跳首页（不必重复注册）。
    """
    user = current_user(request)
    if user is not None:
        return RedirectResponse(url=next or "/api/tasks", status_code=302)
    if not _register_open():
        return RedirectResponse(url="/api/auth/login", status_code=302)
    return templates.TemplateResponse(
        request,
        "register.html",
        {
            "page_title": "注册 - Rollout Insight",
            "next": next or "",
            "error": None,
            "username": "",
        },
    )


@router.post("/api/auth/register")
async def api_register(
    request: Request,
    next: Optional[str] = Query(default=None),
) -> Response:
    """注册端点。

    规则：
    - register 须开放（``ri_user_table`` 行数 == 1）才允许；否则 403。
    - 首位注册（admin 不算）拿 ``admin`` 角色（保留未来扩展）；第二位起 = ``user``。
    - 创建成功后自动登录（写 session）。
    """
    if not _register_open():
        return Response.error(code=403, msg="注册通道已关闭（请联系管理员）")

    svc = _user_service()
    if svc is None:
        return Response.error(code=50002, msg="DBUserService 不可用")

    body = await _read_body(request)
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    password_confirm = body.get("password_confirm") or ""

    if not username:
        return Response.error(code=50001, msg="用户名不能为空")
    if not _valid_username(username):
        return Response.error(
            code=50001,
            msg="用户名只能包含字母、数字、点、下划线、连字符，长度 3~32",
        )
    if not password or len(password) < 6:
        return Response.error(code=50001, msg="密码长度至少 6 位")
    if password != password_confirm:
        return Response.error(code=50001, msg="两次密码输入不一致")
    if svc.is_username_taken(username):
        return Response.error(code=50001, msg="用户名已被占用")

    # 决定 role：register_open 意味着 user 表仅 1 行（admin），本注册即第 2 个用户；
    # 我们按 plan "首位 = admin，其余 = user" 的约定：当前允许 admin 的存在，
    # 所以新建的就是普通 user。
    role = "user"
    new_user = svc.create_user(
        username=username,
        password=password,
        role=role,
        is_active=True,
    )
    if new_user is None:
        # 可能是 unique 冲突或 DB 写入失败
        accept = (request.headers.get("accept") or "").lower()
        if "text/html" in accept or "text/*" in accept:
            return templates.TemplateResponse(
                request,
                "register.html",
                {
                    "page_title": "注册 - Rollout Insight",
                    "next": next or "",
                    "error": "注册失败（用户名重复或 DB 错误）",
                    "username": username,
                },
                status_code=200,
            )
        return Response.error(code=50003, msg="注册失败")

    # 自动登录
    set_login_session(request, int(new_user.primary_id))
    try:
        svc.set_last_login(int(new_user.primary_id))
    except Exception:  # noqa: BLE001
        pass

    target = next or "/api/tasks"
    accept = (request.headers.get("accept") or "").lower()
    wants_html = (
        "text/html" in accept or "text/*" in accept
    ) and not (accept and "json" in accept and "text/html" not in accept)
    if wants_html:
        return RedirectResponse(url=target, status_code=302)
    return Response.success(
        msg="注册成功",
        data={
            "user_id": new_user.primary_id,
            "username": new_user.username,
            "role": new_user.role,
            "redirect": target,
        },
    )


def _valid_username(username: str) -> bool:
    """用户名：3~32 字符，字母/数字/点/下划线/连字符。"""
    if not username or len(username) < 3 or len(username) > 32:
        return False
    allowed = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    return all(ch in allowed for ch in username)


# ---------------------------------------------------------------------------
# 路由：logout
# ---------------------------------------------------------------------------


@router.post("/api/auth/logout")
async def api_logout(request: Request) -> Response:
    """登出：清 session，返回 /api/tasks。"""
    clear_session(request)
    accept = (request.headers.get("accept") or "").lower()
    wants_html = (
        "text/html" in accept or "text/*" in accept
    ) and not (accept and "json" in accept and "text/html" not in accept)
    if wants_html:
        return RedirectResponse(url="/api/tasks", status_code=302)
    return Response.success(msg="已登出", data={"redirect": "/api/tasks"})


# ---------------------------------------------------------------------------
# 路由：me
# ---------------------------------------------------------------------------


@router.get("/api/auth/me")
async def api_me(request: Request) -> Response:
    """返回当前用户 dict；未登录返回 401 JSON。"""
    user = current_user(request)
    if user is None:
        return Response.error(code=401, msg="login required")
    return Response.success(
        msg="ok",
        data={
            "user_id": user.primary_id,
            "username": user.username,
            "role": user.role,
            "is_active": bool(user.is_active),
        },
    )


# ---------------------------------------------------------------------------
# body 解析辅助（form + JSON 兼容）
# ---------------------------------------------------------------------------


async def _read_body(request: Request) -> Dict[str, Any]:
    """同时支持 form-encoded 和 JSON 两种 body。

    - 优先按 ``Content-Type`` 解析
    - 表单字段也允许 ``username`` / ``password`` / ``password_confirm``
    """
    content_type = (request.headers.get("content-type") or "").lower()
    out: Dict[str, Any] = {}
    try:
        if "application/json" in content_type:
            raw = await request.body()
            if raw:
                payload = json.loads(raw.decode("utf-8"))
                if isinstance(payload, dict):
                    out.update(payload)
        elif (
            "application/x-www-form-urlencoded" in content_type
            or "multipart/form-data" in content_type
        ):
            form = await request.form()
            for key in ("username", "password", "password_confirm", "next"):
                if key in form:
                    out[key] = form.get(key)
        else:
            # 兜底：尝试 JSON
            raw = await request.body()
            if raw:
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    if isinstance(payload, dict):
                        out.update(payload)
                except Exception:  # noqa: BLE001
                    pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("_read_body 解析失败（忽略，返回空 dict）: {}", exc)
    return out