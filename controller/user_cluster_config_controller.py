#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight v0.5.1 per-user cluster 配置 CRUD 控制器。

路由（路由级别不写 ``require_login``，由每个 handler 显式声明）：
- GET  /api/me/cluster-configs                     列表（JSON + HTML 双模式）
- GET  /api/me/cluster-configs/new                 渲染新建表单
- GET  /api/me/cluster-configs/{cluster_key}/edit  渲染编辑表单（带预填）
- POST /api/me/cluster-configs                     upsert（form 或 JSON body）
- POST /api/me/cluster-configs/{cluster_key}/delete 删除

字段顺序与 Shell-dsv2/automate/db/db_cluster_config.py 保持一致，便于后续
v0.5.3+ 直接替换默认 ``cluster_config`` 来源。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from loguru import logger
from starlette.templating import Jinja2Templates

from data.response import Response
from data.rollout_insight_exception import RolloutInsightException
from utils.auth_session import current_user, require_login
from utils.config_loader import config

# router 级别不写 require_login —— handler 显式声明更灵活（GET 列表也强制要求登录）
router = APIRouter()

templates = Jinja2Templates(directory=str(config.templates_dir))


# ---------------------------------------------------------------------------
# 延迟导入辅助
# ---------------------------------------------------------------------------


def _safe_import_attr(module_name: str, attr: str):
    try:
        import importlib

        mod = importlib.import_module(module_name)
        return getattr(mod, attr, None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("延迟导入 {}.{} 失败: {}", module_name, attr, exc)
        return None


def _cluster_service():
    return _safe_import_attr(
        "db.db_user_cluster_config_service", "DBUserClusterConfigService"
    )


# cluster_key 校验：^[a-z][a-z0-9._-]*$，最大 32 字符（DB 列宽）
_CLUSTER_KEY_RE = re.compile(r"^[a-z][a-z0-9._-]*$")
_CLUSTER_KEY_MAX_LEN = 32


def _valid_cluster_key(key: str) -> bool:
    if not key or len(key) > _CLUSTER_KEY_MAX_LEN:
        return False
    return bool(_CLUSTER_KEY_RE.match(key))


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------


def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    """ORM → 模板/JSON dict。"""
    if cfg is None:
        return {}
    return {
        "primary_id": getattr(cfg, "primary_id", None),
        "user_id": getattr(cfg, "user_id", None),
        "cluster_key": getattr(cfg, "cluster_key", None),
        "display_label": getattr(cfg, "display_label", "") or "",
        "api_base_url": getattr(cfg, "api_base_url", "") or "",
        "user_id_field": getattr(cfg, "user_id_field", "") or "",
        "app_key": getattr(cfg, "app_key", "") or "",
        "app_secret": getattr(cfg, "app_secret", "") or "",
        "auth_token_url": getattr(cfg, "auth_token_url", "") or "",
        "token_http_method": getattr(cfg, "token_http_method", "POST") or "POST",
        "token_request_body": getattr(cfg, "token_request_body", "") or "",
        "token_enabled": bool(getattr(cfg, "token_enabled", False)),
        "create_time": getattr(cfg, "create_time", None),
        "update_time": getattr(cfg, "update_time", None),
    }


def _format_dt(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _wants_html(request: Request) -> bool:
    accept = (request.headers.get("accept") or "").lower()
    if "application/json" in accept:
        return False
    if "text/html" in accept or "text/*" in accept:
        return True
    return False


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


@router.get("/api/me/cluster-configs", include_in_schema=False)
async def page_cluster_configs(
    request: Request,
    _user=Depends(require_login),
):
    """渲染列表页（HTML）或返回 JSON 列表。"""
    svc = _cluster_service()
    if svc is None:
        return Response.error(code=50002, msg="DBUserClusterConfigService 不可用")

    rows = svc.list_by_user(int(_user.primary_id))
    items = [_cfg_to_dict(r) for r in rows]

    if _wants_html(request):
        return templates.TemplateResponse(
            request,
            "user_cluster_config.html",
            {
                "page_title": "集群配置 - Rollout Insight",
                "mode": "list",
                "configs": items,
                "editing": None,
                "error": None,
                "success": None,
            },
        )
    return Response.success(
        msg="ok",
        data={"configs": items, "count": len(items)},
    )


@router.get("/api/me/cluster-configs/new", include_in_schema=False)
async def page_cluster_config_new(
    request: Request,
    _user=Depends(require_login),
):
    """渲染新建表单（cluster_key 为空 → 新建模式）。"""
    empty = {
        "primary_id": None,
        "cluster_key": "",
        "display_label": "",
        "api_base_url": "",
        "user_id_field": "",
        "app_key": "",
        "app_secret": "",
        "auth_token_url": "",
        "token_http_method": "POST",
        "token_request_body": "",
        "token_enabled": False,
    }
    return templates.TemplateResponse(
        request,
        "user_cluster_config.html",
        {
            "page_title": "新建集群配置 - Rollout Insight",
            "mode": "new",
            "configs": [],
            "editing": empty,
            "error": None,
            "success": None,
        },
    )


@router.get(
    "/api/me/cluster-configs/{cluster_key}/edit",
    include_in_schema=False,
)
async def page_cluster_config_edit(
    request: Request,
    cluster_key: str,
    _user=Depends(require_login),
):
    """渲染编辑表单（带预填数据）。"""
    svc = _cluster_service()
    if svc is None:
        return templates.TemplateResponse(
            request,
            "user_cluster_config.html",
            {
                "page_title": "集群配置 - Rollout Insight",
                "mode": "list",
                "configs": [],
                "editing": None,
                "error": "DBUserClusterConfigService 不可用",
                "success": None,
            },
            status_code=500,
        )
    row = svc.get_by_user_and_key(int(_user.primary_id), cluster_key)
    if row is None:
        return templates.TemplateResponse(
            request,
            "user_cluster_config.html",
            {
                "page_title": "集群配置 - Rollout Insight",
                "mode": "list",
                "configs": [],
                "editing": None,
                "error": f"未找到集群配置 cluster_key={cluster_key}",
                "success": None,
            },
            status_code=404,
        )
    return templates.TemplateResponse(
        request,
        "user_cluster_config.html",
        {
            "page_title": f"编辑 {cluster_key} - Rollout Insight",
            "mode": "edit",
            "configs": [],
            "editing": _cfg_to_dict(row),
            "error": None,
            "success": None,
        },
    )


@router.post("/api/me/cluster-configs")
async def api_cluster_config_upsert(
    request: Request,
    cluster_key: str = Form(default=""),
    display_label: str = Form(default=""),
    api_base_url: str = Form(default=""),
    user_id_field: str = Form(default=""),
    app_key: str = Form(default=""),
    app_secret: str = Form(default=""),
    auth_token_url: str = Form(default=""),
    token_http_method: str = Form(default="POST"),
    token_request_body: str = Form(default=""),
    token_enabled: str = Form(default="0"),
    _user=Depends(require_login),
):
    """upsert：新建 / 更新都走这里。"""
    svc = _cluster_service()
    if svc is None:
        return Response.error(code=50002, msg="DBUserClusterConfigService 不可用")

    # 注：先按用户提交的原始 cluster_key 校验（不让 `BADKEY → badkey` 蒙混过关），
    # 校验通过后再归一为小写
    raw_key = (cluster_key or "").strip()
    if not _valid_cluster_key(raw_key):
        if _wants_html(request):
            return _render_error_with_form(
                request,
                msg=f"cluster_key 不合法（须小写字母开头，仅含字母数字._-，最大 {_CLUSTER_KEY_MAX_LEN} 字符）",
                editing=_form_to_dict(
                    cluster_key, display_label, api_base_url, user_id_field,
                    app_key, app_secret, auth_token_url, token_http_method,
                    token_request_body, token_enabled,
                ),
            )
        return Response.error(code=50001, msg="cluster_key 不合法")

    key = raw_key.lower()

    enabled_flag = token_enabled in ("1", "true", "True", "on", "yes")
    try:
        row = svc.upsert(
            user_id=int(_user.primary_id),
            cluster_key=key,
            display_label=display_label,
            api_base_url=api_base_url,
            user_id_field=user_id_field,
            app_key=app_key,
            app_secret=app_secret,
            auth_token_url=auth_token_url,
            token_http_method=token_http_method,
            token_request_body=token_request_body,
            token_enabled=enabled_flag,
        )
    except ValueError as exc:
        return Response.error(code=50001, msg=str(exc))

    if row is None:
        if _wants_html(request):
            return _render_error_with_form(
                request,
                msg="保存失败（校验未通过：display_label / api_base_url 在新建时必填）",
                editing=_form_to_dict(
                    cluster_key, display_label, api_base_url, user_id_field,
                    app_key, app_secret, auth_token_url, token_http_method,
                    token_request_body, token_enabled,
                ),
            )
        return Response.error(code=50003, msg="保存失败")

    if _wants_html(request):
        return RedirectResponse(
            url="/api/me/cluster-configs", status_code=302
        )
    return Response.success(
        msg="ok",
        data={"config": _cfg_to_dict(row)},
    )


@router.post("/api/me/cluster-configs/{cluster_key}/delete")
async def api_cluster_config_delete(
    request: Request,
    cluster_key: str,
    _user=Depends(require_login),
):
    """删除（仅自己的）。"""
    svc = _cluster_service()
    if svc is None:
        return Response.error(code=50002, msg="DBUserClusterConfigService 不可用")
    ok = svc.delete_by_user_and_key(int(_user.primary_id), cluster_key)
    if not ok:
        if _wants_html(request):
            return RedirectResponse(
                url="/api/me/cluster-configs", status_code=302
            )
        return Response.error(code=404, msg="未找到该配置")
    if _wants_html(request):
        return RedirectResponse(
            url="/api/me/cluster-configs", status_code=302
        )
    return Response.success(msg="已删除", data={"cluster_key": cluster_key})


# ---------------------------------------------------------------------------
# HTML 错误回显辅助
# ---------------------------------------------------------------------------


def _form_to_dict(
    cluster_key, display_label, api_base_url, user_id_field,
    app_key, app_secret, auth_token_url, token_http_method,
    token_request_body, token_enabled,
) -> Dict[str, Any]:
    return {
        "primary_id": None,
        "cluster_key": cluster_key or "",
        "display_label": display_label or "",
        "api_base_url": api_base_url or "",
        "user_id_field": user_id_field or "",
        "app_key": app_key or "",
        "app_secret": app_secret or "",
        "auth_token_url": auth_token_url or "",
        "token_http_method": (token_http_method or "POST").upper(),
        "token_request_body": token_request_body or "",
        "token_enabled": token_enabled in ("1", "true", "True", "on", "yes"),
    }


def _render_error_with_form(
    request: Request,
    msg: str,
    editing: Dict[str, Any],
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "user_cluster_config.html",
        {
            "page_title": "集群配置 - Rollout Insight",
            "mode": "edit" if editing.get("primary_id") else "new",
            "configs": [],
            "editing": editing,
            "error": msg,
            "success": None,
        },
        status_code=200,
    )