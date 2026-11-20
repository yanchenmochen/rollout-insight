#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight 站点级静态资源控制器（favicon / PWA manifest）。

路由：
- GET /favicon.ico           返回多尺寸 ICO（16/32/48），浏览器标签页自动挑选
- GET /apple-touch-icon.png  返回 180×180 PNG（iOS "添加到主屏幕"）
- GET /site.webmanifest      返回 PWA manifest JSON

设计：
- **不挂 router 级 require_login**：浏览器在未登录（页面被 302 到登录页）
  时同样会请求 ``/favicon.ico``，若此处要求登录会让图标加载失败、标签页
  显示默认图标。
- **不接收外部路径参数**：只按固定文件名在 ``static/`` 下查找，杜绝目录穿越。
- 图标由 ``scripts/gen_favicon.py`` 从 ``static/branding/ri_icon_source.png``
  生成，本控制器只负责分发。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from utils.config_loader import config

router = APIRouter(tags=["site"])

# 站点元信息：品牌名与主题色。
# 品牌名走 config.site_brand_name（默认「洞见」），与模板标题 / 顶栏 logo 一致。
_THEME_COLOR = "#1f3550"
_BACKGROUND_COLOR = "#e8ebef"


def _static_dir() -> Path:
    """定位 static 目录（优先 config.static_dir，回退到仓库内 static/）。"""
    configured = getattr(config, "static_dir", "") or ""
    if configured:
        candidate = Path(configured)
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parent.parent / "static"


def _file_response(filename: str, media_type: str) -> FileResponse:
    """返回 static/ 下指定文件；缺失时 404。"""
    path = _static_dir() / filename
    if not path.is_file():
        return FileResponse(status_code=404, content=b"not found", media_type="text/plain")
    return FileResponse(path, media_type=media_type)


@router.get("/favicon.ico", include_in_schema=False)
async def get_favicon() -> FileResponse:
    """返回多尺寸 favicon.ico（浏览器标签页 / 书签栏图标）。"""
    return _file_response("favicon.ico", "image/x-icon")


@router.get("/apple-touch-icon.png", include_in_schema=False)
async def get_apple_touch_icon() -> FileResponse:
    """返回 180×180 apple-touch-icon.png（iOS 添加到主屏幕）。"""
    return _file_response("apple-touch-icon.png", "image/png")


@router.get("/site.webmanifest", include_in_schema=False)
async def get_site_manifest() -> JSONResponse:
    """返回 PWA manifest（声明名称、主题色与各尺寸图标）。"""
    manifest = {
        "name": config.site_brand_name,
        "short_name": config.site_brand_name,
        "start_url": "/",
        "display": "standalone",
        "background_color": _BACKGROUND_COLOR,
        "theme_color": _THEME_COLOR,
        "icons": [
            {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png"},
        ],
    }
    return JSONResponse(content=manifest)
