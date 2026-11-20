#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""统一创建 Jinja2Templates 实例（注入全局模板变量）。

背景
----
所有 controller 都各自 ``Jinja2Templates(directory=str(config.templates_dir))``，
导致**全局模板变量无处安放** —— 比如站点品牌名 ``site_brand_name``（默认「洞见」），
若要每个 controller 都往 context 里塞一遍，既重复又容易漏。

本模块提供 :func:`create_templates`，把 ``env.globals`` 一次性配好：

- ``site_brand_name`` —— 站点品牌名。模板里 ``base.html`` 用它渲染浏览器
  标签页标题后缀与顶栏 logo 文字。

模板若直接写死品牌名，改名就要逐个模板改；走这里只需改 ``config.json``
 的 ``site_brand_name``（或 ``utils.config_loader`` 里的默认值）。
"""
from __future__ import annotations

from starlette.templating import Jinja2Templates

from utils.config_loader import config

__all__ = ["create_templates"]


def create_templates() -> Jinja2Templates:
    """创建带全局模板变量的 Jinja2Templates 实例。

    Returns:
        可直接用于 ``templates.TemplateResponse(request, name, context)`` 的对象。
    """
    templates = Jinja2Templates(directory=str(config.templates_dir))
    templates.env.globals["site_brand_name"] = config.site_brand_name
    return templates
