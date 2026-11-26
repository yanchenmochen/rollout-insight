#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""界面图标清单：把「菜单项 → 图标 → 一句话说明」收敛到一处。

图标本体是 Lucide 的 SVG（MIT License），已 vendor 到 ``static/icons/``：
部署环境为离线内网，CDN 不可达，因此不走外链。

每个图标都带一句话说明（``hint``）：顶栏 / 任务内跳转栏这类入口悬停时
展示，让用户一眼看出每个菜单项管什么。

用法（模板）::

    {% include "_nav_icons.html" ignore missing %}   {# 或直接内联 #}
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

__all__ = [
    "ICON_DIR",
    "NAV_ICONS",
    "icon_path",
    "nav_icon",
    "icon_svg",
]

# vendor 的图标目录（相对仓库根）
ICON_DIR = "static/icons"


# 菜单项 → 图标说明。
# key 用「路由前缀 + 语义」，避免不同页面出现同名图标语义冲突。
NAV_ICONS: Dict[str, Dict[str, str]] = {
    "tasks": {
        "icon": "list-checks",
        "hint": "所有训练任务的列表：状态、rollout 数、attempt 数",
    },
    "tensorboard": {
        "icon": "chart-line",
        "hint": "多任务 TensorBoard 曲线对比：把多个 run 的指标叠在一起看",
    },
    "cross-task-compare": {
        "icon": "git-compare",
        "hint": "跨任务对比：单任务看题目随训练变化，多任务比共同题目变好还是变坏",
    },
    "insights": {
        "icon": "lightbulb",
        "hint": "洞察：自动生成的训练结论与观察",
    },
    "nanhu": {
        "icon": "repeat",
        "hint": "南湖作业：按关键字/算法库检索并批量关联创建任务",
    },
    "settings": {
        "icon": "settings",
        "hint": "系统设置：每人维护自己的南湖 app_key / app_secret",
    },
    "login": {
        "icon": "log-in",
        "hint": "登录已有账号",
    },
    "register": {
        "icon": "user-round",
        "hint": "注册新账号（首个用户开放）",
    },
    "logout": {
        "icon": "log-out",
        "hint": "退出当前登录",
    },
    "cluster-configs": {
        "icon": "settings",
        "hint": "个人设置：维护自己的南湖集群凭据",
    },
    # 任务内跳转栏
    "task-detail": {
        "icon": "list-checks",
        "hint": "任务详情：概览与全量通过率",
    },
    "task-curves": {
        "icon": "chart-spline",
        "hint": "训练曲线：按指标按钮组看训练过程",
    },
    "task-failures": {
        "icon": "triangle-alert",
        "hint": "失败分析：每个 attempt 怎么失败的",
    },
    "task-problems": {
        "icon": "scan-search",
        "hint": "问题分析：逐题看通过率与变好变坏",
    },
    "task-reward": {
        "icon": "gauge",
        "hint": "奖励诊断：奖励为什么不涨",
    },
}


def icon_path(name: str) -> str:
    """返回图标在 static 下的访问路径。"""
    return f"/static/icons/{str(name or '').strip()}.svg"


def nav_icon(key: str) -> Dict[str, str]:
    """按 key 取图标信息；未知返回空 dict。"""
    return dict(NAV_ICONS.get(str(key or "").strip(), {}))


def icon_svg(name: str, size: int = 16) -> str:
    """读出图标 SVG 文本，内联到 HTML（可随 ``currentColor`` 变色）。"""
    path = Path(__file__).resolve().parent.parent / ICON_DIR / f"{str(name or '').strip()}.svg"
    try:
        svg = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if size and size > 0:
        svg = svg.replace(
            "<svg\n",
            f'<svg width="{size}" height="{size}"\n',
            1,
        )
    return svg.strip()
