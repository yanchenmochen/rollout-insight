"""TensorBoard 单 tag 深链（不改启动命令和端口）。

对照本机已安装的 tensorboard 2.20.0（``webfiles.zip`` 里的 ``index.js``）：

- 活跃页面是 Angular ``<tb-webapp>``，默认路由 path 为 ``/``。
- ``useHash`` 初始为 false，``setUseHash`` 没有调用点。
  ``#timeseries`` 以及 ``#timeseries&tagFilter=`` 不会被当前页面当成 tag 过滤。
- 指标 DeepLinkProvider 从 ``window.location.search`` 读取查询参数
  ``tagFilter``（正则），写入 ``metrics.tagFilter``，用来过滤 scalar 卡片。

因此 2.20.x 可以定位到单个 tag，方式是 ``?tagFilter=^转义后的 tag$``，不是 hash。
未核对过的版本不追加该参数，只打开该任务的 TensorBoard 根地址。
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

__all__ = [
    "CHECKED_TENSORBOARD_VERSION",
    "TAG_FILTER_QUERY_KEY",
    "append_tag_filter",
    "installed_tensorboard_version",
    "tag_filter_supported",
    "tensorboard_tag_filter",
]

# 已打开 webfiles 核对过深链行为的版本。
CHECKED_TENSORBOARD_VERSION = "2.20.0"
TAG_FILTER_QUERY_KEY = "tagFilter"


def installed_tensorboard_version() -> str:
    """已安装的 tensorboard 版本；导入失败时返回空串。"""
    try:
        import tensorboard
    except Exception:  # noqa: BLE001
        return ""
    return str(getattr(tensorboard, "__version__", "") or "")


def tag_filter_supported(version: Optional[str] = None) -> bool:
    """该版本是否能用 ``?tagFilter=`` 定位到单个 scalar tag。"""
    ver = installed_tensorboard_version() if version is None else str(version or "")
    return ver == CHECKED_TENSORBOARD_VERSION or ver.startswith("2.20.")


def tensorboard_tag_filter(tag: str) -> str:
    """精确匹配一个 tag 的正则（``tagFilter`` 按正则解释）。"""
    return "^" + re.escape(str(tag or "").strip()) + "$"


def append_tag_filter(base_url: str, tag: str) -> str:
    """在 TensorBoard 根 URL 上追加 ``tagFilter``，并丢掉 hash。"""
    text = (base_url or "").strip()
    if not text:
        return ""
    parts = urlsplit(text)
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key != TAG_FILTER_QUERY_KEY
    ]
    query.append((TAG_FILTER_QUERY_KEY, tensorboard_tag_filter(tag)))
    path = parts.path or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, urlencode(query), ""))
