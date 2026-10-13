"""本地 rollout-insight Web API 客户端。

- 默认 base URL: http://localhost:39998
- 通过 ``--api-base`` / ``ROLLOUT_INSIGHT_API_BASE`` 覆盖
- 零外部依赖：仅用标准库 ``urllib``
- 复用本地 controller 的返回结构（``Response`` = ``{code, msg, data}``）

错误约定：
- 网络/超时/非 200 → 抛 ``ApiError``；CLI 转 ``exit(1)``
- 业务层 ``code != 0`` → 同样抛 ``ApiError``（携带 code/msg/data）
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping, Optional


DEFAULT_API_BASE = "http://localhost:39998"


class ApiError(RuntimeError):
    """Web API 错误（含业务 code）。"""

    def __init__(self, message: str, code: int = -1, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def resolve_api_base(explicit: Optional[str] = None) -> str:
    """解析 API base URL；优先级：

    1. 显式参数 ``explicit``
    2. 环境变量 ``ROLLOUT_INSIGHT_API_BASE``
    3. 默认 ``http://localhost:39998``
    """
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("ROLLOUT_INSIGHT_API_BASE")
    if env:
        return env.rstrip("/")
    return DEFAULT_API_BASE


def _request(
    method: str,
    url: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    form: Optional[Mapping[str, Any]] = None,
    json_body: Optional[Mapping[str, Any]] = None,
    timeout: float = 30.0,
) -> dict:
    """底层 HTTP 请求；返回解析后的 dict。"""
    if params:
        # urllib.parse.quote for url fragment; keys/values may be str/int
        qs = urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None},
            doseq=False,
        )
        url = f"{url}?{qs}"

    body: Optional[bytes] = None
    headers = {"Accept": "application/json"}
    if form is not None:
        body = urllib.parse.urlencode(form).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif json_body is not None:
        body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, method=method)

    for k, v in headers.items():
        req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ApiError(
                    f"无法解析响应（{method} {url}）: {exc}; body 开头={raw[:120]!r}",
                    code=-1,
                ) from exc
    except urllib.error.HTTPError as exc:
        # 业务错误（5xx / 4xx）；读 body 仍可能给出结构化 Response
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "code" in payload:
            raise ApiError(
                payload.get("msg") or f"HTTP {exc.code}",
                code=payload.get("code", exc.code),
                data=payload.get("data"),
            ) from exc
        raise ApiError(
            f"HTTP {exc.code} {exc.reason}: {raw[:120] if raw else ''}",
            code=exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        raise ApiError(
            f"无法连接 {url}（{exc.reason}）", code=-1
        ) from exc
    except TimeoutError as exc:
        raise ApiError(f"请求 {url} 超时: {exc}", code=-1) from exc


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def list_tasks(
    api_base: str,
    *,
    task_name: Optional[str] = None,
    status: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> dict:
    """GET /api/tasks；返回完整 Response dict。"""
    url = f"{api_base}/api/tasks"
    return _request(
        "GET",
        url,
        params={
            "task_name": task_name,
            "status": status,
            "page": page,
            "page_size": page_size,
            "limit": limit,
            "offset": offset,
        },
    )


def create_task(
    api_base: str,
    *,
    task_name: str,
    rollout_root: str,
    description: str = "",
    tokenizer_dir: str = "",
    nanhu_job_id: str = "",
    source_type: str = "local_dump",
    recursive: bool = False,
) -> dict:
    """POST /api/tasks（form-encoded）。"""
    url = f"{api_base}/api/tasks"
    return _request(
        "POST",
        url,
        form={
            "task_name": task_name,
            "rollout_root": rollout_root,
            "description": description,
            "tokenizer_dir": tokenizer_dir,
            "nanhu_job_id": nanhu_job_id,
            "source_type": source_type,
            "recursive": "1" if recursive else "0",
        },
    )


def rescan_task(api_base: str, task_id: int) -> dict:
    """POST /api/tasks/{id}/rescan。"""
    url = f"{api_base}/api/tasks/{task_id}/rescan"
    return _request("POST", url)


def delete_task(api_base: str, task_id: int) -> dict:
    """POST /api/tasks/{id}/delete。"""
    url = f"{api_base}/api/tasks/{task_id}/delete"
    return _request("POST", url)


def get_task(api_base: str, task_id: int) -> dict:
    """GET /api/tasks/{id}（JSON-friendly 优先：尝试加 .json 后缀失败时退回 HTML）。

    controller 当前没有 /api/tasks/{id}.json，所以这里直接 ``_request``，
    失败时返回 ``{"code": -1, "msg": "..."}``。
    """
    url = f"{api_base}/api/tasks/{task_id}.json"
    try:
        return _request("GET", url)
    except ApiError as exc:
        # 404 时 controller 没有 .json 端点；提示 user
        raise ApiError(
            f"无法以 JSON 形式获取任务 #{task_id}（controller 当前仅 HTML 渲染）：{exc}",
            code=exc.code,
            data=exc.data,
        ) from exc


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def search_segments(
    api_base: str,
    *,
    task_id: Optional[int] = None,
    min_reward: Optional[float] = None,
    keyword: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> dict:
    """GET /api/segments/search。"""
    url = f"{api_base}/api/segments/search"
    return _request(
        "GET",
        url,
        params={
            "task_id": task_id,
            "min_reward": min_reward,
            "keyword": keyword,
            "page": page,
            "page_size": page_size,
        },
    )


def show_segment(api_base: str, segment_id: int) -> dict:
    """GET /api/segments/{id}/transcript.json（直接拿 JSON transcript）。"""
    url = f"{api_base}/api/segments/{segment_id}/transcript.json"
    return _request("GET", url)


def list_segments(
    api_base: str,
    *,
    task_id: int,
    min_reward: Optional[float] = None,
    keyword: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> dict:
    """segments list —— 复 search_segments（controller 阶段 2C 仅提供 search 路径）。"""
    return search_segments(
        api_base,
        task_id=task_id,
        min_reward=min_reward,
        keyword=keyword,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def healthz(api_base: str) -> dict:
    """GET /healthz。"""
    return _request("GET", f"{api_base}/healthz")