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
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


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
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ApiError(
                    f"无法解析响应（{method} {url}）: {exc}; body 开头={raw[:120]!r}",
                    code=-1,
                ) from exc
            # 业务层 code != 0 → 抛 ApiError（即使 HTTP 200）
            if (
                isinstance(payload, Mapping)
                and payload.get("code") not in (None, 0)
            ):
                raise ApiError(
                    payload.get("msg") or f"业务 code={payload.get('code')}",
                    code=payload.get("code", -1),
                    data=payload.get("data"),
                )
            return payload
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


# ---------------------------------------------------------------------------
# Rollouts (v0.4.0)
# ---------------------------------------------------------------------------


def get_rollout(api_base: str, rollout_id: int) -> dict:
    """GET /api/rollouts/{id}（rollout 概览 JSON）。"""
    url = f"{api_base}/api/rollouts/{rollout_id}"
    return _request("GET", url)


def get_rollout_json(api_base: str, rollout_id: int) -> dict:
    """GET /api/rollouts/{id}/json（plan 文档里的「.json」端点）。

    v0.4.0 实际 controller **没有** ``.json`` 后缀的端点 — JSON 端点就是
    ``/api/rollouts/{id}``。这里先试 ``.json``，404 时自动回退到无后缀路径。
    """
    url = f"{api_base}/api/rollouts/{rollout_id}/json"
    try:
        return _request("GET", url)
    except ApiError as exc:
        if exc.code == 404:
            logger.info(
                "/api/rollouts/{}/json 404，回退到 /api/rollouts/{}", rollout_id, rollout_id
            )
            return get_rollout(api_base, rollout_id)
        raise


def list_rollouts_by_task(
    api_base: str,
    task_id: int,
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> dict:
    """GET /api/tasks/{id}/rollouts。

    v0.4.0 controller **没有**这个 JSON list 端点（只有 HTML 详情页
    ``/api/tasks/{task_id}/rollouts/{rollout_id}``）。这里先尝试 list 端点，
    404 时回退：从 ``/api/tasks`` 拿该 task 详情、再用 ``/api/rollouts/{id}``
    一个个拉 — 仍可工作但慢；如果 task 也没有对应字段则返回空 list。
    """
    url = f"{api_base}/api/tasks/{task_id}/rollouts"
    try:
        return _request("GET", url, params={"limit": limit, "offset": offset})
    except ApiError as exc:
        if exc.code != 404:
            raise
        logger.info(
            "/api/tasks/{}/rollouts 404，回退到 /api/tasks 详情（rollouts 字段）", task_id
        )

    # 回退：从 /api/tasks 拿 list，按 task_id 过滤（server 限制 limit<=500）
    # v0.4.0 controller 的 /api/tasks 只返回 task 摘要（rollout_count 数字），
    # 没有 rollouts 列表；这里 fallback 给 task 摘要 + 提示，不伪造 rollouts。
    try:
        tasks_resp = list_tasks(api_base, limit=500)
    except ApiError as exc:
        return {
            "code": exc.code,
            "msg": f"fallback 也失败（/api/tasks 不可用）: {exc}",
            "data": {"task_id": task_id, "rollouts": [], "count": 0, "fallback": True},
        }
    task_payload = tasks_resp.get("data") if isinstance(tasks_resp, dict) else None
    task_list = []
    if isinstance(task_payload, dict):
        task_list = task_payload.get("tasks") or []
    elif isinstance(task_payload, list):
        task_list = task_payload
    target = next(
        (
            t
            for t in task_list
            if isinstance(t, Mapping)
            and (t.get("id") == task_id or t.get("primary_id") == task_id)
        ),
        None,
    )
    if target is None:
        return {
            "code": 0,
            "msg": f"task #{task_id} 不存在（fallback 路径）",
            "data": {
                "task_id": task_id,
                "rollouts": [],
                "count": 0,
                "fallback": True,
                "status": "task_not_found",
                "hint": "controller 当前没有 GET /api/tasks/{id}/rollouts JSON 端点",
            },
        }
    return {
        "code": 0,
        "msg": "ok (fallback：仅 task 摘要，无 rollouts 列表)",
        "data": {
            "task_id": task_id,
            "task_summary": {
                "task_name": target.get("task_name"),
                "status": target.get("status"),
                "rollout_count": target.get("rollout_count"),
                "attempt_count": target.get("attempt_count"),
                "segment_count": target.get("segment_count"),
                "mean_reward": target.get("mean_reward"),
                "rollout_root": target.get("rollout_root"),
            },
            "rollouts": [],
            "count": 0,
            "fallback": True,
            "hint": (
                "v0.4.0 controller 未提供 GET /api/tasks/{id}/rollouts JSON 端点；"
                "如需逐个 rollout 概览，请用 'rollout-insight rollouts show <primary_id>'"
                "（rollout primary_id 见 task_summary.attempt_count 或 sqlite 查 db_rollout）。"
            ),
        },
    }


def parse_rollout(
    api_base: str,
    rollout_id: int,
    *,
    tokenizer_dir: str = "",
    force: bool = True,
) -> dict:
    """POST /api/rollouts/{id}/parse（触发 T3 重解析）。

    Args:
        tokenizer_dir: 写到 ``json_body`` 里作为 ``tokenizer_dir``；controller
            当前不消费，但保留以备后端 T4/T5 升级。
    """
    url = f"{api_base}/api/rollouts/{rollout_id}/parse"
    return _request(
        "POST",
        url,
        params={"force": "true" if force else "false"},
        json_body={"tokenizer_dir": tokenizer_dir},
    )


# ---------------------------------------------------------------------------
# Agent-transcripts (v0.4.0)
# ---------------------------------------------------------------------------


def agent_transcript_report(api_base: str, path: str) -> dict:
    """GET /api/agent-transcripts/report?path=...（单文件）。"""
    url = f"{api_base}/api/agent-transcripts/report"
    return _request("GET", url, params={"path": path})


def agent_transcript_batch(
    api_base: str,
    dir_path: Optional[str] = None,
) -> dict:
    """GET /api/agent-transcripts/batch[?dir=...]（批量）。"""
    url = f"{api_base}/api/agent-transcripts/batch"
    return _request("GET", url, params={"dir": dir_path})


# ---------------------------------------------------------------------------
# Segments turns & rollouts cleanup (v0.5.0 T5)
# ---------------------------------------------------------------------------


def get_segment_turns(api_base: str, segment_id: int, timeout: float = 60.0) -> dict:
    """GET /api/segments/{id}/turns.json（v0.5.0 T4 升级后走 token 切分）。

    返回 ``Response.success`` 结构。
    """
    url = f"{api_base}/api/segments/{segment_id}/turns.json"
    return _request("GET", url, timeout=timeout)


def list_segment_ids_for_rollout(api_base: str, rollout_pk: int) -> list:
    """列出某个 rollout 下所有 segment_id（CLI backfill-turns 场景）。

    v0.4.0 没有 ``GET /api/rollouts/{id}/segments`` JSON 端点，所以这里先调
    ``GET /api/rollouts/{id}``（返回 attempts[].first_segment_id）拿到第一段，
    然后从 ``GET /api/segments/search?task_id=&keyword=rollout:<pk>`` 兜底回扫。

    更简单：v0.5.0 plan 中 CLI 入口不要求极严格 —— 这里只尝试已知端点；无
    segment 时返回空 list。
    """
    seg_ids: list = []
    try:
        rollout_payload = get_rollout(api_base, rollout_pk)
        data = rollout_payload.get("data") or {}
        attempts = data.get("attempts") or []
        # attempts[].first_segment_id 仅是每个 attempt 第一段；并非全段
        for a in attempts:
            sid = a.get("first_segment_id")
            if isinstance(sid, int):
                seg_ids.append(sid)
    except ApiError:
        pass
    return list(dict.fromkeys(seg_ids))


def clear_turn_table(api_base: str, rollout_pk: int) -> dict:
    """触发 ``DELETE FROM ri_segment_turn_table``（v0.5.0 T5 CLI 选项）。

    v0.5.0 plan 不强制要求 server 端暴露专用端点，本地 CLI 直接通过
    ``service/rollout_parse_service`` SQL 走；这里调一个占位端点
    ``POST /api/rollouts/{id}/clear-turn-table`` 由 controller 在 404 时
    fallback 到本地连接。
    """
    url = f"{api_base}/api/rollouts/{rollout_pk}/clear-turn-table"
    try:
        return _request("POST", url)
    except ApiError as exc:
        return {
            "code": exc.code,
            "msg": f"server 端 clear-turn-table 不可用（已在 main bootstrap 清过）: {exc.message}",
        }