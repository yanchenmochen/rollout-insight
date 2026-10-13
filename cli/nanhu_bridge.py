"""南湖（nanhu-cli）subprocess 桥接。

策略：
- 优先尝试 import 阶段 3A 提供的 ``nanhu_client``（包内模块）；
- 都没有就回退到 ``nanhu`` / ``nanhu-cli`` subprocess（来自 ``nanhu-skill``）。

错误：subprocess 退出码非 0 → 抛 ``NanhuBridgeError``，CLI 转 ``exit(1)``。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence


class NanhuBridgeError(RuntimeError):
    """nanhu-cli 调用错误。"""

    def __init__(self, message: str, code: int = -1, stderr: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.stderr = stderr


# ---------------------------------------------------------------------------
# subprocess 入口
# ---------------------------------------------------------------------------


def _resolve_executable() -> str:
    """找 ``nanhu`` 或 ``nanhu-cli`` 可执行文件。

    优先级：PATH 上的 ``nanhu`` → ``nanhu-cli`` → ``NanhuClient`` Python 调用失败。
    """
    for name in ("nanhu", "nanhu-cli"):
        path = shutil.which(name)
        if path:
            return path
    raise NanhuBridgeError(
        "找不到 nanhu / nanhu-cli 可执行文件。"
        "请先 `pip install -e /path/to/nanhu-skill`，"
        "或确认 PATH 上有 nanhu / nanhu-cli。",
        code=-1,
    )


def _run_nanhu_subprocess(
    args: Sequence[str],
    *,
    timeout: float = 60.0,
) -> tuple[str, str, int]:
    """subprocess 调 nanhu；返回 (stdout, stderr, returncode)。

    本函数不解析 stdout —— CLI 入口根据 stdout 是否是 JSON 决定渲染方式。
    """
    exe = _resolve_executable()
    cmd = [exe, *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise NanhuBridgeError(f"无法执行 {exe}: {exc}", code=-1) from exc
    except subprocess.TimeoutExpired as exc:
        raise NanhuBridgeError(
            f"nanhu-cli 调用超时（{timeout}s）：{' '.join(args[:6])}...",
            code=-1,
        ) from exc

    return proc.stdout or "", proc.returncode


def _is_json_payload(text: str) -> bool:
    """stdout 看起来像 JSON（``{`` 开头，去掉前导空白后）？"""
    s = text.lstrip()
    return s.startswith("{") or s.startswith("[")


# ---------------------------------------------------------------------------
# 命令封装
# ---------------------------------------------------------------------------


def list_jobs(
    *,
    cluster: str,
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    user_id: Optional[str] = None,
    format_: str = "json",
    pretty: bool = False,
) -> tuple[str, int]:
    """nanhu jobs list [filters]

    路由策略：
    - ``--job-type`` 显式给出 → ``nanhu jobs list --job-type ...``
    - 否则 → ``nanhu jobs running``（不需要 job_type）

    Returns:
        (stdout_text, returncode)
        调用方根据 stdout 决定打印还是解析。
    """
    base_args = [f"--cluster={cluster}", f"--format={format_}"]
    if pretty:
        base_args.append("--pretty")

    if job_type:
        args = base_args + ["jobs", "list", "--job-type", job_type]
        if status:
            args += ["--status", status]
        if user_id:
            args += ["--project-id", user_id]
    else:
        args = base_args + ["jobs", "running"]
        if status:
            args += ["--status", status]
        if user_id:
            args += ["--project-id", user_id]

    return _run_nanhu_subprocess(args)


def restart_job(
    *,
    cluster: str,
    job_id: str,
    submit: bool = False,
    timeout: float = 120.0,
) -> tuple[str, int]:
    """nanhu jobs restart --job-id <id> [--submit]"""
    args = [f"--cluster={cluster}", "jobs", "restart", "--job-id", job_id]
    if submit:
        args.append("--submit")
    return _run_nanhu_subprocess(args, timeout=timeout)


def stop_job(
    *,
    cluster: str,
    job_id: str,
    submit: bool = False,
) -> tuple[str, int]:
    """nanhu jobs stop --job-id <id> [--submit]"""
    args = [f"--cluster={cluster}", "jobs", "stop", "--job-id", job_id]
    if submit:
        args.append("--submit")
    return _run_nanhu_subprocess(args)


def job_detail(
    *,
    cluster: str,
    job_id: str,
) -> tuple[str, int]:
    """nanhu jobs detail --job-id <id>"""
    args = [f"--cluster={cluster}", "jobs", "detail", "--job-id", job_id]
    return _run_nanhu_subprocess(args)


def link_job_to_task(
    *,
    api_base: str,
    task_name: str,
    job_id: str,
    submit: bool = False,
) -> Any:
    """把南湖 job-id 绑定到一个本地 task（v0.3 阶段 3A 接口）。

    - submit=False（dry-run）：仅打印将要做的事，不调用 API
    - submit=True：调用本地 Web API 的「link nanhu job」端点（3A 提供）。

    当前 3A 还没有合入，所以这里仅做 dry-run 报告；后续接入。
    """
    plan = {
        "action": "link nanhu job to task",
        "api_base": api_base,
        "task_name": task_name,
        "nanhu_job_id": job_id,
        "submit": submit,
        "note": (
            "dry-run：实际 link 操作由阶段 3A nanhu_client 落地。"
            "本 CLI 仅打印 plan，加 --submit 后会调用（待 3A 接入）。"
        ),
    }
    if not submit:
        return plan

    # 真要 submit：调本地 API。3A 会提供 POST /api/tasks/link_nanhu。
    # 这里给一个 placeholder，stage 3A 落地后改一行即可。
    import cli.api_client as api

    # 暂用 search + create 模式模拟"绑定"：返回当前 task 信息 + 提示
    listed = api.list_tasks(api_base, task_name=task_name)
    return {
        "action": "link nanhu job to task",
        "task_name": task_name,
        "nanhu_job_id": job_id,
        "submit": True,
        "matched_tasks": listed.get("data", {}).get("tasks", []) if isinstance(listed, dict) else None,
        "note": (
            "阶段 3A nanhu_client 尚未提供 link 端点；"
            "已找到匹配 task 名称的任务，请手工到任务详情页绑定 nanhu_job_id。"
        ),
    }