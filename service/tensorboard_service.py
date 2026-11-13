"""托管本机 TensorBoard 进程：按任务 tbs/ 拼 ``--logdir_spec``，新标签打开。

单例：同一时刻只跑一个 TB；spec 不变且进程存活则复用。
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from loguru import logger

from utils.tb_run_paths import discover_tbs_dir, resolve_run_dir

__all__ = [
    "TensorBoardRun",
    "build_runs_for_tasks",
    "build_logdir_spec",
    "get_tensorboard_port",
    "public_tensorboard_url",
    "start_tensorboard",
    "stop_tensorboard",
    "tensorboard_status",
]

_SPEC_UNSAFE = re.compile(r"[,:\s]+")
_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {
    "proc": None,  # subprocess.Popen | None
    "spec": "",
    "port": 0,
    "runs": [],
    "started_at": None,
}


@dataclass
class TensorBoardRun:
    task_id: int
    name: str
    tbs_dir: str
    run_dir: str = ""


def get_tensorboard_port() -> int:
    env = (os.environ.get("ROLLOUT_INSIGHT_TENSORBOARD_PORT") or "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    try:
        from utils.config_loader import config

        return int(config.get_int("tensorboard_port", 6006))
    except Exception:  # noqa: BLE001
        return 6006


def _sanitize_run_name(raw: str, task_id: int) -> str:
    text = (raw or "").strip() or f"task_{task_id}"
    text = _SPEC_UNSAFE.sub("_", text)
    text = text.strip("._") or f"task_{task_id}"
    # TensorBoard run 名不宜过长
    if len(text) > 80:
        text = text[:80].rstrip("._")
    return text


def build_runs_for_tasks(
    tasks: Sequence[Any],
) -> Tuple[List[TensorBoardRun], List[Dict[str, Any]]]:
    """从任务对象列表解析 tbs/。

    ``tasks`` 元素需有 ``primary_id`` / ``task_name`` / ``rollout_root``
    （DBTask 或兼容 dict）。
    """
    runs: List[TensorBoardRun] = []
    skipped: List[Dict[str, Any]] = []
    used_names: Dict[str, int] = {}

    for task in tasks:
        if isinstance(task, dict):
            task_id = int(task.get("primary_id") or task.get("id") or 0)
            task_name = str(task.get("task_name") or "")
            rollout_root = str(task.get("rollout_root") or "")
        else:
            task_id = int(getattr(task, "primary_id", 0) or 0)
            task_name = str(getattr(task, "task_name", "") or "")
            rollout_root = str(getattr(task, "rollout_root", "") or "")

        if task_id <= 0:
            continue

        run_dir = resolve_run_dir(rollout_root)
        tbs = discover_tbs_dir(run_dir) if run_dir is not None else None
        if tbs is None:
            skipped.append(
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "reason": "no_tb_events"
                    if run_dir is not None
                    else "no_run_dir",
                    "rollout_root": rollout_root,
                }
            )
            continue

        base = _sanitize_run_name(task_name, task_id)
        if base in used_names:
            name = f"{base}_{task_id}"
        else:
            name = base
        used_names[name] = task_id
        runs.append(
            TensorBoardRun(
                task_id=task_id,
                name=name,
                tbs_dir=str(tbs),
                run_dir=str(run_dir) if run_dir else "",
            )
        )
    return runs, skipped


def build_logdir_spec(runs: Sequence[TensorBoardRun]) -> str:
    """拼 TensorBoard ``--logdir_spec`` 参数。"""
    parts = [f"{r.name}:{r.tbs_dir}" for r in runs if r.name and r.tbs_dir]
    return ",".join(parts)


def public_tensorboard_url(request_host: Optional[str], port: int) -> str:
    """用请求 Host 的主机名拼 TB URL（远端访问 RI 时不写死 127.0.0.1）。"""
    host = "127.0.0.1"
    raw = (request_host or "").strip()
    if raw:
        # Host 可能带端口
        if "://" not in raw:
            raw = "http://" + raw
        try:
            parsed = urlparse(raw)
            if parsed.hostname:
                host = parsed.hostname
        except Exception:  # noqa: BLE001
            pass
    return f"http://{host}:{int(port)}/"


def _proc_alive(proc: Optional[subprocess.Popen]) -> bool:
    if proc is None:
        return False
    return proc.poll() is None


def _terminate(proc: Optional[subprocess.Popen], *, timeout: float = 5.0) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        proc.kill()
    except OSError:
        pass


def stop_tensorboard() -> Dict[str, Any]:
    with _LOCK:
        proc = _STATE.get("proc")
        _terminate(proc)
        _STATE["proc"] = None
        _STATE["spec"] = ""
        _STATE["port"] = 0
        _STATE["runs"] = []
        _STATE["started_at"] = None
    return {"running": False}


def tensorboard_status() -> Dict[str, Any]:
    with _LOCK:
        proc = _STATE.get("proc")
        alive = _proc_alive(proc)
        if not alive and proc is not None:
            _STATE["proc"] = None
            _STATE["spec"] = ""
            _STATE["runs"] = []
            _STATE["started_at"] = None
            alive = False
        return {
            "running": alive,
            "port": int(_STATE.get("port") or 0) if alive else 0,
            "spec": _STATE.get("spec") or "" if alive else "",
            "runs": list(_STATE.get("runs") or []) if alive else [],
            "pid": int(proc.pid) if alive and proc is not None else None,
            "started_at": _STATE.get("started_at") if alive else None,
        }


def start_tensorboard(
    runs: Sequence[TensorBoardRun],
    *,
    request_host: Optional[str] = None,
    port: Optional[int] = None,
) -> Dict[str, Any]:
    """启动或复用 TensorBoard；返回 url / port / runs。"""
    if not runs:
        raise ValueError("没有可打开的 tbs/ 目录（任务均无 TensorBoard 事件）")

    spec = build_logdir_spec(runs)
    if not spec:
        raise ValueError("logdir_spec 为空")

    tb_bin = shutil.which("tensorboard")
    if not tb_bin:
        raise RuntimeError(
            "未找到 tensorboard 可执行文件，请先 pip install tensorboard"
        )

    use_port = int(port if port is not None else get_tensorboard_port())
    runs_payload = [
        {
            "task_id": r.task_id,
            "name": r.name,
            "tbs_dir": r.tbs_dir,
            "run_dir": r.run_dir,
        }
        for r in runs
    ]

    with _LOCK:
        proc = _STATE.get("proc")
        if (
            _proc_alive(proc)
            and _STATE.get("spec") == spec
            and int(_STATE.get("port") or 0) == use_port
        ):
            url = public_tensorboard_url(request_host, use_port)
            return {
                "url": url,
                "port": use_port,
                "reused": True,
                "runs": runs_payload,
                "pid": int(proc.pid) if proc is not None else None,
            }

        # spec / port 变化或进程已死：先停再起
        _terminate(proc)
        _STATE["proc"] = None

        cmd = [
            tb_bin,
            "--logdir_spec",
            spec,
            "--host",
            "0.0.0.0",
            "--port",
            str(use_port),
            "--reload_interval",
            "30",
        ]
        logger.info("启动 TensorBoard port={} runs={}", use_port, len(runs_payload))
        try:
            new_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError(f"启动 TensorBoard 失败: {exc}") from exc

        # 短暂等待，捕获立即退出
        time.sleep(0.4)
        if new_proc.poll() is not None:
            raise RuntimeError(
                f"TensorBoard 进程立即退出（code={new_proc.returncode}），"
                f"请检查端口 {use_port} 是否被占用或 tbs 路径是否可读"
            )

        _STATE["proc"] = new_proc
        _STATE["spec"] = spec
        _STATE["port"] = use_port
        _STATE["runs"] = runs_payload
        _STATE["started_at"] = time.time()

        url = public_tensorboard_url(request_host, use_port)
        return {
            "url": url,
            "port": use_port,
            "reused": False,
            "runs": runs_payload,
            "pid": int(new_proc.pid),
        }


def collect_tasks_for_scope(
    scope: str,
    task_id: Optional[int] = None,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """按 scope 取任务并解析 runs。

    返回 ``(runs, skipped)``；scope=task 且任务不存在时抛 TASK_NOT_FOUND。
    """
    from data.rollout_insight_exception import RolloutInsightException
    from db.db_task_service import DBTaskService

    scope_norm = (scope or "all").strip().lower()
    if scope_norm == "task":
        if task_id is None or int(task_id) <= 0:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "task 范围需要 task_id"
            )
        task = DBTaskService.get_by_id(int(task_id))
        if task is None:
            raise RolloutInsightException(
                RolloutInsightException.TASK_NOT_FOUND,
                f"任务 #{task_id} 不存在",
            )
        return build_runs_for_tasks([task])

    # all：尽量拉全量（分页上限放大）
    tasks = DBTaskService.list_tasks(limit=5000, offset=0)
    return build_runs_for_tasks(tasks)
