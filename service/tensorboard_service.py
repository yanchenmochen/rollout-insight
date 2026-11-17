"""托管本机 TensorBoard 进程：用任务名做 Run 名，新标签打开。

每个任务的 ``tbs/`` 或 ``tf_logs/`` 会挂成符号链接，链接名即库内任务名，
TensorBoard 左侧 Run 列因此可以直接按任务名对照。

单例：同一时刻只跑一个 TB；任务集合不变且进程存活则复用。
配置端口上若已有其他 TensorBoard，启动时会先停掉再绑定。
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
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

_SPEC_UNSAFE = re.compile(r"[,:\s/\\]+")
_SS_PID = re.compile(r"pid=(\d+)")
# 同一 tf_logs/ 里训练重启会再写一个从 step 0 开始的短事件文件。
# 默认 --load_fast（Rust data server）只保留这份短文件，曲线就剩几步
# （例如 63 个 rollout 的 passrate 只显示 step 0–2）。关掉快速加载，
# 并关闭 orphan purge，让 Python 读取全部事件文件。
_TB_LAUNCH_ID = "load_fast=false,purge_orphaned_data=false"
_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {
    "proc": None,  # subprocess.Popen | None
    "spec": "",
    "launch": "",
    "port": 0,
    "runs": [],
    "logdir": "",
    "started_at": None,
}


@dataclass
class TensorBoardRun:
    task_id: int
    name: str
    tbs_dir: str
    run_dir: str = ""
    task_name: str = ""


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
                task_name=task_name,
            )
        )
    return runs, skipped


def build_logdir_spec(runs: Sequence[TensorBoardRun]) -> str:
    """拼运行集合指纹（亦兼容 ``--logdir_spec`` 形式）。

    实际拉起时用符号链接目录，Run 名取 ``name``（即任务名），避免
    TensorBoard 把实验目录别名当成 Run。
    """
    parts = [f"{r.name}:{r.tbs_dir}" for r in runs if r.name and r.tbs_dir]
    return ",".join(parts)


def _tensorboard_logdir_root() -> Path:
    return Path(tempfile.gettempdir()) / "rollout-insight-tb"


def _materialize_task_logdir(runs: Sequence[TensorBoardRun]) -> Path:
    """把每个任务的事件目录挂成「任务名 → tbs/tf_logs」符号链接。"""
    root = _tensorboard_logdir_root()
    root.mkdir(parents=True, exist_ok=True)
    desired = {r.name: r.tbs_dir for r in runs if r.name and r.tbs_dir}

    for child in list(root.iterdir()):
        if child.name in desired:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()

    for name, target in desired.items():
        link = root / name
        if link.is_symlink():
            try:
                current = os.readlink(link)
            except OSError:
                current = ""
            if current == target:
                continue
            link.unlink()
        elif link.exists():
            raise RuntimeError(f"无法用任务名创建 TensorBoard Run 链接，路径已存在: {link}")
        os.symlink(target, link, target_is_directory=True)
    return root


def _pids_from_ss_output(text: str, port: int) -> List[int]:
    pids: List[int] = []
    boundary = re.compile(rf":{int(port)}(?:\s|$)")
    for line in text.splitlines():
        if not boundary.search(line):
            continue
        for match in _SS_PID.finditer(line):
            pids.append(int(match.group(1)))
    return pids


def _listener_pids(port: int) -> List[int]:
    """返回正在监听 ``port`` 的 pid。解析失败时返回空列表。"""
    try:
        out = subprocess.check_output(
            ["ss", "-ltnp"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return _pids_from_ss_output(out, port)


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()


def _reclaim_tensorboard_port(port: int, keep_pids: Set[int]) -> None:
    """配置端口上若是别的 TensorBoard，先停掉，避免打开到旧的 Run 列表。"""
    foreign = [pid for pid in _listener_pids(port) if pid not in keep_pids]
    if not foreign:
        return
    for pid in foreign:
        cmd = _cmdline(pid)
        if "tensorboard" not in cmd.lower():
            raise RuntimeError(
                f"端口 {port} 已被其他程序占用（pid={pid} {cmd[:160]}），无法启动 TensorBoard"
            )
        logger.warning(
            "端口 {} 被其他 TensorBoard 占用，先停止 pid={} cmd={}",
            port,
            pid,
            cmd[:200],
        )
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue

    deadline = time.time() + 3.0
    while time.time() < deadline:
        still = [pid for pid in _listener_pids(port) if pid not in keep_pids]
        if not still:
            return
        time.sleep(0.1)
    for pid in foreign:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(0.2)
    still = [pid for pid in _listener_pids(port) if pid not in keep_pids]
    if still:
        raise RuntimeError(f"无法释放端口 {port}，仍被占用: {still}")


def _stderr_path() -> Path:
    try:
        from utils.config_loader import config

        log_dir = Path(config.log_dir or "")
        if str(log_dir):
            log_dir.mkdir(parents=True, exist_ok=True)
            return log_dir / "tensorboard-proc.log"
    except Exception:  # noqa: BLE001
        pass
    return Path(tempfile.gettempdir()) / "rollout-insight-tensorboard.log"


def _stderr_tail(limit: int = 500) -> str:
    path = _stderr_path()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _ensure_we_own_port(proc: subprocess.Popen, port: int, attempts: int = 20) -> None:
    """进程退出，或端口仍被别的 pid 占用时，视为启动失败。"""
    for _ in range(attempts):
        if proc.poll() is not None:
            tail = _stderr_tail()
            extra = f"；{tail}" if tail else ""
            raise RuntimeError(
                f"TensorBoard 进程已退出（code={proc.returncode}）{extra}"
            )
        holders = _listener_pids(port)
        if proc.pid in holders:
            return
        if holders and proc.pid not in holders:
            time.sleep(0.25)
            continue
        time.sleep(0.25)
    if proc.poll() is not None:
        tail = _stderr_tail()
        extra = f"；{tail}" if tail else ""
        raise RuntimeError(
            f"TensorBoard 进程已退出（code={proc.returncode}）{extra}"
        )
    holders = [pid for pid in _listener_pids(port) if pid != proc.pid]
    if holders:
        raise RuntimeError(
            f"端口 {port} 仍被其他进程占用（pid={holders}），本次 TensorBoard 未绑定成功"
        )


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
        _STATE["launch"] = ""
        _STATE["port"] = 0
        _STATE["runs"] = []
        _STATE["logdir"] = ""
        _STATE["started_at"] = None
    return {"running": False}


def tensorboard_status() -> Dict[str, Any]:
    with _LOCK:
        proc = _STATE.get("proc")
        alive = _proc_alive(proc)
        if not alive and proc is not None:
            _STATE["proc"] = None
            _STATE["spec"] = ""
            _STATE["launch"] = ""
            _STATE["runs"] = []
            _STATE["logdir"] = ""
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
            "task_name": r.task_name or r.name,
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
            and _STATE.get("launch") == _TB_LAUNCH_ID
            and int(_STATE.get("port") or 0) == use_port
        ):
            try:
                logdir = _materialize_task_logdir(runs)
            except OSError as exc:
                raise RuntimeError(f"准备 TensorBoard 任务名目录失败: {exc}") from exc
            _STATE["logdir"] = str(logdir)
            url = public_tensorboard_url(request_host, use_port)
            return {
                "url": url,
                "port": use_port,
                "reused": True,
                "runs": runs_payload,
                "logdir": str(logdir),
                "pid": int(proc.pid) if proc is not None else None,
            }

        # spec / port 变化或进程已死：先停再起。
        # 配置端口上的其他 TensorBoard 一并停掉，否则浏览器会打开旧 Run 列表。
        _terminate(proc)
        _STATE["proc"] = None
        _reclaim_tensorboard_port(use_port, set())

        try:
            logdir = _materialize_task_logdir(runs)
        except OSError as exc:
            raise RuntimeError(f"准备 TensorBoard 任务名目录失败: {exc}") from exc

        cmd = [
            tb_bin,
            "--logdir",
            str(logdir),
            "--host",
            "0.0.0.0",
            "--port",
            str(use_port),
            "--reload_interval",
            "30",
            "--load_fast=false",
            "--purge_orphaned_data=false",
        ]
        logger.info(
            "启动 TensorBoard port={} runs={} logdir={}",
            use_port,
            len(runs_payload),
            logdir,
        )
        stderr_path = _stderr_path()
        try:
            err_file = open(stderr_path, "wb")
        except OSError:
            err_file = None
        try:
            new_proc = subprocess.Popen(
                cmd,
                stdout=err_file if err_file is not None else subprocess.DEVNULL,
                stderr=err_file if err_file is not None else subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError(f"启动 TensorBoard 失败: {exc}") from exc
        finally:
            if err_file is not None:
                err_file.close()

        try:
            _ensure_we_own_port(new_proc, use_port)
        except RuntimeError:
            _terminate(new_proc)
            raise

        _STATE["proc"] = new_proc
        _STATE["spec"] = spec
        _STATE["launch"] = _TB_LAUNCH_ID
        _STATE["port"] = use_port
        _STATE["runs"] = runs_payload
        _STATE["logdir"] = str(logdir)
        _STATE["started_at"] = time.time()

        url = public_tensorboard_url(request_host, use_port)
        return {
            "url": url,
            "port": use_port,
            "reused": False,
            "runs": runs_payload,
            "logdir": str(logdir),
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
