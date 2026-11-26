"""运行中任务的自动增量重扫。

手动「重扫」会整份目录比对指纹，但训练还在跑时新的 ``rollout_*.pt`` 是按
步长周期落盘的。这里按南湖作业状态 + 已落盘文件的时间间隔决定下一次扫描，
避免固定高频轮询，也避免等用户手点。

``python main.py`` 会拉起后台循环；测试默认不启动。
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from loguru import logger

# 南湖作业仍在产出 rollout 的状态
ACTIVE_JOB_STATUSES = frozenset(
    {
        "RUNNING",
        "PENDING",
        "STARTING",
        "QUEUED",
        "CREATING",
        "WAITING",
        "INITIALIZING",
        "SCHEDULING",
    }
)
# 作业已结束。若落盘文件仍在按节奏增长，仍视为在跑（本地状态可能滞后）。
TERMINAL_JOB_STATUSES = frozenset(
    {
        "SUCCEEDED",
        "SUCCESS",
        "FAILED",
        "STOPPED",
        "KILLED",
        "CANCELED",
        "CANCELLED",
        "FINISHED",
        "COMPLETED",
    }
)

DEFAULT_CADENCE_SECONDS = 120.0
MIN_CADENCE_SECONDS = 45.0
MAX_CADENCE_SECONDS = 900.0
# 刚写完的 .pt 再等一会儿，避免扫到还在落盘的半截文件
SETTLE_SECONDS = 10.0
# 最新文件年龄小于该倍数的节奏，就认为目录还在出 rollout
RHYTHM_LIVE_FACTOR = 2.5
# 没有南湖状态时，只盯最近扫描过的任务，避免每轮都去扫历史目录
WATCH_AFTER_SCAN_SECONDS = 6 * 3600

_stop = threading.Event()
_thread: Optional[threading.Thread] = None
_thread_guard = threading.Lock()


def cadence_from_mtimes(mtimes: Sequence[float]) -> float:
    """用最近几次 rollout 文件的 mtime 间隔估计下一步要等多久。

    间隔取最近至多 8 个、且不小于 5 秒的差值的中位数，再夹到
    ``[45s, 15min]``。样本不够时用 120 秒。
    """
    ordered = sorted(float(item) for item in mtimes if item and float(item) > 0)
    deltas: List[float] = []
    for prev, nxt in zip(ordered, ordered[1:]):
        gap = nxt - prev
        if gap >= 5.0:
            deltas.append(gap)
    if not deltas:
        return DEFAULT_CADENCE_SECONDS
    recent = deltas[-8:]
    recent.sort()
    mid = recent[len(recent) // 2]
    return max(MIN_CADENCE_SECONDS, min(MAX_CADENCE_SECONDS, mid))


def task_is_producing(job_status: str, file_mtimes: Sequence[float], now: datetime) -> bool:
    """任务是否还在往目录里落 rollout。

    南湖状态为运行中时直接算在跑。状态已结束但最新文件仍落在节奏窗口内，
    也算在跑（作业状态缓存可能还没刷新）。没有作业状态时只看文件节奏。
    """
    status = (job_status or "").strip().upper()
    newest = max((float(item) for item in file_mtimes if item), default=0.0)
    cadence = cadence_from_mtimes(file_mtimes)
    rhythm_live = False
    if newest > 0:
        rhythm_live = (now.timestamp() - newest) <= cadence * RHYTHM_LIVE_FACTOR
    if status in ACTIVE_JOB_STATUSES:
        return True
    if status in TERMINAL_JOB_STATUSES:
        return rhythm_live
    return rhythm_live


def decide_autoscan(
    *,
    job_status: str,
    task_status: str,
    file_mtimes: Sequence[float],
    last_scanned_at: Optional[datetime],
    now: datetime,
) -> Dict[str, Any]:
    """决定这一轮要不要触发增量重扫。

    返回 ``due / reason / cadence_seconds / producing``。
    """
    cadence = cadence_from_mtimes(file_mtimes)
    producing = task_is_producing(job_status, file_mtimes, now)
    base = {
        "due": False,
        "reason": "idle",
        "cadence_seconds": cadence,
        "producing": producing,
    }
    if (task_status or "").strip().upper() == "SCANNING":
        base["reason"] = "scanning"
        return base

    newest = max((float(item) for item in file_mtimes if item), default=0.0)
    newest_dt = datetime.fromtimestamp(newest) if newest > 0 else None
    file_is_new = newest_dt is not None and (
        last_scanned_at is None
        or newest_dt > last_scanned_at + timedelta(milliseconds=500)
    )

    if not producing:
        if last_scanned_at is None and file_mtimes:
            base["due"] = True
            base["reason"] = "never_scanned"
            return base
        if file_is_new:
            base["due"] = True
            base["reason"] = "catch_up_new_file"
            return base
        base["reason"] = "not_running"
        return base

    if last_scanned_at is None:
        base["due"] = True
        base["reason"] = "never_scanned"
        return base
    if file_is_new:
        age = now.timestamp() - newest
        if age < SETTLE_SECONDS:
            base["reason"] = "file_settling"
            return base
        base["due"] = True
        base["reason"] = "new_rollout"
        return base

    anchor = last_scanned_at
    if newest_dt is not None and newest_dt > anchor:
        anchor = newest_dt
    elapsed = (now - anchor).total_seconds()
    if elapsed >= cadence:
        base["due"] = True
        base["reason"] = "cadence"
        return base
    base["reason"] = "wait_cadence"
    return base


def worth_watching(item: Dict[str, Any], now: datetime) -> bool:
    """要不要对这个任务做目录 stat。已结束且长时间没扫过的任务跳过。"""
    status = (item.get("job_status") or "").strip().upper()
    if status in ACTIVE_JOB_STATUSES:
        return True
    task_status = (item.get("task_status") or "").strip().upper()
    if task_status in {"PENDING", "SCANNING"}:
        return True
    last = item.get("last_scanned_at")
    if isinstance(last, datetime):
        return (now - last).total_seconds() <= WATCH_AFTER_SCAN_SECONDS
    created = item.get("create_time")
    if isinstance(created, datetime):
        return (now - created).total_seconds() <= WATCH_AFTER_SCAN_SECONDS
    return False


def _file_mtimes(paths: Sequence[Path]) -> List[float]:
    mtimes: List[float] = []
    for path in paths:
        try:
            mtimes.append(float(path.stat().st_mtime))
        except OSError:
            continue
    return mtimes


def _default_candidates() -> List[Dict[str, Any]]:
    from db.db_task_service import DBTaskService

    return DBTaskService.list_autoscan_candidates()


def _default_resolve_paths(item: Dict[str, Any]) -> List[Path]:
    from service.rollout_scan_service import RolloutScanService

    svc = RolloutScanService()
    return svc._resolve_dump_paths(
        item.get("task_name") or "",
        item.get("rollout_root") or "",
        recursive=bool(item.get("recursive")),
    )


def _default_scan_one(task_id: int) -> Dict[str, Any]:
    from service.rollout_scan_service import RolloutScanService

    return RolloutScanService().rescan_task(task_id, force=False)


def run_autoscan_tick(
    now: Optional[datetime] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    resolve_paths: Optional[Callable[[Dict[str, Any]], Sequence[Path]]] = None,
    scan_one: Optional[Callable[[int], Any]] = None,
) -> List[Dict[str, Any]]:
    """看一轮任务。最多真正扫一个，避免和训练抢磁盘。

    ``candidates`` / ``resolve_paths`` / ``scan_one`` 可注入，便于单测。
    返回本轮触发的动作列表（含未触发的 producing 任务时只含 due 的那条）。
    """
    moment = now or datetime.now()
    if candidates is None:
        try:
            candidates = _default_candidates()
        except Exception as exc:  # noqa: BLE001
            logger.exception("[auto-scan] 读取任务失败: {}", exc)
            return []
    resolve = resolve_paths or _default_resolve_paths
    scan = scan_one or _default_scan_one
    actions: List[Dict[str, Any]] = []
    for item in candidates:
        if not worth_watching(item, moment):
            continue
        task_id = int(item["task_id"])
        try:
            paths = list(resolve(item) or [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("[auto-scan] 解析路径失败 task_id={} err={}", task_id, exc)
            continue
        mtimes = _file_mtimes(paths)
        decision = decide_autoscan(
            job_status=str(item.get("job_status") or ""),
            task_status=str(item.get("task_status") or ""),
            file_mtimes=mtimes,
            last_scanned_at=item.get("last_scanned_at"),
            now=moment,
        )
        if not decision["due"]:
            continue
        logger.info(
            "[auto-scan] task_id={} reason={} cadence={:.0f}s files={}",
            task_id,
            decision["reason"],
            decision["cadence_seconds"],
            len(paths),
        )
        try:
            outcome = scan(task_id) or {}
        except Exception as exc:  # noqa: BLE001
            logger.exception("[auto-scan] 重扫失败 task_id={}: {}", task_id, exc)
            actions.append({"task_id": task_id, "error": str(exc), **decision})
            break
        busy = bool(outcome.get("busy")) if isinstance(outcome, dict) else False
        actions.append({"task_id": task_id, "busy": busy, "outcome": outcome, **decision})
        if busy:
            continue
        break
    return actions


def _loop(interval: float) -> None:
    # 给进程留几秒把路由和库准备好，再开始第一次 stat
    if _stop.wait(5):
        return
    while not _stop.is_set():
        try:
            run_autoscan_tick()
        except Exception as exc:  # noqa: BLE001
            logger.exception("[auto-scan] tick 失败: {}", exc)
        if _stop.wait(interval):
            return


def start_auto_scan_loop(interval: Optional[float] = None) -> None:
    """启动守护线程。重复调用不会再开一条。"""
    global _thread
    if interval is None:
        from utils.config_loader import config

        interval = float(config.get_int("rollout_auto_scan_interval_seconds", 15))
    with _thread_guard:
        if _thread is not None and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(
            target=_loop,
            args=(float(interval),),
            name="rollout-auto-scan",
            daemon=True,
        )
        _thread.start()
        logger.info("[auto-scan] 已启动，轮询间隔 {}s", interval)


def stop_auto_scan_loop() -> None:
    """停掉守护线程。主要给测试用。"""
    global _thread
    _stop.set()
    with _thread_guard:
        thread = _thread
        _thread = None
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=2)


def auto_scan_enabled() -> bool:
    """仅当进程显式打开时才在应用启动时拉起循环。"""
    return os.environ.get("ROLLOUT_INSIGHT_AUTO_SCAN", "") == "1"
