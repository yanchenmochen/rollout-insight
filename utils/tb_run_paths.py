"""从 task.rollout_root 解析训练 run 目录与 TensorBoard 事件目录。

兼容常见目录名：``tbs/``、``tf_logs/``（优先 tbs）。
供 TensorBoard 启停服务复用。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "TB_EVENT_DIR_NAMES",
    "resolve_run_dir",
    "discover_tbs_dir",
    "discover_tb_event_dir",
]

# 按优先级探测
TB_EVENT_DIR_NAMES: Tuple[str, ...] = ("tbs", "tf_logs")


def resolve_run_dir(rollout_root: Optional[str]) -> Optional[Path]:
    """从 ``task.rollout_root``（通常指向 ``.../generated/rollout_dumps``）推 run 根目录。"""
    text = (rollout_root or "").strip()
    if not text:
        return None
    try:
        path = Path(text).expanduser()
    except (TypeError, ValueError):
        return None
    try:
        if path.is_file():
            path = path.parent
    except OSError:
        pass

    candidates: List[Path] = []
    if path.name == "rollout_dumps":
        parent = path.parent
        if parent.name == "generated":
            candidates.append(parent.parent)
        candidates.append(parent)
    candidates.append(path)

    seen = set()
    ordered: List[Path] = []
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cand)

    for cand in ordered:
        if _looks_like_run_dir(cand):
            return cand

    cur = path
    for _ in range(3):
        cur = cur.parent
        if str(cur) in seen or cur == cur.parent:
            break
        if _looks_like_run_dir(cur):
            return cur
    return ordered[0] if ordered else None


def _looks_like_run_dir(path: Path) -> bool:
    try:
        if (path / "logs").is_dir() or (path / "manifest.yaml").is_file():
            return True
        for name in TB_EVENT_DIR_NAMES:
            if (path / name).is_dir():
                return True
        return False
    except OSError:
        return False


def _dir_has_tb_events(directory: Path) -> bool:
    """目录内有 tfevents 文件，或含可能嵌套 events 的子目录。"""
    try:
        if not directory.is_dir():
            return False
        for child in directory.iterdir():
            name = child.name
            if name.startswith("events.out.tfevents"):
                return True
            if "tfevents" in name and child.is_file():
                return True
            if child.is_dir():
                return True
    except OSError:
        return False
    return False


def discover_tb_event_dir(run_dir: Path) -> Optional[Path]:
    """在 run 根下按 ``TB_EVENT_DIR_NAMES`` 找可用的 TensorBoard 事件目录。"""
    for name in TB_EVENT_DIR_NAMES:
        candidate = run_dir / name
        if _dir_has_tb_events(candidate):
            return candidate
    return None


def discover_tbs_dir(run_dir: Path) -> Optional[Path]:
    """兼容旧名：等同 ``discover_tb_event_dir``。"""
    return discover_tb_event_dir(run_dir)
