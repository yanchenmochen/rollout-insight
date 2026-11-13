"""从 run 目录解析训练侧 P0 指标（奖励设计 / 漏斗）。

优先读 ``tbs/`` TensorBoard scalars；无 TB 或读取失败时，回退扫描
主训练日志里的 ``perf N:`` / ``data.py:194 - rollout/filtered``。

这些指标不在 ``generated/rollout_dumps/`` 的单条 sample 里，而是按
rollout step 的时序汇总。
"""
from __future__ import annotations

import ast
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

__all__ = [
    "SIDECAR_NAME",
    "P0_TAG_PREFIXES",
    "empty_p0_payload",
    "resolve_run_dir",
    "discover_tbs_dir",
    "discover_main_log",
    "load_p0_metrics",
    "load_p0_from_tbs",
    "load_p0_from_log",
    "warm_p0_metrics",
    "clear_p0_cache_for_tests",
]

SIDECAR_NAME = ".ri_p0_metrics.json"
_SIDECAR_VERSION = 2

# run_dir → {"fingerprint": dict, "payload": dict}
_P0_MEM_CACHE: Dict[str, Dict[str, Any]] = {}
_P0_REFRESHING: set = set()
_P0_LOCK = threading.Lock()

# 只抽取奖励设计相关 tags，避免把 400+ perf 明细全灌进前端
P0_EXACT_TAGS = (
    "passrate/pass@1",
    "passrate/pass@2",
    "passrate/pass@4",
    "passrate/pass@8",
    "rollout/generated/group_reward/mixed_fraction",
    "rollout/generated/group_reward/all_one_fraction",
    "rollout/generated/group_reward/all_zero_fraction",
    "rollout/generated/group_reward/mixed_count",
    "rollout/generated/group_reward/all_one_count",
    "rollout/generated/group_reward/all_zero_count",
    "rollout/generated/group_reward/classified_group_count",
    "rollout/removed_ratio",
    "rollout/funnel/loss_eligible_attempts",
    "rollout/funnel/launched_attempts",
    "rollout/funnel/filter_kept_attempts",
    "rollout/generated/raw_reward",
    "rollout/generated/shaped_reward",
    "rollout/filtered/raw_reward",
    "rollout/filtered/shaped_reward",
    "rollout/population/trainable_attempt_count",
)

P0_TAG_PREFIXES = (
    "passrate/",
    "rollout/generated/group_reward/",
    "rollout/removed/",
    "rollout/funnel/",
    "rollout/generated/raw_reward",
    "rollout/generated/shaped_reward",
    "rollout/filtered/raw_reward",
    "rollout/filtered/shaped_reward",
    "rollout/removed_ratio",
    "rollout/population/trainable_attempt_count",
)

_PERF_RE = re.compile(
    r"rollout\.py:\d+\s+-\s+perf\s+(\d+):\s*(\{.*\})\s*$"
)
_FILTERED_RE = re.compile(
    r"data\.py:\d+\s+-\s+rollout/filtered\s+(\d+):\s*(\{.*\})\s*$"
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def empty_p0_payload(*, source: str = "none", run_dir: str = "") -> Dict[str, Any]:
    return {
        "source": source,
        "run_dir": run_dir or "",
        "tbs_dir": "",
        "log_path": "",
        "n_steps": 0,
        "steps": [],
        "passrate": {},
        "group_reward": {},
        "removed": {"ratio": [], "by_reason": []},
        "funnel": {},
        "reward": {},
        "summary": {},
        "warning": None,
        "stale": False,
        "refresh_pending": False,
        "cache": "none",
    }


def clear_p0_cache_for_tests() -> None:
    """仅供单测清空进程内缓存与后台刷新标记。"""
    with _P0_LOCK:
        _P0_MEM_CACHE.clear()
        _P0_REFRESHING.clear()


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

    # 再向上爬最多 3 层
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
        return (path / "tbs").is_dir() or (path / "logs").is_dir() or (
            path / "manifest.yaml"
        ).is_file()
    except OSError:
        return False


def discover_tbs_dir(run_dir: Path) -> Optional[Path]:
    tbs = run_dir / "tbs"
    try:
        if not tbs.is_dir():
            return None
        # 至少有一个 event 文件才算可用
        for child in tbs.iterdir():
            name = child.name
            if name.startswith("events.out.tfevents") or child.is_dir():
                return tbs
    except OSError:
        return None
    return None


def discover_main_log(run_dir: Path) -> Optional[Path]:
    logs = run_dir / "logs"
    try:
        if not logs.is_dir():
            return None
        candidates = [
            p
            for p in logs.iterdir()
            if p.is_file() and p.suffix == ".log" and p.name != "eval_pins.log"
        ]
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return candidates[0]


def load_p0_metrics(
    rollout_root: Optional[str],
    *,
    use_cache: bool = True,
    allow_stale: bool = True,
) -> Dict[str, Any]:
    """对外入口：按 rollout_root 解析 P0 指标。

    缓存策略（``use_cache=True``）—— **sidecar/内存优先，避免先扫 NAS tbs/**：
    1. 进程内存有 payload → 立刻返回，后台校验指纹
    2. sidecar 有 payload → 立刻返回，后台校验指纹（过期则刷新）
    3. 否则冷读 tbs/日志并写 sidecar
    """
    run_dir = resolve_run_dir(rollout_root)
    if run_dir is None:
        payload = empty_p0_payload(source="none")
        payload["warning"] = "无法从 rollout_root 推断 run 目录"
        return payload

    cache_key = str(run_dir)
    root_text = str(rollout_root or "")

    if use_cache:
        # 1) 内存：不问指纹，先出图（后台再校验）
        with _P0_LOCK:
            mem = _P0_MEM_CACHE.get(cache_key)
        if mem is not None and isinstance(mem.get("payload"), dict):
            out = dict(mem["payload"])
            if int(out.get("n_steps") or 0) > 0:
                out["stale"] = False
                out["refresh_pending"] = True
                out["cache"] = "memory"
                _schedule_verify_and_refresh(
                    root_text, cache_key, mem.get("fingerprint")
                )
                return out

        # 2) sidecar：先读盘再出图，指纹校验放到后台（NAS iterdir 很慢）
        blob = _read_sidecar_blob(run_dir)
        if blob is not None:
            payload = blob.get("payload")
            if isinstance(payload, dict) and int(payload.get("n_steps") or 0) > 0:
                served = dict(payload)
                ver = int(blob.get("version") or 0)
                served_fp = blob.get("fingerprint")
                exact_ver = ver == _SIDECAR_VERSION
                served["stale"] = not exact_ver
                served["refresh_pending"] = True
                served["cache"] = "sidecar" if exact_ver else "sidecar_stale"
                if isinstance(served_fp, dict):
                    _store_mem(cache_key, served_fp, served)
                if allow_stale or exact_ver:
                    _schedule_verify_and_refresh(root_text, cache_key, served_fp)
                    return served

    tbs_dir = discover_tbs_dir(run_dir)
    log_path = discover_main_log(run_dir)
    fingerprint = _fingerprint(run_dir, tbs_dir, log_path)

    payload = _compute_p0_payload(run_dir, tbs_dir, log_path)
    payload["stale"] = False
    payload["refresh_pending"] = False
    payload["cache"] = "computed"
    if payload.get("n_steps", 0) > 0:
        _write_sidecar(run_dir, fingerprint, payload)
        _store_mem(cache_key, fingerprint, payload)
    return payload


def warm_p0_metrics(rollout_root: Optional[str]) -> bool:
    """后台预热：优先走缓存；无缓存时才冷读 tbs。"""
    try:
        payload = load_p0_metrics(
            rollout_root, use_cache=True, allow_stale=True
        )
        return int(payload.get("n_steps") or 0) > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("预热 P0 指标失败 root={} err={}", rollout_root, exc)
        return False


def _store_mem(
    cache_key: str, fingerprint: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    with _P0_LOCK:
        _P0_MEM_CACHE[cache_key] = {
            "fingerprint": dict(fingerprint),
            "payload": dict(payload),
        }


def _schedule_verify_and_refresh(
    rollout_root: str,
    cache_key: str,
    known_fp: Optional[Mapping[str, Any]],
) -> None:
    """后台核对 tbs/日志 fingerprint；一致则标 fresh，否则冷读刷新。"""
    with _P0_LOCK:
        if cache_key in _P0_REFRESHING:
            return
        _P0_REFRESHING.add(cache_key)

    def _job() -> None:
        try:
            run_dir = Path(cache_key)
            tbs_dir = discover_tbs_dir(run_dir)
            log_path = discover_main_log(run_dir)
            fingerprint = _fingerprint(run_dir, tbs_dir, log_path)
            if known_fp is not None and dict(known_fp) == fingerprint:
                with _P0_LOCK:
                    mem = _P0_MEM_CACHE.get(cache_key)
                    if mem and isinstance(mem.get("payload"), dict):
                        mem["fingerprint"] = fingerprint
                        payload = dict(mem["payload"])
                        payload["stale"] = False
                        payload["refresh_pending"] = False
                        payload["cache"] = "memory"
                        mem["payload"] = payload
                return
            payload = _compute_p0_payload(run_dir, tbs_dir, log_path)
            payload["stale"] = False
            payload["refresh_pending"] = False
            payload["cache"] = "computed"
            if int(payload.get("n_steps") or 0) > 0:
                _write_sidecar(run_dir, fingerprint, payload)
                _store_mem(cache_key, fingerprint, payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "后台核对/刷新 P0 失败 key={} err={}", cache_key, exc
            )
        finally:
            with _P0_LOCK:
                _P0_REFRESHING.discard(cache_key)

    threading.Thread(
        target=_job, name=f"p0-verify:{Path(cache_key).name}", daemon=True
    ).start()


def _schedule_background_refresh(rollout_root: str, cache_key: str) -> None:
    """兼容旧调用：等价于忽略 known_fp 强制刷新。"""
    _schedule_verify_and_refresh(rollout_root, cache_key, None)


def _compute_p0_payload(
    run_dir: Path,
    tbs_dir: Optional[Path],
    log_path: Optional[Path],
) -> Dict[str, Any]:
    payload: Dict[str, Any]
    active_tbs = tbs_dir
    if active_tbs is not None:
        try:
            payload = load_p0_from_tbs(active_tbs)
            payload["source"] = "tbs"
            payload["run_dir"] = str(run_dir)
            payload["tbs_dir"] = str(active_tbs)
            payload["log_path"] = str(log_path) if log_path else ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("读 tbs 失败 run_dir={} err={}", run_dir, exc)
            payload = empty_p0_payload(source="none", run_dir=str(run_dir))
            payload["tbs_dir"] = str(active_tbs)
            payload["warning"] = f"tbs 解析失败，尝试日志回退: {exc}"
            active_tbs = None
    else:
        payload = empty_p0_payload(source="none", run_dir=str(run_dir))

    if payload.get("n_steps", 0) <= 0 and log_path is not None:
        try:
            log_payload = load_p0_from_log(log_path)
            log_payload["source"] = "log"
            log_payload["run_dir"] = str(run_dir)
            log_payload["tbs_dir"] = str(active_tbs) if active_tbs else ""
            log_payload["log_path"] = str(log_path)
            if payload.get("warning"):
                log_payload["warning"] = payload["warning"]
            payload = log_payload
        except Exception as exc:  # noqa: BLE001
            logger.warning("读主日志失败 log={} err={}", log_path, exc)
            payload = empty_p0_payload(source="none", run_dir=str(run_dir))
            payload["log_path"] = str(log_path)
            payload["warning"] = f"tbs/日志均未能解析 P0 指标: {exc}"

    if payload.get("n_steps", 0) <= 0 and not payload.get("warning"):
        payload["warning"] = "run 目录下未找到可用的 tbs/ 或 perf 日志指标"

    payload["summary"] = _build_summary(payload)
    return payload


def load_p0_from_tbs(tbs_dir: Path) -> Dict[str, Any]:
    """用 TensorBoard EventAccumulator 读 scalars。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("未安装 tensorboard，无法读 tbs/") from exc

    ea = EventAccumulator(str(tbs_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars") or [])
    wanted = _select_tags(tags)
    series: Dict[str, List[Tuple[int, float]]] = {}
    for tag in wanted:
        events = ea.Scalars(tag)
        series[tag] = [(int(e.step), float(e.value)) for e in events]
    return _series_to_payload(series)


def load_p0_from_log(log_path: Path) -> Dict[str, Any]:
    """流式扫描主日志，抽取 perf / filtered 字典中的 P0 字段。"""
    by_step: Dict[int, Dict[str, float]] = {}
    with open(log_path, "r", errors="replace") as fh:
        for raw in fh:
            if "perf " not in raw and "rollout/filtered " not in raw:
                continue
            line = _ANSI_RE.sub("", raw).rstrip()
            m = _PERF_RE.search(line)
            if m:
                step = int(m.group(1))
                data = _safe_literal_dict(m.group(2))
                if data:
                    bucket = by_step.setdefault(step, {})
                    for k, v in data.items():
                        if _is_p0_tag(str(k)) and isinstance(v, (int, float)):
                            bucket[str(k)] = float(v)
                continue
            m = _FILTERED_RE.search(line)
            if m:
                step = int(m.group(1))
                data = _safe_literal_dict(m.group(2))
                if data:
                    bucket = by_step.setdefault(step, {})
                    for k, v in data.items():
                        if _is_p0_tag(str(k)) and isinstance(v, (int, float)):
                            bucket[str(k)] = float(v)

    series: Dict[str, List[Tuple[int, float]]] = {}
    for step, kv in by_step.items():
        for tag, val in kv.items():
            series.setdefault(tag, []).append((step, val))
    for tag in series:
        series[tag].sort(key=lambda x: x[0])
    return _series_to_payload(series)


def _is_p0_tag(tag: str) -> bool:
    if tag in P0_EXACT_TAGS:
        return True
    if tag.startswith("rollout/removed/count_"):
        return True
    if tag.startswith("rollout/generated/group_reward/pass_count_"):
        return True
    return any(tag.startswith(p) for p in P0_TAG_PREFIXES)


def _select_tags(available: Iterable[str]) -> List[str]:
    return sorted(t for t in available if _is_p0_tag(t))


def _safe_literal_dict(text: str) -> Dict[str, Any]:
    try:
        obj = ast.literal_eval(text)
    except (SyntaxError, ValueError, MemoryError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _series_to_payload(series: Dict[str, List[Tuple[int, float]]]) -> Dict[str, Any]:
    payload = empty_p0_payload()
    if not series:
        return payload

    pass_tags = [t for t in (
        "passrate/pass@1",
        "passrate/pass@2",
        "passrate/pass@4",
        "passrate/pass@8",
    ) if t in series]

    # 统一 steps：以 pass@1 或任意非空 series 为准
    anchor = pass_tags[0] if pass_tags else next(iter(series))
    steps = [s for s, _ in series[anchor]]
    payload["steps"] = steps
    payload["n_steps"] = len(steps)

    def _vals(tag: str) -> List[Optional[float]]:
        lookup = {s: v for s, v in series.get(tag, [])}
        return [lookup.get(s) for s in steps]

    payload["passrate"] = {
        tag.split("/", 1)[-1]: _compact_floats(_vals(tag)) for tag in pass_tags
    }
    payload["group_reward"] = {
        "mixed_fraction": _compact_floats(_vals(
            "rollout/generated/group_reward/mixed_fraction"
        )),
        "all_one_fraction": _compact_floats(_vals(
            "rollout/generated/group_reward/all_one_fraction"
        )),
        "all_zero_fraction": _compact_floats(_vals(
            "rollout/generated/group_reward/all_zero_fraction"
        )),
        "mixed_count": _compact_floats(_vals(
            "rollout/generated/group_reward/mixed_count"
        )),
        "all_one_count": _compact_floats(_vals(
            "rollout/generated/group_reward/all_one_count"
        )),
        "all_zero_count": _compact_floats(_vals(
            "rollout/generated/group_reward/all_zero_count"
        )),
        "pass_count_last": _pass_count_histogram(series, steps[-1] if steps else None),
    }
    payload["funnel"] = {
        "launched_attempts": _compact_floats(_vals(
            "rollout/funnel/launched_attempts"
        )),
        "loss_eligible_attempts": _compact_floats(_vals(
            "rollout/funnel/loss_eligible_attempts"
        )),
        "filter_kept_attempts": _compact_floats(_vals(
            "rollout/funnel/filter_kept_attempts"
        )),
        "trainable_attempt_count": _compact_floats(_vals(
            "rollout/population/trainable_attempt_count"
        )),
        "removed_ratio": _compact_floats(_vals("rollout/removed_ratio")),
    }
    payload["reward"] = {
        "generated_raw": _compact_floats(_vals("rollout/generated/raw_reward")),
        "generated_shaped": _compact_floats(_vals("rollout/generated/shaped_reward")),
        "filtered_raw": _compact_floats(_vals("rollout/filtered/raw_reward")),
        "filtered_shaped": _compact_floats(_vals("rollout/filtered/shaped_reward")),
    }

    removed_tags = sorted(
        t for t in series if t.startswith("rollout/removed/count_")
    )
    by_reason: List[Dict[str, Any]] = []
    for tag in removed_tags:
        vals = [v for _, v in series[tag]]
        reason = tag[len("rollout/removed/count_") :]
        by_reason.append(
            {
                "reason": reason,
                "tag": tag,
                "total": float(sum(vals)),
                "n_steps": len(vals),
                "last": float(vals[-1]) if vals else 0.0,
                "series": _compact_floats(_vals(tag)),
            }
        )
    by_reason.sort(key=lambda x: x["total"], reverse=True)
    payload["removed"] = {
        "ratio": _compact_floats(_vals("rollout/removed_ratio")),
        "by_reason": by_reason,
    }
    return payload


def _pass_count_histogram(
    series: Mapping[str, Sequence[Tuple[int, float]]],
    last_step: Optional[int],
) -> List[Dict[str, Any]]:
    if last_step is None:
        return []
    out: List[Dict[str, Any]] = []
    for k in range(0, 9):
        tag = f"rollout/generated/group_reward/pass_count_{k}_of_8_count"
        lookup = {s: v for s, v in series.get(tag, [])}
        if last_step not in lookup:
            continue
        out.append({"k": k, "label": f"{k}/8", "count": float(lookup[last_step])})
    return out


def _compact_floats(vals: Sequence[Optional[float]]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for v in vals:
        if v is None:
            out.append(None)
        else:
            out.append(round(float(v), 6))
    return out


def _last_num(vals: Sequence[Optional[float]]) -> Optional[float]:
    for v in reversed(vals):
        if v is not None:
            return float(v)
    return None


def _first_num(vals: Sequence[Optional[float]]) -> Optional[float]:
    for v in vals:
        if v is not None:
            return float(v)
    return None


def _build_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    passrate = payload.get("passrate") or {}
    p1 = passrate.get("pass@1") or []
    group = payload.get("group_reward") or {}
    funnel = payload.get("funnel") or {}
    removed = payload.get("removed") or {}
    reward = payload.get("reward") or {}
    by_reason = list(removed.get("by_reason") or [])
    return {
        "pass_at_1_first": _first_num(p1),
        "pass_at_1_last": _last_num(p1),
        "mixed_fraction_last": _last_num(group.get("mixed_fraction") or []),
        "all_one_fraction_last": _last_num(group.get("all_one_fraction") or []),
        "all_zero_fraction_last": _last_num(group.get("all_zero_fraction") or []),
        "removed_ratio_last": _last_num(funnel.get("removed_ratio") or removed.get("ratio") or []),
        "loss_eligible_last": _last_num(funnel.get("loss_eligible_attempts") or []),
        "launched_last": _last_num(funnel.get("launched_attempts") or []),
        "generated_raw_last": _last_num(reward.get("generated_raw") or []),
        "filtered_raw_last": _last_num(reward.get("filtered_raw") or []),
        "top_removed": [
            {"reason": r.get("reason"), "total": r.get("total")}
            for r in by_reason[:5]
        ],
    }


def _fingerprint(
    run_dir: Path,
    tbs_dir: Optional[Path],
    log_path: Optional[Path],
) -> Dict[str, Any]:
    """轻量指纹：只盯 tfevents* / 主日志 mtime，避免 tbs/ 全量 iterdir。"""

    def _file_mtime(p: Optional[Path]) -> float:
        if p is None:
            return 0.0
        try:
            return float(p.stat().st_mtime)
        except OSError:
            return 0.0

    def _tfevents_mtime(directory: Optional[Path]) -> float:
        if directory is None:
            return 0.0
        try:
            if not directory.is_dir():
                return 0.0
        except OSError:
            return 0.0
        best = 0.0
        try:
            for child in directory.iterdir():
                name = child.name
                if not name.startswith("events.out.tfevents"):
                    # 子目录内也可能有 events（少见）；只扫一层文件即可
                    if child.is_dir():
                        try:
                            for nested in child.iterdir():
                                if nested.name.startswith("events.out.tfevents"):
                                    try:
                                        best = max(best, float(nested.stat().st_mtime))
                                    except OSError:
                                        continue
                        except OSError:
                            continue
                    continue
                try:
                    best = max(best, float(child.stat().st_mtime))
                except OSError:
                    continue
        except OSError:
            return best
        return best

    return {
        "run_dir": str(run_dir),
        "tbs_mtime": _tfevents_mtime(tbs_dir),
        "log_mtime": _file_mtime(log_path),
        "tbs_dir": str(tbs_dir) if tbs_dir else "",
        "log_path": str(log_path) if log_path else "",
    }


def _sidecar_path(run_dir: Path) -> Path:
    return run_dir / SIDECAR_NAME


def _read_sidecar_blob(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = _sidecar_path(run_dir)
    try:
        if not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(blob, dict):
        return None
    # v1/v2 均可作 stale 源；精确命中仍要求当前 version
    if int(blob.get("version") or 0) not in {1, 2, _SIDECAR_VERSION}:
        return None
    payload = blob.get("payload")
    if not isinstance(payload, dict):
        return None
    return blob


def _write_sidecar(
    run_dir: Path,
    fingerprint: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    path = _sidecar_path(run_dir)
    # 落盘时去掉瞬时字段，避免污染 fingerprint 比较无关字段
    stored = dict(payload)
    stored.pop("stale", None)
    stored.pop("refresh_pending", None)
    stored.pop("cache", None)
    blob = {
        "version": _SIDECAR_VERSION,
        "fingerprint": dict(fingerprint),
        "payload": stored,
    }
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("写 P0 sidecar 失败 path={} err={}", path, exc)
