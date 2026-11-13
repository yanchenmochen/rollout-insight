#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""读取 run 目录下 ``attempt_artifacts/``：summary 索引、patch、OpenHands 轨迹。

目录约定（Slime / OpenHands agentic rollout）::

    {run}/attempt_artifacts/
        summaries/{instance_id}__{attempt_id}.json
        patches/{instance_id}__{attempt_id}.patch
        failed_trajectories/{instance_id}__{attempt_id}.history.jsonl
        passed_trajectories/{instance_id}__{attempt_id}.history.jsonl   # 可选

``summary.group_index`` / ``summary.rollout_id`` 与 ``ri_attempt_table`` 的
``group_index`` / ``attempt_id`` 同值（abort 样本 ``rollout_id`` 为空）。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

from utils.failure_insight import parse_pytest_failure

__all__ = [
    "ARTIFACTS_DIRNAME",
    "locate_artifacts_dir",
    "summarize_entry",
    "load_summary_index",
    "load_summary",
    "load_patch",
    "find_trajectory_path",
    "load_trajectory_events",
    "strip_ansi",
    "clear_artifact_cache_for_tests",
]

ARTIFACTS_DIRNAME = "attempt_artifacts"
_SUMMARIES = "summaries"
_PATCHES = "patches"
_TRAJ_DIRS = ("failed_trajectories", "passed_trajectories")
_TRAJ_SUFFIX = ".history.jsonl"
_SIDECAR_NAME = ".ri_artifacts_index.json"
_SIDECAR_VERSION = 1
_TEXT_MAX = 4000
_PATCH_MAX = 200_000

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}


def clear_artifact_cache_for_tests() -> None:
    """仅供单测清空进程内索引缓存。"""
    _INDEX_CACHE.clear()


# ---------------------------------------------------------------------------
# 目录定位
# ---------------------------------------------------------------------------

def _candidate_run_dirs(rollout_root: str) -> List[Path]:
    path = Path(rollout_root).expanduser()
    out: List[Path] = []
    cur = path
    for _ in range(4):
        out.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    return out


def locate_artifacts_dir(rollout_root: Optional[str]) -> Optional[Path]:
    """从 ``task.rollout_root``（通常是 ``{run}/generated/rollout_dumps``）向上找 ``attempt_artifacts``。"""
    text = (rollout_root or "").strip()
    if not text:
        return None
    for cand in _candidate_run_dirs(text):
        target = cand / ARTIFACTS_DIRNAME
        try:
            if (target / _SUMMARIES).is_dir():
                return target
        except OSError:
            continue
    return None


def _safe_stem(stem: str) -> str:
    if not stem or "/" in stem or "\\" in stem or stem.startswith("."):
        raise ValueError(f"非法 artifact 名: {stem!r}")
    return stem


# ---------------------------------------------------------------------------
# summary 索引
# ---------------------------------------------------------------------------

def _as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def summarize_entry(summary: Dict[str, Any], stem: str) -> Dict[str, Any]:
    """把单条 summary JSON 压成索引行（列表 / 聚类所需字段）。"""
    evaluation = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else {}
    phase_times = summary.get("phase_times") if isinstance(summary.get("phase_times"), dict) else {}
    failed_test, error_summary = parse_pytest_failure(str(evaluation.get("stdout_tail") or ""))
    return {
        "stem": stem,
        "attempt_key": str(summary.get("attempt_id") or ""),
        "instance_id": str(summary.get("instance_id") or ""),
        "group_index": _as_int(summary.get("group_index")),
        "rollout_id": _as_int(summary.get("rollout_id")),
        "sample_index": _as_int(summary.get("sample_index")),
        "status": str(summary.get("status") or ""),
        "reward": summary.get("reward"),
        "failed": bool(summary.get("failed")),
        "diagnosis": summary.get("diagnosis"),
        "abort_reason": summary.get("abort_reason"),
        "agent_outcome": summary.get("agent_outcome"),
        "agent_exit_code": summary.get("agent_exit_code"),
        "agent_turns": summary.get("agent_turns"),
        "agent_tool_calls": summary.get("agent_tool_calls"),
        "agent_tool_errors": summary.get("agent_tool_errors"),
        "peak_context_tokens": summary.get("attempt_peak_context_tokens"),
        "diff_line_count": summary.get("diff_line_count"),
        "eval_exit_code": evaluation.get("exit_code"),
        "eval_timed_out": evaluation.get("timed_out"),
        "failed_test": failed_test,
        "error_summary": error_summary,
        "attempt_total_s": phase_times.get("phase_time/attempt_total"),
        "recorded_at": summary.get("recorded_at"),
    }


def _summaries_fingerprint(summaries_dir: Path) -> Optional[Tuple[float, int]]:
    try:
        st = summaries_dir.stat()
        count = sum(1 for _ in os.scandir(summaries_dir))
    except OSError:
        return None
    return float(st.st_mtime), int(count)


def _trajectory_stems(artifacts_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in _TRAJ_DIRS:
        d = artifacts_dir / name
        if not d.is_dir():
            continue
        try:
            for entry in os.scandir(d):
                if entry.name.endswith(_TRAJ_SUFFIX):
                    out[entry.name[: -len(_TRAJ_SUFFIX)]] = name
        except OSError as exc:
            logger.warning("扫描轨迹目录失败 dir={} err={}", d, exc)
    return out


def _scan_summaries(artifacts_dir: Path) -> List[Dict[str, Any]]:
    summaries_dir = artifacts_dir / _SUMMARIES
    traj = _trajectory_stems(artifacts_dir)
    entries: List[Dict[str, Any]] = []
    for entry in os.scandir(summaries_dir):
        if not entry.name.endswith(".json"):
            continue
        stem = entry.name[: -len(".json")]
        try:
            with open(entry.path, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("读取 summary 失败 path={} err={}", entry.path, exc)
            continue
        if not isinstance(summary, dict):
            continue
        row = summarize_entry(summary, stem)
        row["trajectory_kind"] = traj.get(stem)
        entries.append(row)
    entries.sort(key=lambda r: (r.get("sample_index") is None, r.get("sample_index") or 0, r["stem"]))
    return entries


def _read_sidecar(artifacts_dir: Path, fp: Tuple[float, int]) -> Optional[List[Dict[str, Any]]]:
    path = artifacts_dir / _SIDECAR_NAME
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != _SIDECAR_VERSION:
        return None
    if payload.get("mtime") != fp[0] or payload.get("count") != fp[1]:
        return None
    entries = payload.get("entries")
    return entries if isinstance(entries, list) else None


def _write_sidecar(artifacts_dir: Path, fp: Tuple[float, int], entries: List[Dict[str, Any]]) -> None:
    path = artifacts_dir / _SIDECAR_NAME
    tmp = path.with_name(path.name + ".tmp")
    payload = {"version": _SIDECAR_VERSION, "mtime": fp[0], "count": fp[1], "entries": entries}
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)
    except OSError as exc:
        logger.info("artifacts 索引 sidecar 不可写（只读目录时正常） dir={} err={}", artifacts_dir, exc)
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def load_summary_index(artifacts_dir: Path, *, use_cache: bool = True) -> List[Dict[str, Any]]:
    """返回 summaries 的索引行列表；按 summaries 目录 mtime + 文件数失效。

    命中顺序：进程内缓存 → ``attempt_artifacts/.ri_artifacts_index.json`` → 全量扫描。
    """
    summaries_dir = artifacts_dir / _SUMMARIES
    fp = _summaries_fingerprint(summaries_dir)
    if fp is None:
        return []
    key = str(artifacts_dir)
    if use_cache:
        cached = _INDEX_CACHE.get(key)
        if cached is not None and cached.get("fp") == fp:
            return cached["entries"]
        side = _read_sidecar(artifacts_dir, fp)
        if side is not None:
            _INDEX_CACHE[key] = {"fp": fp, "entries": side}
            return side
    entries = _scan_summaries(artifacts_dir)
    _INDEX_CACHE[key] = {"fp": fp, "entries": entries}
    _write_sidecar(artifacts_dir, fp, entries)
    logger.info("artifacts 索引完成 dir={} n={}", artifacts_dir, len(entries))
    return entries


# ---------------------------------------------------------------------------
# 单条读取：summary / patch / 轨迹
# ---------------------------------------------------------------------------

def load_summary(artifacts_dir: Path, stem: str) -> Optional[Dict[str, Any]]:
    """读取单条完整 summary JSON；不存在返回 None。"""
    path = artifacts_dir / _SUMMARIES / f"{_safe_stem(stem)}.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None


def load_patch(artifacts_dir: Path, stem: str, max_chars: int = _PATCH_MAX) -> Dict[str, Any]:
    """读取 unified diff；超长截断并标记 ``truncated``。"""
    path = artifacts_dir / _PATCHES / f"{_safe_stem(stem)}.patch"
    if not path.is_file():
        return {"exists": False, "text": "", "truncated": False, "n_chars": 0}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read(max_chars + 1)
    truncated = len(text) > max_chars
    return {
        "exists": True,
        "text": text[:max_chars],
        "truncated": truncated,
        "n_chars": len(text[:max_chars]),
    }


def find_trajectory_path(artifacts_dir: Path, stem: str) -> Optional[Path]:
    """在 failed_ / passed_trajectories 中查找同名 history.jsonl。"""
    safe = _safe_stem(stem)
    for name in _TRAJ_DIRS:
        path = artifacts_dir / name / f"{safe}{_TRAJ_SUFFIX}"
        if path.is_file():
            return path
    return None


def strip_ansi(text: Any) -> str:
    """去掉终端颜色控制符（pytest / bash 输出常带）。"""
    return _ANSI_RE.sub("", "" if text is None else str(text))


def _clip(text: Any, limit: int = _TEXT_MAX) -> str:
    s = strip_ansi(text)
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _content_text(blocks: Any) -> str:
    """OpenHands content 块（list[{type,text}] / dict / str）→ 纯文本。"""
    if blocks is None:
        return ""
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, dict):
        if "content" in blocks:
            return _content_text(blocks.get("content"))
        return str(blocks.get("text") or "")
    if isinstance(blocks, list):
        parts = [_content_text(b) for b in blocks]
        return "\n".join(p for p in parts if p)
    return str(blocks)


def _action_brief(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    for key in ("command", "code", "message", "thought"):
        val = action.get(key)
        if isinstance(val, str) and val.strip():
            if key == "command" and action.get("path"):
                return f"{val} {action.get('path')}"
            return val
    return ""


def _normalize_event(index: int, raw: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(raw.get("kind") or raw.get("type") or "")
    ev: Dict[str, Any] = {
        "index": index,
        "kind": kind,
        "source": raw.get("source"),
        "timestamp": raw.get("timestamp"),
        "tool_name": raw.get("tool_name"),
        "title": "",
        "text": "",
        "detail": "",
        "is_error": False,
    }
    if kind == "ActionEvent":
        ev["title"] = _clip(raw.get("summary") or _action_brief(raw.get("action")), 400)
        ev["text"] = _clip(_content_text(raw.get("thought")) or raw.get("reasoning_content"))
        action = raw.get("action") if isinstance(raw.get("action"), dict) else {}
        ev["detail"] = _clip(action.get("command") if raw.get("tool_name") == "execute_bash" else _action_brief(action))
    elif kind == "ObservationEvent":
        obs = raw.get("observation") if isinstance(raw.get("observation"), dict) else {}
        ev["text"] = _clip(_content_text(obs))
        exit_code = obs.get("exit_code")
        ev["is_error"] = bool(obs.get("is_error")) or (isinstance(exit_code, int) and exit_code not in (0, -1))
        ev["title"] = f"exit_code={exit_code}" if exit_code is not None else ""
    elif kind == "MessageEvent":
        msg = raw.get("llm_message") if isinstance(raw.get("llm_message"), dict) else {}
        ev["title"] = str(msg.get("role") or raw.get("source") or "")
        ev["text"] = _clip(_content_text(msg))
    elif kind in ("ConversationErrorEvent", "AgentErrorEvent"):
        ev["is_error"] = True
        ev["title"] = str(raw.get("code") or kind)
        ev["text"] = _clip(raw.get("detail") or raw.get("error"))
    elif kind == "SystemPromptEvent":
        ev["title"] = "system prompt"
        ev["text"] = _clip(_content_text(raw.get("system_prompt")), 800)
    else:
        ev["title"] = str(raw.get("reason") or kind or "event")
        ev["text"] = _clip(json.dumps(raw, ensure_ascii=False, default=str), 800)
    return ev


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def load_trajectory_events(
    path: Path,
    *,
    offset: int = 0,
    limit: int = 200,
    errors_only: bool = False,
) -> Dict[str, Any]:
    """分页读取并规整轨迹事件；同时返回全量统计（总数 / 错误数 / 工具分布 / 末尾错误）。"""
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 1000))
    total = 0
    n_errors = 0
    tool_counts: Dict[str, int] = {}
    last_error: Optional[Dict[str, Any]] = None
    page: List[Dict[str, Any]] = []
    matched = 0
    for i, raw in enumerate(_iter_jsonl(path)):
        total += 1
        ev = _normalize_event(i, raw)
        if ev["kind"] == "ActionEvent" and ev.get("tool_name"):
            name = str(ev["tool_name"])
            tool_counts[name] = tool_counts.get(name, 0) + 1
        if ev["is_error"]:
            n_errors += 1
            last_error = ev
        if errors_only and not ev["is_error"]:
            continue
        if offset <= matched < offset + limit:
            page.append(ev)
        matched += 1
    return {
        "total": total,
        "n_matched": matched,
        "n_errors": n_errors,
        "tool_counts": dict(sorted(tool_counts.items(), key=lambda kv: -kv[1])),
        "last_error": last_error,
        "offset": offset,
        "limit": limit,
        "events": page,
    }
