#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight v0.4.0 —— .pt dump 解析 utils。

复用 inspect-rollout-trajectory 的 ``_attempt_layout`` 规则：
attempt 键 = ``(group_index, rollout_id)``；每个 sample 是该 attempt 的一个
segment，按 ``metadata.segment_index`` 排序。

约定：
- ``load_pt`` 抑制 MUSA / CUDA 的 ``CreatePlatform failed`` 噪音（见 view_trajectory.py）
- ``parse_attempts_and_segments`` 返回结构化的 attempts + segments + 整文响应
- ``split_response_to_turns`` 按 ``<|im_start|>`` 切分并识别 role
- ``token_estimate`` 默认按字符 / 3.5 粗估
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# 在 import torch 前先屏蔽 MUSA / CUDA 探测输出
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import torch

from loguru import logger

from data.rollout_insight_exception import RolloutInsightException


__all__ = [
    "load_pt",
    "parse_attempts_and_segments",
    "split_response_to_turns",
    "token_estimate",
    "_attempt_layout",  # 暴露给测试
]


# ---------------------------------------------------------------------------
# MUSA / CUDA 抑制
# ---------------------------------------------------------------------------


def _suppress_platform_noise():
    """把 stdout/stderr 重定向到 StringIO，包裹 torch / MUSA 探测噪音。

    view_trajectory.py 用 ``with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    import torch`` 来避免平台探测日志泄漏；本函数把这一行做成可复用的 context manager。
    """
    return contextlib.ExitStack()


# ---------------------------------------------------------------------------
# .pt 加载
# ---------------------------------------------------------------------------


def load_pt(path: str) -> Dict[str, Any]:
    """用 ``torch.load`` 加载 .pt dump，抑制平台探测噪音。

    Args:
        path: .pt 文件绝对路径

    Returns:
        dump dict，形如 ``{"rollout_id": int, "samples": [...]}``

    Raises:
        RolloutInsightException: 路径不存在 / 加载失败时
    """
    if not path:
        raise RolloutInsightException(
            RolloutInsightException.INVALID_ARGUMENT, "load_pt: path 不能为空"
        )
    if not os.path.isfile(path):
        raise RolloutInsightException(
            RolloutInsightException.PARSE_DUMP_NOT_FOUND,
            f".pt dump 文件不存在: {path}",
        )
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            dump = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        logger.exception("torch.load 失败 path={} err={}", path, exc)
        raise RolloutInsightException(
            RolloutInsightException.PARSE_DUMP_CORRUPTED,
            f"加载 .pt dump 失败 {path}: {exc}",
        ) from exc

    if not isinstance(dump, dict):
        raise RolloutInsightException(
            RolloutInsightException.PARSE_DUMP_CORRUPTED,
            f".pt dump 不是 dict 类型: {type(dump).__name__}",
        )
    return dump


# ---------------------------------------------------------------------------
# attempt / segment 布局
# ---------------------------------------------------------------------------


def _attempt_layout(samples: List[Dict[str, Any]]) -> Tuple[
    List[Tuple[Optional[int], Any]], Dict[Tuple[Optional[int], Any], List[int]]
]:
    """复用 inspect-rollout-trajectory 的 attempt 键定义。

    Returns:
        row_keys: 与 samples 等长，每行一个 ``(group_index, rollout_id)``
        attempt_rows: attempt key → 该 attempt 下所有 sample 行号列表
    """
    attempt_rows: Dict[Tuple[Optional[int], Any], List[int]] = {}
    row_keys: List[Tuple[Optional[int], Any]] = []
    for i, sample in enumerate(samples):
        attempt_id = sample.get("rollout_id")
        if attempt_id is None:
            attempt_id = sample.get("index", f"row-{i}")
        key = (sample.get("group_index"), attempt_id)
        row_keys.append(key)
        attempt_rows.setdefault(key, []).append(i)
    return row_keys, attempt_rows


def _status_text(status: Any) -> str:
    return str(status).split(".")[-1].lower()


def _segment_payload(
    sample: Dict[str, Any],
    attempt_key: Tuple[Optional[int], Any],
    siblings: List[int],
) -> Dict[str, Any]:
    """把单个 sample 转成 segment dict，按 ``metadata.segment_index`` 排序友好。"""
    metadata = sample.get("metadata") or {}
    segment_index = metadata.get("segment_index")
    segment_count = metadata.get("segment_count")
    if not isinstance(segment_index, int) or not isinstance(segment_count, int):
        # 旧 .pt 没有 segment_index，按 siblings 中的顺序估算
        try:
            segment_index = siblings.index(samples_index_for(sample))
        except ValueError:
            segment_index = 0
        segment_count = len(siblings)
    response_text = sample.get("response") or sample.get("response_text") or ""
    prompt_text = sample.get("prompt") or ""
    prompt_summary = prompt_text[:200] if isinstance(prompt_text, str) else ""
    response_length = sample.get("response_length")
    if not isinstance(response_length, int):
        response_length = len(response_text) if isinstance(response_text, str) else 0
    loss_mask = sample.get("loss_mask") or []
    mask_sum = sum(loss_mask) if isinstance(loss_mask, (list, tuple)) else 0
    return {
        "group_index": attempt_key[0],
        "attempt_id": attempt_key[1],
        "rollout_id": sample.get("rollout_id"),
        "segment_index": segment_index,
        "segment_count": segment_count,
        "instance_id": metadata.get("instance_id") or sample.get("label") or "",
        "sandbox_id": metadata.get("sandbox_id") or "",
        "reward": sample.get("reward"),
        "status": _status_text(sample.get("status")),
        "agent_turns": metadata.get("agent_turns"),
        "tool_calls": metadata.get("tool_calls"),
        "tool_results": metadata.get("tool_results"),
        "diff_line_count": metadata.get("diff_line_count"),
        "phase_time": metadata.get("phase_time") or {},
        "response_length": response_length,
        "mask_sum": mask_sum,
        "trainable_tokens": mask_sum,
        "response_text": response_text,
        "prompt": prompt_text,
        "prompt_summary": prompt_summary,
    }


def samples_index_for(_sample: Dict[str, Any]) -> int:
    """占位：见 _segment_payload 中的 try/except；这里返回 0 让 except 兜底。"""
    return 0


# ---------------------------------------------------------------------------
# 解析主入口
# ---------------------------------------------------------------------------


def parse_attempts_and_segments(dump: Dict[str, Any]) -> Dict[str, Any]:
    """把 dump 拆成 attempts + segments + 整文 response。

    Returns:
        {
            "rollout_id": int,
            "attempts": [
                {
                    "key": (group_index, rollout_id),
                    "segments": [segment_dict, ...],   # 按 segment_index 升序
                    "segment_count": int,
                    "mean_reward": float,
                },
                ...
            ],
            "all_segments": [segment_dict, ...],   # 与 attempts 顺序对应
        }
    """
    if not isinstance(dump, dict):
        raise RolloutInsightException(
            RolloutInsightException.PARSE_DUMP_CORRUPTED, "dump 不是 dict"
        )
    samples = dump.get("samples") or []
    if not isinstance(samples, list):
        raise RolloutInsightException(
            RolloutInsightException.PARSE_DUMP_CORRUPTED, "dump.samples 不是 list"
        )

    rollout_id = dump.get("rollout_id")

    row_keys, attempt_rows = _attempt_layout(samples)
    # 用 sample idx 把 row_keys 排序到与 samples 一致
    attempts_out: List[Dict[str, Any]] = []
    all_segments: List[Dict[str, Any]] = []
    for key, row_indices in attempt_rows.items():
        segs: List[Dict[str, Any]] = []
        rewards: List[float] = []
        for row_idx in row_indices:
            sample = samples[row_idx]
            seg = _segment_payload(sample, key, row_indices)
            segs.append(seg)
            if isinstance(seg.get("reward"), (int, float)):
                rewards.append(float(seg["reward"]))
        # 按 segment_index 升序
        segs.sort(
            key=lambda s: s.get("segment_index")
            if isinstance(s.get("segment_index"), int)
            else 0
        )
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        attempts_out.append(
            {
                "key": list(key),  # tuple → list 便于 JSON 序列化
                "group_index": key[0],
                "rollout_id": key[1],
                "segments": segs,
                "segment_count": len(segs),
                "mean_reward": mean_reward,
            }
        )
        all_segments.extend(segs)

    return {
        "rollout_id": rollout_id,
        "attempts": attempts_out,
        "all_segments": all_segments,
    }


# ---------------------------------------------------------------------------
# turn 切分
# ---------------------------------------------------------------------------

_IM_START_RE = re.compile(r"<\|im_start\|>")
_IM_END_RE = re.compile(r"<\|im_end\|>")
_TOOL_CALL_NAME_RE = re.compile(r"<\|tool_call[^>]*>\s*(\w+)", re.IGNORECASE)
_TOOL_CALL_END_RE = re.compile(r"</?tool_call>", re.IGNORECASE)


def split_response_to_turns(response_text: str) -> List[Dict[str, Any]]:
    """按 ``<|im_start|>role\\n...<|im_end|>`` 切分整文 response。

    每 turn 输出：
        {
            "turn_index": int,    # 1-based
            "role": str,          # 原始 role 字符串
            "role_label": str,    # 归一化：system/user/assistant/tool/unknown
            "body": str,          # turn body（去掉 im_start/im_end）
            "has_tool_call": bool,
            "has_think": bool,
            "body_char_count": int,
            "tool_call_names": List[str],
            "im_start_marker": str,   # 该 turn 起始的 im_start 文本
            "token_estimate": int,
        }
    """
    if not response_text:
        return []

    parts = _IM_START_RE.split(response_text)
    # parts[0] = 首个 im_start 之前的内容（无 role），作为 unknown turn
    turns: List[Dict[str, Any]] = []
    if parts and parts[0].strip():
        turns.append(_build_turn(0, "unknown", "unknown", parts[0], ""))

    for chunk in parts[1:]:
        if not chunk:
            continue
        newline_idx = chunk.find("\n")
        if newline_idx < 0:
            role = chunk.strip()
            body = ""
        else:
            role = chunk[:newline_idx].strip()
            body = chunk[newline_idx + 1 :]
        end_idx = body.find("<|im_end|>")
        if end_idx >= 0:
            body = body[:end_idx]
        role_label = _normalize_role(role)
        turns.append(_build_turn(0, role, role_label, body, "<|im_start|>"))

    # 回填 turn_index
    for i, t in enumerate(turns, 1):
        t["turn_index"] = i
        t["body_char_count"] = len(t["body"])
        t["token_estimate"] = token_estimate(t["body"])
    return turns


def _normalize_role(role: str) -> str:
    r = (role or "").lower().strip()
    if r in ("system",):
        return "system"
    if r in ("user", "human"):
        return "user"
    if r in ("assistant", "gpt", "model"):
        return "assistant"
    if r in ("tool", "tool_response", "toolresponse", "toolresult"):
        return "tool"
    return r or "unknown"


def _build_turn(
    turn_index: int,
    role: str,
    role_label: str,
    body: str,
    im_start_marker: str,
) -> Dict[str, Any]:
    tool_call_names = _TOOL_CALL_NAME_RE.findall(body or "")
    has_tool_call = bool(tool_call_names) or bool(
        _TOOL_CALL_END_RE.search(body or "")
    )
    has_think = "<think>" in (body or "") or "</think>" in (body or "")
    return {
        "turn_index": turn_index,
        "role": role,
        "role_label": role_label,
        "body": body,
        "has_tool_call": has_tool_call,
        "has_think": has_think,
        "body_char_count": len(body or ""),
        "tool_call_names": tool_call_names,
        "im_start_marker": im_start_marker,
        "token_estimate": token_estimate(body or ""),
    }


# ---------------------------------------------------------------------------
# token 估算
# ---------------------------------------------------------------------------


def token_estimate(text: str, chars_per_token: float = 3.5) -> int:
    """粗估 token 数：默认按字符数 / 3.5。"""
    if not text:
        return 0
    return max(1, int(round(len(text) / max(0.1, chars_per_token))))
