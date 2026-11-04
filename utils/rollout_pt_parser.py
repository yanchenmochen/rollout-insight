#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight v0.4.0 —— .pt dump 解析 utils。

复用 inspect-rollout-trajectory 的 ``_attempt_layout`` 规则：
attempt 键 = ``(group_index, rollout_id)``；每个 sample 是该 attempt 的一个
segment，按 ``metadata.segment_index`` 排序。

约定：
- ``load_pt`` 优先走 ``analyze_dump.load_dump``（zip/pickle，无需可 import 的 torch）；
  torch.load 仅作 analyze_dump 内部惰性兜底，本模块**不**在 import 时拉起 torch
- ``parse_attempts_and_segments`` 返回结构化的 attempts + segments + 整文响应
- ``split_response_to_turns`` 按 ``<|im_start|>`` 切分并识别 role
- ``token_estimate`` 默认按字符 / 3.5 粗估
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from data.rollout_insight_exception import RolloutInsightException


__all__ = [
    "load_pt",
    "parse_attempts_and_segments",
    "split_response_to_turns",
    "token_estimate",
    "_attempt_layout",  # 暴露给测试
    "split_response_to_turns_with_tokenizer",  # v0.5.0 T1
    "_discover_tokenizer_path",  # v0.5.0 T1
    "_load_tokenizer",  # v0.5.0 T1
    "_split_token_turns",  # v0.5.0 T1
]


# ---------------------------------------------------------------------------
# .pt 加载（无硬依赖 torch）
# ---------------------------------------------------------------------------


def load_pt(path: str) -> Dict[str, Any]:
    """加载 .pt dump：优先 zip/pickle，不要求环境能 ``import torch``。

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
        from utils.analyze_dump import load_dump

        dump = load_dump(path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("load_pt 失败 path={} err={}", path, exc)
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
_TOOL_CALL_NAME_RE = re.compile(
    r"<(?:\|)?tool_call[^>]*>\s*([A-Za-z0-9_]+)", re.IGNORECASE
)
_MD_TURN_HEAD_RE = re.compile(r"(?m)^### turn \d+ · \S+\s*$")
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
    # parts[0] = 首个 im_start 之前的内容（无 role 标记）：
    # dump 里常见为第一轮 assistant（含 <think>/<tool_call>），与 view_trajectory 一致
    turns: List[Dict[str, Any]] = []
    if parts and parts[0].strip():
        turns.append(_build_turn(0, "assistant", "assistant", parts[0], ""))

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


def split_markdown_headed_turns(text: str) -> List[Dict[str, Any]]:
    """把 ``### turn N · role`` 标题切成多轮。

    inspect / ensure_transcript 落库的正文经常没有 ``<|im_start|>``，
    但保留了这种 markdown 轮次标题。少于 2 个标题时返回空列表，调用方继续用原切分。
    """
    if not text or text.count("### turn") < 2:
        return []
    matches = list(_MD_TURN_HEAD_RE.finditer(text))
    if len(matches) < 2:
        return []
    turns: List[Dict[str, Any]] = []
    for i, match in enumerate(matches):
        head = match.group(0).strip()
        role = head.split("·")[-1].strip() if "·" in head else "unknown"
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        turns.append(_build_turn(i + 1, role, _normalize_role(role), body, "### turn"))
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


# ---------------------------------------------------------------------------
# v0.5.0 T1：基于 tokenizer 的 turn 切分
# ---------------------------------------------------------------------------


def _discover_tokenizer_path(dump_path: str) -> Optional[Path]:
    """在 ``dump_path`` 的所有父目录中找 ``resolved_config.yaml``，再读 ``env.SFT_MODEL`` / ``env.HF_CHECKPOINT``。

    复用 inspect-rollout-trajectory 的解析规则：managed syctool run 的 yaml 通常位于
    ``<run_root>/resolved_config.yaml``，env 段里给出 SFT_MODEL 或 HF_CHECKPOINT 作为
    tokenizer 目录。

    Args:
        dump_path: .pt dump 绝对路径

    Returns:
        解出的 tokenizer 目录路径（``Path``），找不到返回 ``None``。
    """
    if not dump_path:
        return None
    try:
        parents = list(Path(dump_path).resolve().parents)
    except Exception as exc:  # noqa: BLE001
        logger.warning("_discover_tokenizer_path: resolve {} 失败: {}", dump_path, exc)
        return None
    try:
        import yaml  # PyYAML
    except Exception as exc:  # noqa: BLE001
        logger.warning("_discover_tokenizer_path: 未安装 PyYAML，跳过发现: {}", exc)
        return None

    for parent in parents:
        config_path = parent / "resolved_config.yaml"
        if not config_path.is_file():
            continue
        try:
            with config_path.open(encoding="utf-8") as config_file:
                cfg = yaml.safe_load(config_file) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_discover_tokenizer_path: 解析 {} 失败: {}", config_path, exc
            )
            continue
        env = (cfg or {}).get("env") or {}
        model_path = env.get("SFT_MODEL") or env.get("HF_CHECKPOINT")
        if model_path:
            return Path(model_path)
    return None


def _load_tokenizer(path: str | Path) -> Any:
    """从本地目录加载 Hugging Face ``tokenizer.json``，不引入 transformers。

    Args:
        path: tokenizer 目录路径；或直接指向 ``tokenizer.json`` 的文件路径。

    Returns:
        ``tokenizers.Tokenizer`` 实例（解码/切分用）。
    """
    # 在函数内导入：避免顶级依赖 + 当未安装 tokenizers 时让整文件可加载
    from tokenizers import Tokenizer

    p = Path(path) if not isinstance(path, Path) else path
    tokenizer_json = p if p.name == "tokenizer.json" else p / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"tokenizer.json not found under {path}")
    return Tokenizer.from_file(str(tokenizer_json))


def _split_token_turns(
    tokens: Sequence[int],
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    """在 token 序列上找 ``<|im_start|>`` 的 id，按出现位置切成多段并 decode。

    返回元素形如::

        {
            "role": str,
            "role_label": str,      # system/user/assistant/tool/unknown
            "body": str,            # 已去掉 <|im_start|>/<|im_end|> 的纯文本
            "token_start": int,     # 闭区间起点
            "token_end": int,       # 开区间终点
            "token_count": int,
        }

    - 若 tokenizer 无 ``<|im_start|>`` token id，整段序列作单元素返回（role="sequence"）。
    - 若 tokens 里没有任何 ``<|im_start|>``，整段也作单元素。
    """
    if tokens is None or len(tokens) == 0 or tokenizer is None:
        return []

    turn_start_id = tokenizer.token_to_id("<|im_start|>")
    if turn_start_id is None:
        # fallback: 把整段视作单个 sequence；前端用 response 文本切分兜底
        text = tokenizer.decode(list(tokens), skip_special_tokens=False)
        return [
            {
                "role": "sequence",
                "role_label": "unknown",
                "body": text,
                "token_start": 0,
                "token_end": len(tokens),
                "token_count": len(tokens),
            }
        ]

    starts = [i for i, tid in enumerate(tokens) if tid == turn_start_id]
    if not starts:
        text = tokenizer.decode(list(tokens), skip_special_tokens=False)
        return [
            {
                "role": "sequence",
                "role_label": "unknown",
                "body": text,
                "token_start": 0,
                "token_end": len(tokens),
                "token_count": len(tokens),
            }
        ]

    bounds = ([0] if starts[0] else []) + starts + [len(tokens)]
    marker = "<|im_start|>"
    turns: List[Dict[str, Any]] = []
    for start, end in zip(bounds, bounds[1:]):
        text = tokenizer.decode(list(tokens[start:end]), skip_special_tokens=False)
        if tokens[start] != turn_start_id:
            # 段首不是 im_start；视作续接文本，拼成 sequence
            turns.append(
                {
                    "role": "sequence",
                    "role_label": "unknown",
                    "body": text,
                    "token_start": int(start),
                    "token_end": int(end),
                    "token_count": int(end - start),
                }
            )
            continue
        body = text[len(marker):] if text.startswith(marker) else text
        role, separator, content = body.partition("\n")
        if not separator:
            role_label = _normalize_role(role.strip() or "")
            turns.append(
                {
                    "role": role.strip() or "?",
                    "role_label": role_label,
                    "body": "",
                    "token_start": int(start),
                    "token_end": int(end),
                    "token_count": int(end - start),
                }
            )
            continue
        content = re.sub(r"<\|im_end\|>\s*$", "", content)
        role_label = _normalize_role(role.strip() or "")
        turns.append(
            {
                "role": role.strip() or "?",
                "role_label": role_label,
                "body": content,
                "token_start": int(start),
                "token_end": int(end),
                "token_count": int(end - start),
            }
        )
    return turns


def _decode_turn_chunks_to_turn_dicts(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """把 ``_split_token_turns`` 的输出转成与 ``split_response_to_turns`` 同 schema 的 dict。

    用法：当 tokens + tokenizer 可用时，外部调 ``_split_token_turns`` 后再过本函数，
    让``upsert_segment_turns`` 的入参 schema 保持一致。
    """
    out: List[Dict[str, Any]] = []
    for idx, ch in enumerate(chunks, 1):
        body = ch.get("body", "") or ""
        role = ch.get("role", "") or ""
        role_label = ch.get("role_label", "unknown") or "unknown"
        out.append(
            _build_turn(
                turn_index=idx,
                role=role,
                role_label=role_label,
                body=body,
                im_start_marker=("<|im_start|>" if ch.get("token_start") is not None else ""),
            )
        )
    return out


def split_response_to_turns_with_tokenizer(
    response_text: str = "",
    *,
    tokens: Optional[Sequence[int]] = None,
    tokenizer: Any = None,
    tokenizer_path: Optional[str] = None,
    dump_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """v0.5.0 T1：优先用 tokens + tokenizer 切分；否则 fallback 到 response 文本。

    Args:
        response_text: 整文 response（兜底用）
        tokens: ``sample['tokens']`` 列表（可选）
        tokenizer: 已加载的 ``tokenizers.Tokenizer``（可选，优先）
        tokenizer_path: 已知的 tokenizer 目录路径（可选，用于临时加载）
        dump_path: ``.pt`` 绝对路径（可选；用于 ``_discover_tokenizer_path``）

    Returns:
        (turns_list, source) — ``source`` 取值 ``"tokens"`` / ``"text"``。
    """
    # 1) 优先：tokens + tokenizer 都给齐了，直接走 token 路径
    if tokens and len(tokens) > 0 and tokenizer is not None:
        chunks = _split_token_turns(tokens, tokenizer)
        return _decode_turn_chunks_to_turn_dicts(chunks), "tokens"

    # 2) 兜底1：tokens 有、tokenizer 没给出 → 尝试加载（先 path，再 discover）
    if tokens and len(tokens) > 0 and tokenizer is None:
        loaded = _try_load_tokenizer_for_dump(
            tokenizer_path=tokenizer_path, dump_path=dump_path
        )
        if loaded is not None:
            chunks = _split_token_turns(tokens, loaded)
            return _decode_turn_chunks_to_turn_dicts(chunks), "tokens"

    # 3) 兜底2：response 文本切分
    return split_response_to_turns(response_text or ""), "text"


def _try_load_tokenizer_for_dump(
    tokenizer_path: Optional[str] = None,
    dump_path: Optional[str] = None,
) -> Optional[Any]:
    """按优先级加载 tokenizer：

    1. 显式 ``tokenizer_path``
    2. ``_discover_tokenizer_path(dump_path)``

    失败时返回 ``None``，不抛异常（让调用方走 fallback）。
    """
    candidates: List[Path] = []
    if tokenizer_path:
        try:
            candidates.append(Path(tokenizer_path))
        except Exception:  # noqa: BLE001
            pass
    discovered = _discover_tokenizer_path(dump_path) if dump_path else None
    if discovered is not None:
        candidates.append(discovered)
    for cand in candidates:
        try:
            return _load_tokenizer(cand)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "split_response_to_turns_with_tokenizer: 加载 tokenizer {} 失败: {}",
                cand,
                exc,
            )
    return None
