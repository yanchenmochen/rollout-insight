#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""训练曲线指标目录：默认按钮组 + tag 匹配规则。

用户需求（训练曲线页按钮组）：
- 默认展示：混合奖励、长度、梯度、entropy_loss、context_exhausted
- 用户可自行勾选要展示的指标，配置持久化后长期生效
- 每个任务可以额外新增自己的展示项（任务级追加）

设计要点：
- 指标用「键（key）」标识，键到实际 TB tag 的匹配由 ``match_tag`` 规则决定，
  避免不同实验的 tag 前缀（``rollout/`` / ``actor/`` / ``critic/`` …）写死在配置里。
- 只登记代码里能对上语义的 tag，不给 loss / kl 等曲线编造含义。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "DEFAULT_CURVE_KEYS",
    "CURVE_METRICS",
    "curve_metric_keys",
    "curve_metric",
    "match_tag",
    "resolve_metric_tags",
    "normalize_curve_keys",
    "available_curve_metrics",
]


# 指标元信息：key → 中文名 / 匹配规则 / 说明
CURVE_METRICS: Dict[str, Dict[str, Any]] = {
    "mixed_reward": {
        "label": "混合奖励",
        "desc": "混合组（既有 reward≈1 又有 reward≈0）的平均奖励。",
        "patterns": (
            r"mixed[_\-/]?reward",
            r"reward[_\-/]?mixed",
            r"mean[_\-/]?mixed[_\-/]?reward",
        ),
        "prefer": ("rollout/mixed_reward", "mixed_reward", "mean_mixed_reward"),
    },
    "response_length": {
        "label": "长度",
        "desc": "响应长度（response length），反映轨迹变长还是变短。",
        "patterns": (
            r"response[_\-/]?length",
            r"resp[_\-/]?len",
            r"sequence[_\-/]?length",
            r"seq[_\-/]?len",
        ),
        "prefer": ("rollout/response_length", "response_length", "mean_response_length"),
    },
    "grad_norm": {
        "label": "梯度",
        "desc": "梯度范数（grad norm），过大通常意味着训练不稳。",
        "patterns": (
            r"grad[_\-/]?norm",
            r"gradient[_\-/]?norm",
        ),
        "prefer": ("actor/grad_norm", "grad_norm", "optimizer/grad_norm"),
    },
    "entropy_loss": {
        "label": "entropy_loss",
        "desc": "熵损失；持续下降通常意味着策略过早收敛。",
        "patterns": (
            r"entropy[_\-/]?loss",
            r"entropy",
        ),
        "prefer": ("actor/entropy_loss", "entropy_loss", "entropy"),
    },
    "context_exhausted": {
        "label": "context_exhausted",
        "desc": "上下文打满的样本占比；升高说明上下文长度成为瓶颈。",
        "patterns": (
            r"context[_\-/]?exhausted",
            r"context[_\-/]?full",
            r"ctx[_\-/]?exhausted",
        ),
        "prefer": ("rollout/context_exhausted", "context_exhausted"),
    },
}

# 默认按钮组（用户未配置时使用）
DEFAULT_CURVE_KEYS: List[str] = [
    "mixed_reward",
    "response_length",
    "grad_norm",
    "entropy_loss",
    "context_exhausted",
]

# 用户可自行追加的自定义指标（键名形如 ``custom:xxx``，直接写 TB tag）
_CUSTOM_PREFIX = "custom:"


def curve_metric_keys() -> List[str]:
    """返回全部内置指标键。"""
    return list(CURVE_METRICS.keys())


def curve_metric(key: str) -> Optional[Dict[str, Any]]:
    """按 key 取指标元信息；未知返回 None。"""
    return CURVE_METRICS.get(str(key or "").strip())


def available_curve_metrics() -> List[Dict[str, Any]]:
    """列出内置指标，供页面渲染勾选框。"""
    return [
        {
            "key": key,
            "label": meta["label"],
            "desc": meta["desc"],
            "is_default": key in DEFAULT_CURVE_KEYS,
        }
        for key, meta in CURVE_METRICS.items()
    ]


def normalize_curve_keys(keys: Optional[Sequence[str]]) -> List[str]:
    """规整指标 key 列表：去空白、去重、保持顺序、过滤未知内置键。

    自定义项（``custom:<tag>``）原样保留。
    """
    out: List[str] = []
    for raw in keys or []:
        text = str(raw or "").strip()
        if not text:
            continue
        if text.startswith(_CUSTOM_PREFIX):
            tag = text[len(_CUSTOM_PREFIX):].strip()
            if not tag:
                continue
            if text not in out:
                out.append(text)
            continue
        if text in CURVE_METRICS and text not in out:
            out.append(text)
    return out


def _compiled_patterns(key: str) -> List[re.Pattern]:
    meta = CURVE_METRICS.get(key)
    if not meta:
        return []
    return [
        re.compile(pattern, re.IGNORECASE)
        for pattern in meta.get("patterns", ())
    ]


def match_tag(key: str, tag: str) -> bool:
    """判断某个 TB tag 是否命中指标 key 的匹配规则。"""
    text = str(tag or "").strip()
    if not text:
        return False
    if str(key or "").strip().startswith(_CUSTOM_PREFIX):
        return text == str(key).strip()[len(_CUSTOM_PREFIX):].strip()
    for pattern in _compiled_patterns(key):
        if pattern.search(text):
            return True
    return False


def resolve_metric_tags(
    key: str,
    available_tags: Sequence[str],
) -> List[str]:
    """在可用 tag 里找出命中该指标的所有 tag（按 prefer 顺序优先）。

    同一个指标可能命中多个 tag（例如 ``rollout/mixed_reward`` 与
    ``critic/mixed_reward``）；这里全部返回，由页面决定画哪条。
    """
    text_key = str(key or "").strip()
    if not text_key:
        return []
    if text_key.startswith(_CUSTOM_PREFIX):
        want = text_key[len(_CUSTOM_PREFIX):].strip()
        return [t for t in available_tags if str(t).strip() == want]

    meta = CURVE_METRICS.get(text_key)
    if not meta:
        return []

    matched = [t for t in available_tags if match_tag(text_key, t)]
    if not matched:
        return []

    prefer = [str(p) for p in meta.get("prefer", ())]
    ordered: List[str] = []
    for candidate in prefer:
        for tag in matched:
            if tag == candidate and tag not in ordered:
                ordered.append(tag)
    for tag in matched:
        if tag not in ordered:
            ordered.append(tag)
    return ordered


def is_custom_key(key: str) -> bool:
    """是否用户自定义指标键。"""
    return str(key or "").strip().startswith(_CUSTOM_PREFIX)
