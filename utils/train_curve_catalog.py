#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""训练曲线指标目录：默认按钮组 + tag 匹配规则。

用户需求（训练曲线页按钮组）：
- 默认展示：奖励、长度、梯度、entropy_loss、context_exhausted、
  max_fake_responses、rollout_time
- 用户可自行勾选要展示的指标，配置持久化后长期生效
- 每个任务可以额外新增自己的展示项（任务级追加）

设计要点：
- 指标用「键（key）」标识，键到实际 TB tag 的匹配由 ``match_tag`` 规则决定，
  避免不同实验的 tag 前缀（``rollout/`` / ``train/`` / ``actor/`` …）写死在配置里。
- 一个指标可以命中**多个 tag**（例如「奖励」对应 generated / paired / filtered
  三条 raw_reward），这些 tag 会画在**同一张图**里，用不同颜色区分。
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
    "is_custom_key",
]


# 指标元信息：key → 中文名 / 匹配规则 / 说明
#
# ``series``：该指标要画在**同一张图**里的标准子曲线（每条一个 tag）。
#   命中时优先按这里取，保证「奖励」就三条曲线，不被 coverage / count 之类的
#   派生指标污染。
# ``patterns``：标准 tag 不存在时的兜底正则（锚定到路径段，避免误匹配
#   ``raw_reward_coverage`` 这类派生指标）。
# ``prefer``：兜底匹配时多个 tag 的展示顺序。
CURVE_METRICS: Dict[str, Dict[str, Any]] = {
    "raw_reward": {
        "label": "奖励",
        "desc": (
            "训练拿到的实际奖励。同一张图里画三条子曲线："
            "generated（全部生成样本）、paired（配对去偏后）、filtered（过滤后进入训练的）。"
        ),
        "patterns": (
            r"(^|/)paired_raw_reward$",
            r"(^|/)raw_reward$",
        ),
        "prefer": (
            "rollout/generated/raw_reward",
            "rollout/generated/paired_raw_reward",
            "rollout/filtered/raw_reward",
        ),
        "series": (
            "rollout/generated/raw_reward",
            "rollout/generated/paired_raw_reward",
            "rollout/filtered/raw_reward",
        ),
    },
    "response_length": {
        "label": "长度",
        "desc": "响应长度均值（rollout/response_len/mean），反映轨迹变长还是变短。",
        "patterns": (
            r"(^|/)response_len/mean$",
            r"(^|/)response_length/mean$",
            r"(^|/)mean_response_length$",
        ),
        "prefer": (
            "rollout/response_len/mean",
            "response_len/mean",
            "mean_response_length",
        ),
        "series": ("rollout/response_len/mean",),
    },
    "grad_norm": {
        "label": "梯度",
        "desc": "梯度范数（grad norm），过大通常意味着训练不稳。",
        "patterns": (
            r"(^|/)grad_norm$",
            r"(^|/)gradient_norm$",
        ),
        "prefer": ("train/grad_norm", "actor/grad_norm", "optimizer/grad_norm"),
        "series": ("train/grad_norm",),
    },
    "entropy_loss": {
        "label": "entropy_loss",
        "desc": "熵损失；持续下降通常意味着策略过早收敛。",
        "patterns": (
            r"(^|/)entropy_loss$",
            r"(^|/)entropy$",
        ),
        "prefer": ("train/entropy_loss", "actor/entropy_loss"),
        "series": ("train/entropy_loss",),
    },
    "context_exhausted": {
        "label": "context_exhausted",
        "desc": "上下文打满的样本占比（rollout/outcome/context_exhausted）；升高说明上下文长度成为瓶颈。",
        "patterns": (
            r"(^|/)context_exhausted$",
            r"(^|/)ctx_exhausted$",
        ),
        "prefer": (
            "rollout/outcome/context_exhausted",
            "rollout/context_exhausted",
        ),
        "series": ("rollout/outcome/context_exhausted",),
    },
    "max_fake_responses": {
        "label": "max_fake_responses",
        "desc": "单条样本最多触发多少次假响应（rollout/outcome/max_fake_responses）；升高说明模型在拿假反馈自我欺骗。",
        "patterns": (
            r"(^|/)max_fake_responses?$",
            r"(^|/)fake_responses?$",
        ),
        "prefer": (
            "rollout/outcome/max_fake_responses",
            "max_fake_responses",
        ),
        "series": ("rollout/outcome/max_fake_responses",),
    },
    "rollout_time": {
        "label": "rollout_time",
        "desc": "一轮 rollout 采样耗时（perf/rollout_time，单位秒）；反映采样性能与并发瓶颈。",
        "patterns": (
            r"(^|/)rollout_time$",
            r"(^|/)rollout_duration$",
        ),
        "prefer": (
            "perf/rollout_time",
            "rollout_time",
        ),
        "series": ("perf/rollout_time",),
    },
}

# 默认按钮组（用户未配置时使用）
DEFAULT_CURVE_KEYS: List[str] = [
    "raw_reward",
    "response_length",
    "grad_norm",
    "entropy_loss",
    "context_exhausted",
    "max_fake_responses",
    "rollout_time",
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
    """在可用 tag 里找出命中该指标的所有 tag。

    解析顺序：
    1. **标准 tag 优先**（``CURVE_METRICS[key]["series"]``）：只要它出现在可用 tag 里
       就直接采用。这保证「奖励」稳定是 generated / paired / filtered 三条曲线，
       不会被 ``raw_reward_coverage`` 之类的派生指标污染。
    2. 标准 tag 缺失时退回**兜底正则**匹配（锚定到路径段），并按 ``prefer`` 排序。

    自定义指标（``custom:<tag>``）按 tag 名精确匹配。
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

    available = [str(t) for t in available_tags]
    available_set = set(available)

    # 1) 标准 tag 命中
    standard = [str(t) for t in meta.get("series", ()) if str(t) in available_set]
    if standard:
        return standard

    # 2) 兜底正则匹配
    matched = [t for t in available if match_tag(text_key, t)]
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
