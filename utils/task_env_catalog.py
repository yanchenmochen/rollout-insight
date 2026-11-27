#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""任务训练环境变量目录与差异计算。

从 run 目录 / 南湖作业读到的 env 往往有上百项；排查奖励不涨时只需要盯住
「影响采样 / 过滤 / 时间预算 / 优化器」的关键键。本模块提供：

- 关键键分组目录（``ENV_KEY_CATALOG``）
- 敏感值脱敏（``redact_env_value``）
- 两份 / 多份 env 的结构化 diff（``diff_env_maps`` / ``compare_task_envs``）
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ENV_KEY_CATALOG",
    "ENV_CATEGORIES",
    "catalog_entry",
    "classify_env_key",
    "redact_env_value",
    "is_sensitive_key",
    "pick_highlight_env",
    "summarize_env_map",
    "diff_env_maps",
    "compare_env_maps",
]

# ---------------------------------------------------------------------------
# 目录：key → (category, label, why_it_matters)
# ---------------------------------------------------------------------------

ENV_CATEGORIES: Dict[str, str] = {
    "job": "作业类型",
    "sampling": "采样与组大小",
    "filter": "动态过滤 / DAPO",
    "budget": "Agent 时间预算",
    "optimizer": "优化器 / RL",
    "model": "模型与路径",
    "eval": "评测",
    "other": "其它",
}

# 顺序即展示优先级；同组内按此表排列
ENV_KEY_CATALOG: Dict[str, Tuple[str, str, str]] = {
    # 作业类型
    "TRAINING_SCRIPT": ("job", "训练脚本", "判定真实训练 / 评测 / rollout-only"),
    "TRAIN_SCRIPT": ("job", "训练脚本(别名)", "同 TRAINING_SCRIPT"),
    "DEBUG_ROLLOUT_ONLY": ("job", "仅 rollout", "1=断点续推，不更新参数"),
    "NUM_ROLLOUT": ("job", "rollout 步数", "0 且未禁评测时按评测作业处理"),
    "DISABLE_EVAL": ("job", "禁用评测", "与 NUM_ROLLOUT 一起判定作业类型"),
    "COLOCATE": ("job", "同机部署", "训练与推理是否共置"),
    "ROLLOUT_SHUFFLE": ("job", "rollout 洗牌", "题目采样顺序"),
    # 采样
    "ROLLOUT_BATCH_SIZE": ("sampling", "每步题目数", "epoch 窗口 / 曲线聚合依据"),
    "N_SAMPLES_PER_PROMPT": ("sampling", "每题 attempt 数", "GRPO 组大小，常见 8"),
    "GROUP_SIZE": ("sampling", "组大小", "同 N_SAMPLES_PER_PROMPT"),
    "NUM_SAMPLES_PER_PROMPT": ("sampling", "每题样本数", "同 N_SAMPLES_PER_PROMPT"),
    # 过滤 / DAPO
    "DYNAMIC_SAMPLING_FILTER_PATH": (
        "filter",
        "动态过滤路径",
        "决定哪些组进训练（如 check_surviving_reward_nonzero_std）",
    ),
    "SWE_DAPO_MIN_SURVIVING_ATTEMPTS": (
        "filter",
        "最少存活 attempt",
        "低于此值整组丢弃；默认 4",
    ),
    "SWE_EXCLUDE_TIME_BUDGET": (
        "filter",
        "排除时间预算 abort",
        "1 时 agent_time_budget_exceeded 不进训练",
    ),
    # Agent 预算（奖励不涨第一主因相关）
    "SWE_AGENT_TIME_BUDGET_SEC": (
        "budget",
        "Agent 时间预算(秒)",
        "超时 → abort → 存活不足 → 组被丢",
    ),
    "AGENT_TIME_BUDGET_SEC": ("budget", "Agent 时间预算(秒)", "同 SWE_AGENT_TIME_BUDGET_SEC"),
    "MAX_AGENT_TIMEOUT": ("budget", "Agent 超时", "超时阈值"),
    "AGENT_MAX_TURNS": ("budget", "最大轮数", "轮数爆炸会拖垮时间预算"),
    "MAX_TURNS": ("budget", "最大轮数", "同 AGENT_MAX_TURNS"),
    "SWE_AGENT_MAX_TURNS": ("budget", "最大轮数", "同 AGENT_MAX_TURNS"),
    # 优化器
    "LR": ("optimizer", "学习率", "过大易塌、过小不涨"),
    "LEARNING_RATE": ("optimizer", "学习率", "同 LR"),
    "KL_COEF": ("optimizer", "KL 系数", "约束策略偏移"),
    "KL_COEFF": ("optimizer", "KL 系数", "同 KL_COEF"),
    "ENTROPY_COEF": ("optimizer", "熵系数", "探索强度"),
    "ENTROPY_COEFF": ("optimizer", "熵系数", "同 ENTROPY_COEF"),
    "CLIP_RATIO": ("optimizer", "PPO clip", "更新幅度"),
    "EPS_CLIP": ("optimizer", "PPO clip", "同 CLIP_RATIO"),
    # 模型 / 路径
    "MODEL_NAME": ("model", "模型名", "基座模型标识"),
    "HF_MODEL_PATH": ("model", "HF 模型路径", "权重目录"),
    "TOKENIZER_DIR": ("model", "Tokenizer 目录", "解码依赖"),
    "OUTPUT_DIR": ("model", "输出目录", "run 根路径"),
    "ROLLOUT_DUMP_DIR": ("model", "Dump 目录", "rollout .pt 落盘"),
    "ROLLOUT_DUMP_TEMPLATE": ("model", "Dump 模板", "南湖侧 dumps 路径模板"),
    # 评测
    "SLIME_EVAL_CONFIG": ("eval", "评测配置", "评测作业 panel / yaml"),
    "EVAL_DATASETS": ("eval", "评测集", "外部评测数据集列表"),
}

_SENSITIVE_SUBSTR = (
    "PASSWORD",
    "SECRET",
    "TOKEN",
    "API_KEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
    "CREDENTIAL",
    "PASSWD",
)


def catalog_entry(key: str) -> Optional[Dict[str, str]]:
    """查目录项；未知键返回 None。"""
    raw = str(key or "").strip()
    if not raw:
        return None
    hit = ENV_KEY_CATALOG.get(raw) or ENV_KEY_CATALOG.get(raw.upper())
    if hit is None:
        # 大小写不敏感匹配
        upper = raw.upper()
        for k, v in ENV_KEY_CATALOG.items():
            if k.upper() == upper:
                hit = v
                break
    if hit is None:
        return None
    category, label, why = hit
    return {
        "key": raw,
        "category": category,
        "category_label": ENV_CATEGORIES.get(category, category),
        "label": label,
        "why": why,
    }


def classify_env_key(key: str) -> str:
    """返回 category 名；未知为 ``other``。"""
    entry = catalog_entry(key)
    return entry["category"] if entry else "other"


def is_sensitive_key(key: str) -> bool:
    upper = str(key or "").upper()
    return any(s in upper for s in _SENSITIVE_SUBSTR)


def redact_env_value(key: str, value: Any) -> str:
    """敏感键脱敏；其余转成短字符串。"""
    text = "" if value is None else str(value)
    if is_sensitive_key(key):
        if not text:
            return ""
        if len(text) <= 4:
            return "****"
        return text[:2] + "****" + text[-2:]
    return text


def pick_highlight_env(
    env: Optional[Mapping[str, Any]],
    *,
    include_unknown: bool = False,
) -> List[Dict[str, Any]]:
    """按目录顺序抽出关键环境变量行。

    每行：``key / value / category / category_label / label / why / present``。
    ``present=False`` 表示目录里有但 env 里没有（便于对比缺省）。
    """
    env_map = {str(k): ("" if v is None else str(v)) for k, v in (env or {}).items()}
    # 大小写不敏感查找
    lower_index = {k.lower(): k for k in env_map}

    rows: List[Dict[str, Any]] = []
    seen_lower = set()
    for catalog_key, (category, label, why) in ENV_KEY_CATALOG.items():
        real_key = lower_index.get(catalog_key.lower())
        present = real_key is not None
        value = env_map.get(real_key, "") if present else ""
        rows.append(
            {
                "key": real_key or catalog_key,
                "catalog_key": catalog_key,
                "value": redact_env_value(catalog_key, value) if present else "",
                "raw_present": present,
                "present": present,
                "category": category,
                "category_label": ENV_CATEGORIES.get(category, category),
                "label": label,
                "why": why,
                "highlighted": True,
            }
        )
        if present:
            seen_lower.add(real_key.lower())

    if include_unknown:
        for key in sorted(env_map.keys(), key=lambda x: x.upper()):
            if key.lower() in seen_lower:
                continue
            rows.append(
                {
                    "key": key,
                    "catalog_key": key,
                    "value": redact_env_value(key, env_map[key]),
                    "raw_present": True,
                    "present": True,
                    "category": "other",
                    "category_label": ENV_CATEGORIES["other"],
                    "label": key,
                    "why": "",
                    "highlighted": False,
                }
            )
    return rows


def summarize_env_map(env: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """单任务 env 摘要：关键行 + 计数。"""
    env_map = {str(k): ("" if v is None else str(v)) for k, v in (env or {}).items()}
    highlight = pick_highlight_env(env_map, include_unknown=False)
    present_highlight = [r for r in highlight if r["present"]]
    by_category: Dict[str, int] = {}
    for row in present_highlight:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1
    return {
        "env_count": len(env_map),
        "highlight_count": len(present_highlight),
        "highlight": present_highlight,
        "highlight_catalog": highlight,  # 含缺省行，便于 UI 显示「未设置」
        "by_category": [
            {
                "category": cat,
                "category_label": ENV_CATEGORIES.get(cat, cat),
                "count": by_category.get(cat, 0),
            }
            for cat in ENV_CATEGORIES
            if by_category.get(cat, 0) > 0
        ],
    }


def diff_env_maps(
    left: Optional[Mapping[str, Any]],
    right: Optional[Mapping[str, Any]],
    *,
    left_label: str = "A",
    right_label: str = "B",
    only_highlight: bool = True,
) -> Dict[str, Any]:
    """两份 env 的差异：changed / only_left / only_right / same。"""
    left_map = {str(k): ("" if v is None else str(v)) for k, v in (left or {}).items()}
    right_map = {str(k): ("" if v is None else str(v)) for k, v in (right or {}).items()}

    if only_highlight:
        keys = list(ENV_KEY_CATALOG.keys())
        # 同时带上两侧实际出现、且在目录别名里的键
        lower_to_canon = {k.lower(): k for k in ENV_KEY_CATALOG}
        for src in (left_map, right_map):
            for k in src:
                canon = lower_to_canon.get(k.lower())
                if canon and canon not in keys:
                    keys.append(canon)
    else:
        keys = sorted(set(left_map) | set(right_map), key=lambda x: x.upper())

    def _lookup(m: Mapping[str, str], key: str) -> Tuple[bool, str, str]:
        if key in m:
            return True, key, m[key]
        lower = key.lower()
        for k, v in m.items():
            if k.lower() == lower:
                return True, k, v
        return False, key, ""

    changed: List[Dict[str, Any]] = []
    only_left: List[Dict[str, Any]] = []
    only_right: List[Dict[str, Any]] = []
    same: List[Dict[str, Any]] = []

    seen = set()
    for catalog_key in keys:
        if catalog_key.lower() in seen:
            continue
        seen.add(catalog_key.lower())
        l_ok, l_key, l_val = _lookup(left_map, catalog_key)
        r_ok, r_key, r_val = _lookup(right_map, catalog_key)
        entry = catalog_entry(catalog_key) or {
            "key": catalog_key,
            "category": "other",
            "category_label": ENV_CATEGORIES["other"],
            "label": catalog_key,
            "why": "",
        }
        base = {
            "key": l_key if l_ok else (r_key if r_ok else catalog_key),
            "catalog_key": catalog_key,
            "category": entry["category"],
            "category_label": entry["category_label"],
            "label": entry["label"],
            "why": entry.get("why") or "",
            "left": redact_env_value(catalog_key, l_val) if l_ok else None,
            "right": redact_env_value(catalog_key, r_val) if r_ok else None,
            "left_present": l_ok,
            "right_present": r_ok,
        }
        if l_ok and r_ok:
            if l_val == r_val:
                same.append({**base, "status": "same"})
            else:
                changed.append({**base, "status": "changed"})
        elif l_ok and not r_ok:
            only_left.append({**base, "status": "only_left"})
        elif r_ok and not l_ok:
            only_right.append({**base, "status": "only_right"})

    return {
        "left_label": left_label,
        "right_label": right_label,
        "changed": changed,
        "only_left": only_left,
        "only_right": only_right,
        "same": same,
        "counts": {
            "changed": len(changed),
            "only_left": len(only_left),
            "only_right": len(only_right),
            "same": len(same),
        },
    }


def compare_env_maps(
    envs: Sequence[Mapping[str, Any]],
    *,
    labels: Optional[Sequence[str]] = None,
    baseline_index: int = 0,
    only_highlight: bool = True,
) -> Dict[str, Any]:
    """多任务 env 对比：相对基线的逐键矩阵 + 汇总。

    返回：
    - ``rows``：每行一个 key，含 ``values[]`` / ``statuses[]`` / ``all_same``
    - ``pair_diffs``：基线 vs 每个其它任务的 ``diff_env_maps`` 摘要
    - ``summary``：差异键数等
    """
    n = len(envs)
    if n == 0:
        return {
            "baseline_index": 0,
            "labels": [],
            "rows": [],
            "pair_diffs": [],
            "summary": {
                "task_count": 0,
                "diff_key_count": 0,
                "same_key_count": 0,
                "compared_key_count": 0,
            },
        }

    label_list: List[str] = []
    for i in range(n):
        if labels and i < len(labels) and labels[i]:
            label_list.append(str(labels[i]))
        else:
            label_list.append(f"task_{i}")

    maps: List[Dict[str, str]] = [
        {str(k): ("" if v is None else str(v)) for k, v in (env or {}).items()}
        for env in envs
    ]
    base_i = max(0, min(int(baseline_index or 0), n - 1))

    if only_highlight:
        key_order = list(ENV_KEY_CATALOG.keys())
    else:
        union = set()
        for m in maps:
            union.update(m.keys())
        key_order = sorted(union, key=lambda x: x.upper())

    def _get(m: Mapping[str, str], key: str) -> Tuple[bool, str]:
        if key in m:
            return True, m[key]
        lower = key.lower()
        for k, v in m.items():
            if k.lower() == lower:
                return True, v
        return False, ""

    rows: List[Dict[str, Any]] = []
    diff_keys = 0
    same_keys = 0
    for catalog_key in key_order:
        entry = catalog_entry(catalog_key) or {
            "category": "other",
            "category_label": ENV_CATEGORIES["other"],
            "label": catalog_key,
            "why": "",
        }
        values: List[Optional[str]] = []
        presents: List[bool] = []
        for m in maps:
            ok, val = _get(m, catalog_key)
            presents.append(ok)
            values.append(redact_env_value(catalog_key, val) if ok else None)

        # 目录键在所有任务都未出现 → 跳过（避免刷一堆空行）
        if only_highlight and not any(presents):
            continue

        base_val = values[base_i]
        base_present = presents[base_i]
        statuses: List[str] = []
        for i, (val, present) in enumerate(zip(values, presents)):
            if i == base_i:
                statuses.append("baseline")
            elif present and base_present and val == base_val:
                statuses.append("same")
            elif present and base_present and val != base_val:
                statuses.append("changed")
            elif present and not base_present:
                statuses.append("only_here")
            elif (not present) and base_present:
                statuses.append("missing")
            else:
                statuses.append("absent")

        # 全部缺席，或全部出现且取值一致 → same
        present_vals = {values[i] for i in range(n) if presents[i]}
        all_same = (not any(presents)) or (
            all(presents) and len(present_vals) == 1
        )
        if all_same:
            same_keys += 1
        else:
            diff_keys += 1

        rows.append(
            {
                "key": catalog_key,
                "catalog_key": catalog_key,
                "category": entry["category"],
                "category_label": entry["category_label"],
                "label": entry["label"],
                "why": entry.get("why") or "",
                "values": values,
                "presents": presents,
                "statuses": statuses,
                "all_same": all_same,
            }
        )

    # 差异行优先
    rows.sort(key=lambda r: (r["all_same"], r["category"], r["key"]))

    pair_diffs: List[Dict[str, Any]] = []
    for i in range(n):
        if i == base_i:
            continue
        pair_diffs.append(
            {
                "task_index": i,
                "label": label_list[i],
                **diff_env_maps(
                    maps[base_i],
                    maps[i],
                    left_label=label_list[base_i],
                    right_label=label_list[i],
                    only_highlight=only_highlight,
                ),
            }
        )

    return {
        "baseline_index": base_i,
        "baseline_label": label_list[base_i],
        "labels": label_list,
        "rows": rows,
        "diff_rows": [r for r in rows if not r["all_same"]],
        "same_rows": [r for r in rows if r["all_same"]],
        "pair_diffs": pair_diffs,
        "summary": {
            "task_count": n,
            "diff_key_count": diff_keys,
            "same_key_count": same_keys,
            "compared_key_count": len(rows),
        },
    }
