#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""奖励设计实践策略（静态文案，不在请求时联网）。

每条策略对应 ``data.agentic_problem_catalog`` 里的问题 id，
页面用锚点链到问题分析 / 奖励设计的问题目录。
"""
from __future__ import annotations

from typing import Any, Dict, List

__all__ = [
    "REWARD_FLOW_NOTES",
    "REWARD_DESIGN_PLAYBOOK",
    "list_playbook_entries",
]

#: 当前已实现、可被可视化的奖励事实（本仓库不计算奖励函数）
REWARD_FLOW_NOTES: List[str] = [
    "进训练的是评测二值分：reward≈1 通过、≈0 失败；.pt 落盘是 raw reward，不是 advantage。",
    "abort / 部分 timeout 不可测，不应写成 reward 0 再进 GRPO，否则策略在学基建噪声。",
    "Slime 在组内做 GRPO（减均值，可选除标准差）；全过或全挂组标准化后 advantage 为 0，但仍占用采样预算。",
    "格式惩罚、长度惩罚、过程分尚未进入当前训练奖励，只出现在诊断建议里。",
]

REWARD_DESIGN_PLAYBOOK: List[Dict[str, Any]] = [
    {
        "id": "group_variance",
        "title": "组内要有方差",
        "summary": (
            "全过 / 全挂组 advantage 为 0。用难度配比或丢掉零方差组后再更新"
            "（动态采样一类做法），保住混合组。"
        ),
        "problem_ids": ["zero_variance_groups", "pool_saturated", "pool_too_hard"],
    },
    {
        "id": "mask_infra",
        "title": "基建失败不要当负样本",
        "summary": (
            "abort、评测器故障（非法 payload / apply 失败）应 mask，"
            "不要写成 reward 0 与真正的 tests_failed 混在一起。"
        ),
        "problem_ids": ["sdk_or_abort", "evaluator_fault"],
    },
    {
        "id": "timeout_separate",
        "title": "超时和上下文打满单独处理",
        "summary": (
            "截断轨迹直接标 0 会让策略学「别做长任务」。"
            "调预算、mask truncated，或与可测失败分开统计。"
        ),
        "problem_ids": ["context_exhausted", "time_budget_exceeded", "turn_or_iter_limit"],
    },
    {
        "id": "format_and_fake",
        "title": "假响应和格式崩溃要有独立代价",
        "summary": (
            "max_fake_responses、malformed_action、无 tool_call、不以 finish 收尾"
            "应单独惩罚或过滤；伪造工具结果不能靠叙述拿到过程分。"
        ),
        "problem_ids": [
            "max_fake_responses",
            "malformed_action",
            "no_tool_call",
            "last_not_finish",
        ],
    },
    {
        "id": "early_finish_diff",
        "title": "过早 finish / 无 diff",
        "summary": "完成前要有真实改动；对过短轨迹或无 diff 样本降权。",
        "problem_ids": ["early_finish", "no_diff"],
    },
    {
        "id": "length_bias",
        "title": "长度偏差",
        "summary": (
            "GRPO 容易偏向更长轨迹。用超长惩罚或 token 级损失，"
            "而不是奖励「做了很多轮」。看训练曲线 response_length / entropy_loss。"
        ),
        "problem_ids": ["turn_or_iter_limit", "context_exhausted"],
    },
    {
        "id": "pool_health",
        "title": "题目池健康度",
        "summary": (
            "太容易（饱和）和太难都会让组内同号。"
            "沿用 saturated_pool_ratio / hard_pool_ratio，移出两端题。"
        ),
        "problem_ids": ["pool_saturated", "pool_too_hard", "tests_failed"],
    },
    {
        "id": "entropy",
        "title": "熵不要过早塌缩",
        "summary": (
            "训练曲线已有 entropy_loss；持续下降通常意味着策略过早收敛。"
            "本页不编造 KL 数值。"
        ),
        "problem_ids": ["zero_variance_groups"],
    },
]


def list_playbook_entries() -> List[Dict[str, Any]]:
    """返回实践策略列表副本。"""
    return [dict(item) for item in REWARD_DESIGN_PLAYBOOK]
