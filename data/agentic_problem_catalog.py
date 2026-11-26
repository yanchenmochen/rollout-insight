#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Agentic RL 常见问题目录（静态定义，页面与拼装服务共用）。

每条问题都能从现有数据算出计数；算不到时显示 0 /「无该字段」，不造数。
``source`` 标明主数据来源，便于 ``assemble_problem_cards`` 路由。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = [
    "AGENTIC_PROBLEMS",
    "problem_by_id",
    "list_agentic_problems",
]

# source:
#   agent_outcome  — failure_analysis agent_outcomes 聚类
#   diagnosis      — failure clusters 的 reason 前缀/关键词
#   behavior       — reward_diagnosis 行为卡片
#   category       — attempt category 计数（abort 等）
#   gradient       — 组语义（零方差）
#   pool           — 题目池健康度

AGENTIC_PROBLEMS: List[Dict[str, Any]] = [
    {
        "id": "max_fake_responses",
        "title": "假响应次数超限",
        "why_blocks_reward": (
            "模型在拿伪造或无效的工具反馈自我欺骗，轨迹跑满上限却没有真实进展。"
        ),
        "grpo_meaning": (
            "组内若大量假响应收尾，可测通过率被压低；"
            "若整组同号失败，GRPO 标准化后 advantage 仍可能为 0。"
        ),
        "reward_action": (
            "对假响应单独降权或硬截断；不要让叙述式「工具结果」拿到过程分。"
        ),
        "source": "agent_outcome",
        "match": {"agent_outcome": "max_fake_responses"},
        "failure_query": {"agent": "max_fake_responses"},
        "denominator": "attempts",
        "warn_ratio": 0.02,
        "bad_ratio": 0.05,
    },
    {
        "id": "malformed_action",
        "title": "动作格式非法",
        "why_blocks_reward": "工具调用解析失败，沙箱无法执行，轨迹提前作废。",
        "grpo_meaning": "格式崩溃样本通常不可测或全失败，挤占混合组预算。",
        "reward_action": "加格式惩罚或在进组前过滤非法动作；强化合法 tool_call schema。",
        "source": "agent_outcome",
        "match": {"agent_outcome": "malformed_action"},
        "failure_query": {"agent": "malformed_action"},
        "denominator": "attempts",
        "warn_ratio": 0.02,
        "bad_ratio": 0.05,
    },
    {
        "id": "no_tool_call",
        "title": "无 tool_call",
        "why_blocks_reward": "assistant 轮次没有工具调用，agent 无法改环境。",
        "grpo_meaning": "常见于格式漂移；与可测失败混在一起会稀释真实能力信号。",
        "reward_action": "格式惩罚或丢弃无 tool_call 样本，强化工具调用格式。",
        "source": "behavior",
        "match": {"behavior_key": "no_tool_call"},
        "failure_query": {},
        "denominator": "failed",
        "warn_ratio": 0.08,
        "bad_ratio": 0.15,
    },
    {
        "id": "last_not_finish",
        "title": "最后不是 finish",
        "why_blocks_reward": "轨迹未以 finish 收尾，评测侧可能拿不到完整提交。",
        "grpo_meaning": "未完成轨迹常被标 fail/timeout，拉低组内均值却不一定提供可学对比。",
        "reward_action": "强化完成条件：最后一步应调用 finish。",
        "source": "behavior",
        "match": {"behavior_key": "last_not_finish"},
        "failure_query": {},
        "denominator": "failed",
        "warn_ratio": 0.1,
        "bad_ratio": 0.2,
    },
    {
        "id": "early_finish",
        "title": "过早 finish",
        "why_blocks_reward": "已 finish 但工具步很少且 diff 很小，等于空跑交卷。",
        "grpo_meaning": "假阴性/假阳性混进可测 fail，污染组内相对排序。",
        "reward_action": "收紧完成条件，或对过短轨迹降权 / 惩罚。",
        "source": "behavior",
        "match": {"behavior_key": "early_finish"},
        "failure_query": {},
        "denominator": "failed",
        "warn_ratio": 0.08,
        "bad_ratio": 0.15,
    },
    {
        "id": "context_exhausted",
        "title": "上下文打满",
        "why_blocks_reward": "上下文长度触顶，策略尚未收尾就被截断。",
        "grpo_meaning": "若把截断标成 reward 0，策略会学到「别做长任务」；应 mask 或单独处理。",
        "reward_action": "调上下文长度 / max_turns，或对 truncated 样本 mask；勿当普通 fail。",
        "source": "agent_outcome",
        "match": {"agent_outcome": "context_exhausted"},
        "failure_query": {"agent": "context_exhausted"},
        "denominator": "attempts",
        "warn_ratio": 0.08,
        "bad_ratio": 0.15,
        "tb_key": "context_exhausted",
    },
    {
        "id": "time_budget_exceeded",
        "title": "时间预算耗尽",
        "why_blocks_reward": "单条 attempt 超时，可测奖励变少。",
        "grpo_meaning": "超时占比升高会吃掉可学习样本；与真实能力失败要分开看。",
        "reward_action": "调时间预算或缩短无效探索；超时样本不要直接当负样本。",
        "source": "agent_outcome",
        "match": {"agent_outcome": "time_budget_exceeded"},
        "failure_query": {"agent": "time_budget_exceeded"},
        "denominator": "attempts",
        "warn_ratio": 0.05,
        "bad_ratio": 0.12,
    },
    {
        "id": "turn_or_iter_limit",
        "title": "迭代 / 轮次上限",
        "why_blocks_reward": "打满迭代或轮次上限仍未解决，轨迹预算被耗尽。",
        "grpo_meaning": "长轨迹同号失败会占满组却无梯度；需控制步数或惩罚空转。",
        "reward_action": "限制无效循环；对达上限且无 diff 的样本降权。",
        "source": "agent_outcome_any",
        "match": {
            "agent_outcomes": ("max_iterations_reached", "turn_limit_reached"),
        },
        "failure_query": {},
        "denominator": "attempts",
        "warn_ratio": 0.03,
        "bad_ratio": 0.08,
    },
    {
        "id": "sdk_or_abort",
        "title": "SDK / 基建中止",
        "why_blocks_reward": "沙箱、driver、SDK 报错导致 attempt abort，奖励不可测。",
        "grpo_meaning": "abort 不当作策略负样本；否则 GRPO 在学环境噪声。",
        "reward_action": "先修环境与 harness；abort 样本 mask，不要写成 reward 0。",
        "source": "abort_or_sdk",
        "match": {"agent_outcome": "sdk_error", "category": "abort"},
        "failure_query": {"cat": "abort"},
        "denominator": "attempts",
        "warn_ratio": 0.05,
        "bad_ratio": 0.12,
    },
    {
        "id": "evaluator_fault",
        "title": "评测器故障",
        "why_blocks_reward": (
            "非法 evaluator payload / apply 失败等，评测本身坏了，不是策略答错。"
        ),
        "grpo_meaning": "与 tests_failed 混在一起会夸大策略失败率，零方差组也会被夸大。",
        "reward_action": "评测器故障应 mask；与真正的 tests_failed 分开统计。",
        "source": "diagnosis",
        "match": {
            "reason_substrings": (
                "invalid_evaluator_payload",
                "apply_failed",
                "evaluator:invalid",
                "evaluator:apply_failed",
            ),
        },
        "failure_query": {},
        "denominator": "attempts",
        "warn_ratio": 0.02,
        "bad_ratio": 0.05,
    },
    {
        "id": "tests_failed",
        "title": "评测未通过（tests_failed）",
        "why_blocks_reward": "可测失败：改动提交了但测试没过——这是真实策略信号。",
        "grpo_meaning": "与全过样本同组时提供有效 advantage；全挂组则仍无梯度。",
        "reward_action": "保留为负样本；结合题目难度配比，避免整批全挂。",
        "source": "diagnosis",
        "match": {"reason_substrings": ("tests_failed", "evaluator:tests_failed")},
        "failure_query": {"cat": "fail", "reason": "evaluator:tests_failed"},
        "denominator": "attempts",
        "warn_ratio": 0.35,
        "bad_ratio": 0.6,
    },
    {
        "id": "no_diff",
        "title": "无有效 diff",
        "why_blocks_reward": "agent 结束但没有可应用的代码改动。",
        "grpo_meaning": "空提交常与过早 finish 叠加，组内缺乏可区分轨迹。",
        "reward_action": "完成前要求真实改动；对无 diff 样本降权。",
        "source": "diagnosis",
        "match": {"reason_substrings": ("no_diff", "agent:no_diff")},
        "failure_query": {"reason": "agent:no_diff"},
        "denominator": "attempts",
        "warn_ratio": 0.03,
        "bad_ratio": 0.08,
    },
    {
        "id": "zero_variance_groups",
        "title": "零方差组（全过 / 全挂）",
        "why_blocks_reward": "组内全部通过或全部失败时，GRPO 标准化后 advantage 全为 0。",
        "grpo_meaning": "零梯度但仍占用 rollout 预算；混合组占比下降即学习信号枯竭。",
        "reward_action": (
            "难度配比或丢掉零方差组后再更新（动态采样一类做法）；提高混合组占比。"
        ),
        "source": "gradient",
        "match": {},
        "failure_query": {},
        "denominator": "groups",
        "warn_ratio": 0.5,
        "bad_ratio": 0.75,
    },
    {
        "id": "pool_saturated",
        "title": "题目池饱和",
        "why_blocks_reward": "大量题已被刷穿，组内容易全过，没有对比信号。",
        "grpo_meaning": "饱和题推高 all_pass 组，learnable_ratio 下降。",
        "reward_action": "移出或降采样饱和题，补充「差一点就能过」的题。",
        "source": "pool",
        "match": {"pool_key": "saturated_pool_ratio"},
        "failure_query": {},
        "denominator": "problems",
        "warn_ratio": 0.35,
        "bad_ratio": 0.6,
    },
    {
        "id": "pool_too_hard",
        "title": "题目池过难",
        "why_blocks_reward": "大量题一直全挂，组内同号失败，同样没有学习信号。",
        "grpo_meaning": "推高 all_fail 组；与饱和对称的另一端枯竭。",
        "reward_action": "降采样全挂题或先补中间难度；结合失败分析区分能力与行为缺陷。",
        "source": "pool",
        "match": {"pool_key": "hard_pool_ratio"},
        "failure_query": {},
        "denominator": "problems",
        "warn_ratio": 0.35,
        "bad_ratio": 0.6,
    },
]


def list_agentic_problems() -> List[Dict[str, Any]]:
    """返回目录副本（只读场景可直接迭代 ``AGENTIC_PROBLEMS``）。"""
    return [dict(item) for item in AGENTIC_PROBLEMS]


def problem_by_id(problem_id: str) -> Optional[Dict[str, Any]]:
    """按 id 取一条目录定义。"""
    key = str(problem_id or "").strip()
    for item in AGENTIC_PROBLEMS:
        if item["id"] == key:
            return dict(item)
    return None
