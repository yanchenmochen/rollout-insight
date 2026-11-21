#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""失败原因归一化：把 artifacts summary 与 .pt 注解的不同字段收敛到同一套分类。

背景
----
同一个失败在两处来源里字段名不同：

- ``attempt_artifacts/summaries/*.json``：``status`` / ``diagnosis`` /
  ``abort_reason`` / ``agent_outcome``
- ``.pt`` sample.metadata：``status`` / ``diagnosis`` / ``abort_reason`` /
  ``conversation_error_code`` / ``agent_outcome``

本模块提供统一的 ``(category, reason)`` 判定与中文标签，使「失败分析」页、
任务详情页「类型柱」与 rollout 详情页「组结果」三处口径一致。

分类优先级
----------
1. **abort**：基础设施 / agent 侧硬中止（``status`` 属 abort 集合，或有 ``abort_reason``）
2. **timeout**：上下文打满 / 超时截断（``status`` 属 timeout 集合，或有超时类错误码）
3. **pass**：可测且 reward≈1
4. **fail**：可测但 reward≈0（评测失败 / agent 未产出有效改动）
5. **other**：其余（不可测且无明确原因）
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

__all__ = [
    "CATEGORY_LABELS",
    "AGENT_OUTCOME_LABELS",
    "AGENT_OUTCOME_REASON_CODES",
    "normalize_reason",
    "classify_from_fields",
    "reason_label",
    "agent_outcome_label",
    "agent_outcome_reason",
    "is_agent_error_outcome",
    "split_reason_prefix",
]

CATEGORY_LABELS: Dict[str, str] = {
    "pass": "通过",
    "fail": "正常失败",
    "timeout": "超时/截断",
    "abort": "abort",
    "other": "其它",
}

#: agent 结束态 → 中文说明（agent 侧异常的统一口径）
AGENT_OUTCOME_LABELS: Dict[str, str] = {
    "finished": "正常结束",
    "context_exhausted": "上下文打满",
    "time_budget_exceeded": "时间预算耗尽",
    "sdk_error": "SDK 报错",
    "max_iterations_reached": "迭代次数超限",
    "max_fake_responses": "假响应次数超限",
    "turn_limit_reached": "轮次上限",
    "malformed_action": "动作格式非法",
}

# abort / 不可测状态集合（与 db/db_attempt_service、utils/rollout_pt_parser 对齐）
_ABORT_STATUSES = frozenset(
    {"aborted", "abort", "cancelled", "canceled", "invalid", "failed_infra"}
)
_TIMEOUT_STATUSES = frozenset({"truncated", "timeout"})

# 超时类错误码关键词
_TIMEOUT_TOKENS = (
    "timeout",
    "context_limit",
    "context_exhausted",
    "timed_out",
    "time_budget",
)

#: agent 结束态 → .pt 侧同名原因码。
#:
#: .pt 侧超时原因是 ``conversation_error_code``（如 ``slime_context_limit_reached``），
#: 而 artifacts summary 只有 ``agent_outcome``（如 ``context_exhausted``）；
#: 这里做一次映射，保证两处口径一致，页面之间的链接才能互相对上。
AGENT_OUTCOME_REASON_CODES: Dict[str, str] = {
    "context_exhausted": "slime_context_limit_reached",
    "time_budget_exceeded": "agent_time_budget_exceeded",
    "sdk_error": "agent:sdk_error",
    "max_iterations_reached": "agent:max_iterations_reached",
    "max_fake_responses": "agent:max_fake_responses",
    "turn_limit_reached": "agent:turn_limit_reached",
    "malformed_action": "agent:malformed_action",
}


def _status_token(status: Any) -> str:
    return str(status or "").split(".")[-1].lower()


def _clean(value: Any) -> str:
    text = str(value or "").strip()
    return text


def agent_outcome_label(agent_outcome: Any) -> str:
    """agent 结束态的中文标签；未知值原样返回。"""
    key = _clean(agent_outcome)
    if not key:
        return ""
    return AGENT_OUTCOME_LABELS.get(key, key)


def is_agent_error_outcome(agent_outcome: Any) -> bool:
    """agent 结束态是否为异常（非 ``finished`` / 空）。"""
    key = _clean(agent_outcome)
    if not key or key == "finished":
        return False
    return True


def _reason_from_diagnosis(diagnosis: str) -> str:
    """``diagnosis`` 形如 ``evaluator:tests_failed`` / ``agent:no_diff``。"""
    return diagnosis or "unknown"


def agent_outcome_reason(agent_outcome: Any) -> str:
    """agent 结束态 → 统一原因码（与 .pt 侧口径一致）。"""
    key = _clean(agent_outcome)
    if not key:
        return ""
    return AGENT_OUTCOME_REASON_CODES.get(key, f"agent:{key}")


def normalize_reason(
    *,
    status: Any = None,
    reward: Any = None,
    diagnosis: Any = None,
    abort_reason: Any = None,
    error_code: Any = None,
    agent_outcome: Any = None,
) -> Tuple[str, str]:
    """把散落的失败字段收敛为 ``(category, reason)``。

    ``reason`` 保留原始可读文本（``evaluator:tests_failed`` /
    ``slime_context_limit_reached`` / ``exception:SandboxApiException`` 等），
    便于聚类与跳转；``category`` 决定 UI 分组与配色。
    """
    status_token = _status_token(status)
    diag = _clean(diagnosis)
    abort = _clean(abort_reason)
    err = _clean(error_code)
    agent = _clean(agent_outcome)

    # 1. abort：基础设施 / agent 侧硬中止
    if abort:
        return "abort", abort
    if status_token in _ABORT_STATUSES:
        return "abort", f"status:{status_token or 'aborted'}"
    # agent 侧硬失败（SDK 报错等）无可测 reward，同样归 abort
    if agent == "sdk_error":
        return "abort", agent_outcome_reason(agent)

    # 2. timeout：上下文打满 / 截断
    if status_token in _TIMEOUT_STATUSES:
        return "timeout", err or agent_outcome_reason(agent) or diag or f"status:{status_token}"
    lowered = f"{err} {diag}".lower()
    if any(tok in lowered for tok in _TIMEOUT_TOKENS):
        return "timeout", err or agent_outcome_reason(agent) or diag or "timeout"

    # 3. pass / fail：按可测 reward 判定
    if isinstance(reward, (int, float)):
        rv = float(reward)
        if abs(rv - 1.0) <= 1e-9:
            return "pass", diag or "resolved"
        if abs(rv) <= 1e-9:
            return "fail", _reason_from_diagnosis(diag)

    # 4. 其它：无可测 reward
    if diag:
        return "other", diag
    if agent:
        return "other", agent_outcome_reason(agent)
    return "other", f"status:{status_token or 'unknown'}"


def classify_from_fields(fields: Dict[str, Any]) -> Tuple[str, str]:
    """从 summary / annotation 字典收敛出 ``(category, reason)``。"""
    return normalize_reason(
        status=fields.get("status"),
        reward=fields.get("reward"),
        diagnosis=fields.get("diagnosis"),
        abort_reason=fields.get("abort_reason"),
        error_code=fields.get("conversation_error_code")
        or fields.get("error_code"),
        agent_outcome=fields.get("agent_outcome"),
    )


def reason_label(category: str, reason: str) -> str:
    """``(category, reason)`` → 中文短标签。"""
    cat = _clean(category)
    rsn = _clean(reason)
    if not rsn:
        return CATEGORY_LABELS.get(cat, cat or "未知")
    return rsn


def split_reason_prefix(reason: str) -> Tuple[str, str]:
    """把 ``evaluator:tests_failed`` 拆成 ``("evaluator", "tests_failed")``。"""
    text = _clean(reason)
    if ":" not in text:
        return "", text
    head, tail = text.split(":", 1)
    return head, tail
