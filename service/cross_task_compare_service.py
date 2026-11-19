#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""跨任务对比：多个任务在**共同题目**上的训练过程变化趋势。

回答的问题：
    不同训练配置下，训练过程中哪些共同的题目变好了，哪些变坏了？

数据来源只用 attempt 结局行（``DBAttemptService.list_outcome_rows_by_task``），
每个任务内部按「同一题目第 k 次出现」作为第 k 个 epoch —— 与单任务问题分析页
的口径完全一致，因此两个任务之间可以按 epoch 对齐比较。

奖励上升指引部分复用 ``reward_diagnosis_service`` 的
``summarize_gradient_groups`` / ``summarize_reward_split``：
有效梯度组占比与可测通过率是奖励能否继续上升的核心杠杆。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from service.problem_analysis_service import summarize_pass_visits

__all__ = [
    "build_cross_task_compare",
    "build_instance_compare",
    "summarize_reward_guidance",
    "list_compareable_tasks",
]

_RATE_EPS = 1e-9
_MAX_TASKS = 6
_MAX_COMMON_INSTANCES = 3000


# ---------------------------------------------------------------------------
# 任务筛选
# ---------------------------------------------------------------------------

def list_compareable_tasks() -> List[Dict[str, Any]]:
    """列出可用于对比的任务（供页面选择）。"""
    from db.db_task_service import DBTaskService

    try:
        tasks = DBTaskService.list_tasks(limit=500, offset=0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("跨任务对比列出任务失败: {}", exc)
        return []
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        primary_id = int(getattr(task, "primary_id", 0) or 0)
        if primary_id <= 0:
            continue
        rows.append(
            {
                "task_id": primary_id,
                "task_name": str(getattr(task, "task_name", "") or ""),
                "status": str(getattr(task, "status", "") or ""),
                "rollout_count": int(getattr(task, "rollout_count", 0) or 0),
                "attempt_count": int(getattr(task, "attempt_count", 0) or 0),
            }
        )
    return rows


def _load_task_meta(task_id: int) -> Optional[Dict[str, Any]]:
    from db.db_task_service import DBTaskService

    task = DBTaskService.get_by_id(int(task_id))
    if task is None:
        return None
    return {
        "task_id": int(task.primary_id),
        "task_name": str(getattr(task, "task_name", "") or ""),
        "status": str(getattr(task, "status", "") or ""),
        "rollout_root": str(getattr(task, "rollout_root", "") or ""),
        "nanhu_job_id": str(getattr(task, "nanhu_job_id", "") or ""),
    }


def _load_visits(task_id: int) -> Dict[str, List[Dict[str, Any]]]:
    """读某任务的 attempt 结局，按题目收成「出现次序 → 通过率」。

    返回 ``{instance_id: [{"epoch", "rollout_id", "n_pass", "n_attempts", "rate"}]}``。
    """
    from db.db_attempt_service import DBAttemptService

    rows = DBAttemptService.list_outcome_rows_by_task(int(task_id))
    trend = summarize_pass_visits(rows)
    visits: Dict[str, List[Dict[str, Any]]] = {}
    for item in trend.get("tasks") or []:
        instance_id = str(item.get("instance_id") or "")
        if not instance_id:
            continue
        ordered: List[Dict[str, Any]] = []
        for index, visit in enumerate(item.get("visits") or []):
            ordered.append(
                {
                    "epoch": index + 1,
                    "rollout_id": visit.get("rollout_id"),
                    "n_pass": int(visit.get("n_pass") or 0),
                    "n_attempts": int(visit.get("n_attempts") or 0),
                    "rate": visit.get("rate"),
                }
            )
        visits[instance_id] = ordered
    return visits


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def _pooled_rate(visits: Sequence[Dict[str, Any]]) -> Optional[float]:
    """把若干次出现的 attempt 加总求通过率。"""
    n_pass = sum(int(v.get("n_pass") or 0) for v in visits)
    n_attempts = sum(int(v.get("n_attempts") or 0) for v in visits)
    if n_attempts <= 0:
        return None
    return n_pass / n_attempts


def _first_last_rate(visits: Sequence[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if not visits:
        return None, None
    first = visits[0].get("rate")
    last = visits[-1].get("rate")
    return (
        float(first) if isinstance(first, (int, float)) else None,
        float(last) if isinstance(last, (int, float)) else None,
    )


def _epoch_curve(
    per_task_visits: Dict[int, Dict[str, List[Dict[str, Any]]]],
    task_ids: Sequence[int],
    instances: Sequence[str],
) -> List[Dict[str, Any]]:
    """按 epoch 汇总共同题目的通过率（每个任务一条曲线）。"""
    max_epoch = 0
    for task_id in task_ids:
        for instance_id in instances:
            visits = per_task_visits.get(task_id, {}).get(instance_id) or []
            if visits:
                max_epoch = max(max_epoch, len(visits))

    curve: List[Dict[str, Any]] = []
    for epoch in range(1, max_epoch + 1):
        point: Dict[str, Any] = {"epoch": epoch}
        for task_id in task_ids:
            picked: List[Dict[str, Any]] = []
            for instance_id in instances:
                visits = per_task_visits.get(task_id, {}).get(instance_id) or []
                if len(visits) >= epoch:
                    picked.append(visits[epoch - 1])
            point[str(task_id)] = _pooled_rate(picked)
            point[f"{task_id}_n_problems"] = len(picked)
        curve.append(point)
    return curve


def _task_summary(
    task_id: int,
    visits: Dict[str, List[Dict[str, Any]]],
    common: Sequence[str],
) -> Dict[str, Any]:
    """某任务在共同题目上的摘要。"""
    common_visits = [visits.get(iid) or [] for iid in common]
    n_with_data = sum(1 for v in common_visits if v)
    deltas: List[float] = []
    for item in common_visits:
        first, last = _first_last_rate(item)
        if first is not None and last is not None and len(item) >= 2:
            deltas.append(last - first)
    improved = sum(1 for d in deltas if d > _RATE_EPS)
    worsened = sum(1 for d in deltas if d < -_RATE_EPS)
    first_rates = [v[0].get("rate") for v in common_visits if v]
    last_rates = [v[-1].get("rate") for v in common_visits if v]
    first_pooled = _pooled_rate([v[0] for v in common_visits if v])
    last_pooled = _pooled_rate([v[-1] for v in common_visits if v])
    return {
        "task_id": int(task_id),
        "n_common": len(common),
        "n_common_with_data": n_with_data,
        "n_improved": improved,
        "n_worsened": worsened,
        "n_unchanged": max(0, len(deltas) - improved - worsened),
        "first_pass_rate": first_pooled,
        "last_pass_rate": last_pooled,
        "delta": (
            (last_pooled - first_pooled)
            if (first_pooled is not None and last_pooled is not None)
            else None
        ),
        "mean_first_rate": (
            sum(r for r in first_rates if isinstance(r, (int, float)))
            / len([r for r in first_rates if isinstance(r, (int, float))])
            if any(isinstance(r, (int, float)) for r in first_rates)
            else None
        ),
        "mean_last_rate": (
            sum(r for r in last_rates if isinstance(r, (int, float)))
            / len([r for r in last_rates if isinstance(r, (int, float))])
            if any(isinstance(r, (int, float)) for r in last_rates)
            else None
        ),
    }


def _classify_pair_delta(delta: Optional[float]) -> str:
    if delta is None:
        return "unknown"
    if delta > _RATE_EPS:
        return "improved"
    if delta < -_RATE_EPS:
        return "worsened"
    return "unchanged"


def _build_instance_rows(
    per_task_visits: Dict[int, Dict[str, List[Dict[str, Any]]]],
    task_ids: Sequence[int],
    common: Sequence[str],
    baseline_id: int,
) -> List[Dict[str, Any]]:
    """逐题给出各任务的首尾通过率、每次出现的通过数量与相对基线的差值。"""
    rows: List[Dict[str, Any]] = []
    for instance_id in common:
        entry: Dict[str, Any] = {"instance_id": instance_id, "tasks": {}}
        for task_id in task_ids:
            visits = per_task_visits.get(task_id, {}).get(instance_id) or []
            first, last = _first_last_rate(visits)
            entry["tasks"][str(task_id)] = {
                "n_visits": len(visits),
                "first_rate": first,
                "last_rate": last,
                "delta": (
                    (last - first)
                    if (first is not None and last is not None and len(visits) >= 2)
                    else None
                ),
                "rates": [v.get("rate") for v in visits],
                # 每次出现（epoch）的通过数量，供前端展示「n_pass / n_attempts」
                "visits": [
                    {
                        "epoch": int(v.get("epoch") or 0),
                        "rollout_id": v.get("rollout_id"),
                        "n_pass": int(v.get("n_pass") or 0),
                        "n_attempts": int(v.get("n_attempts") or 0),
                        "rate": v.get("rate"),
                    }
                    for v in visits
                ],
            }
        # 相对基线的差值：各任务末次通过率 减 基线末次通过率
        base_last = (entry["tasks"].get(str(baseline_id)) or {}).get("last_rate")
        for task_id in task_ids:
            last = (entry["tasks"].get(str(task_id)) or {}).get("last_rate")
            entry["tasks"][str(task_id)]["vs_baseline"] = (
                (last - base_last)
                if (isinstance(last, (int, float)) and isinstance(base_last, (int, float)))
                else None
            )
        rows.append(entry)
    return rows


def _sort_instance_rows(rows: List[Dict[str, Any]], task_ids: Sequence[int]) -> List[Dict[str, Any]]:
    """按「相对基线变化绝对值」降序，让变好/变坏的题目排在最前。"""
    def _key(row: Dict[str, Any]) -> Tuple[int, float, str]:
        worst = 0.0
        for task_id in task_ids:
            value = (row["tasks"].get(str(task_id)) or {}).get("vs_baseline")
            if isinstance(value, (int, float)) and abs(value) > abs(worst):
                worst = float(value)
        return (0 if abs(worst) > _RATE_EPS else 1, -abs(worst), str(row.get("instance_id")))

    return sorted(rows, key=_key)


# ---------------------------------------------------------------------------
# 奖励上升指引
# ---------------------------------------------------------------------------

def summarize_reward_guidance(task_id: int) -> Dict[str, Any]:
    """给出「奖励还能怎么涨」的指标视角。

    复用奖励诊断的 ``build_reward_levers`` 口径（与单任务奖励诊断页一致）：
    - ``learnable_ratio``：有效梯度组占比，掉下来就没有上升空间；
    - ``measurable_pass_rate``：可测通过率（去掉 abort / timeout 的分母）；
    - ``timeout_ratio`` / ``abort_ratio``：奖励被环境吃掉的比例；
    - ``raw_mean_reward``：训练实际拿到的奖励。
    """
    from db.db_attempt_service import DBAttemptService
    from service.reward_diagnosis_service import build_reward_levers

    rows = DBAttemptService.list_outcome_rows_by_task(int(task_id))
    levers = build_reward_levers(rows)

    return {
        "task_id": int(task_id),
        "learnable_ratio": levers["values"].get("learnable_ratio"),
        "n_mixed": levers["values"].get("n_mixed"),
        "n_groups": levers["values"].get("n_groups"),
        "raw_mean_reward": levers["values"].get("raw_mean_reward"),
        "measurable_pass_rate": levers["values"].get("measurable_pass_rate"),
        "timeout_ratio": levers["values"].get("timeout_ratio"),
        "abort_ratio": levers["values"].get("abort_ratio"),
        "saturated_pool_ratio": levers["values"].get("saturated_pool_ratio"),
        "hard_pool_ratio": levers["values"].get("hard_pool_ratio"),
        "n_attempts": levers["values"].get("n_attempts"),
        "gradient_dips": levers.get("gradient_dips") or [],
        "pool_health": levers.get("pool_health") or {},
        "levers": levers.get("levers") or [],
        "note": (
            "有效梯度组占比（混合组）是奖励能否继续上升的核心杠杆："
            "混合组掉下来，后面再怎么调采样也拿不到新奖励。"
            "可测通过率是策略真实变好的部分；timeout / abort 占比是奖励被环境吃掉的部分。"
            "题目池饱和度（末次通过率贴近 1）与难度天花板（贴近 0）"
            "分别对应「刷穿了没题可学」和「太难学不动」两种天花板。"
        ),
    }


def _reward_guidance_list(
    task_ids: Sequence[int],
    task_meta: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """逐任务给出奖励上升指引卡片。"""
    rows: List[Dict[str, Any]] = []
    for task_id in task_ids:
        meta = task_meta.get(task_id) or {}
        try:
            guidance = summarize_reward_guidance(int(task_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("奖励指引统计失败 task_id={} err={}", task_id, exc)
            continue
        guidance["task_name"] = meta.get("task_name") or ""
        rows.append(guidance)
    return rows


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def build_cross_task_compare(
    task_ids: Sequence[int],
    *,
    instance_id: str = "",
    baseline_task_id: Optional[int] = None,
) -> Dict[str, Any]:
    """构建跨任务对比视图。

    Args:
        task_ids: 参与对比的任务（至少 2 个）。
        instance_id: 只看某一道题时传入；为空则看全部共同题目。
        baseline_task_id: 用户主动选择的**基线任务**；为空时取第一个任务。
            基线决定「相对基线」这一列的参照对象。
    """
    from data.rollout_insight_exception import RolloutInsightException

    cleaned: List[int] = []
    for raw in task_ids or []:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0 and value not in cleaned:
            cleaned.append(value)
    if len(cleaned) < 2:
        raise RolloutInsightException(
            RolloutInsightException.INVALID_ARGUMENT, "跨任务对比至少需要选择 2 个任务"
        )
    if len(cleaned) > _MAX_TASKS:
        raise RolloutInsightException(
            RolloutInsightException.INVALID_ARGUMENT,
            f"跨任务对比最多支持 {_MAX_TASKS} 个任务",
        )

    task_meta: Dict[int, Dict[str, Any]] = {}
    for task_id in cleaned:
        meta = _load_task_meta(task_id)
        if meta is None:
            raise RolloutInsightException(
                RolloutInsightException.TASK_NOT_FOUND, f"task_id={task_id} 不存在"
            )
        task_meta[task_id] = meta

    per_task_visits: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    for task_id in cleaned:
        per_task_visits[task_id] = _load_visits(task_id)

    # 共同题目 = 所有选中任务都出现过的题目
    common_set: Optional[set] = None
    for task_id in cleaned:
        ids = set(per_task_visits[task_id].keys())
        common_set = ids if common_set is None else (common_set & ids)
    common: List[str] = sorted(common_set or set())
    if len(common) > _MAX_COMMON_INSTANCES:
        common = common[:_MAX_COMMON_INSTANCES]

    # 基线任务：用户显式指定优先；指定的不在参与对比的任务里则退回第一个
    baseline_id = cleaned[0]
    if baseline_task_id is not None:
        try:
            wanted = int(baseline_task_id)
        except (TypeError, ValueError):
            wanted = 0
        if wanted in cleaned:
            baseline_id = wanted

    focus = str(instance_id or "").strip()
    focus_rows: List[Dict[str, Any]] = []
    if focus:
        focus_rows = _build_instance_rows(
            per_task_visits, cleaned, [focus], baseline_id
        )

    instance_rows = _build_instance_rows(per_task_visits, cleaned, common, baseline_id)
    instance_rows = _sort_instance_rows(instance_rows, cleaned)

    summaries = {
        str(task_id): _task_summary(task_id, per_task_visits[task_id], common)
        for task_id in cleaned
    }
    curve = _epoch_curve(per_task_visits, cleaned, common)

    # 题目级汇总，两套口径：
    # - buckets：各任务**自身训练轨迹**（首末通过率之差）→ 回答"哪些题变好/变坏"
    # - buckets_vs_baseline：相对基线任务的末次通过率差 → 回答"哪个配置更强"
    bucket: Dict[int, Dict[str, int]] = {
        task_id: {"improved": 0, "worsened": 0, "unchanged": 0, "unknown": 0}
        for task_id in cleaned
    }
    bucket_vs_baseline: Dict[int, Dict[str, int]] = {
        task_id: {"improved": 0, "worsened": 0, "unchanged": 0, "unknown": 0}
        for task_id in cleaned
    }
    for row in instance_rows:
        for task_id in cleaned:
            info = row["tasks"].get(str(task_id)) or {}
            own = _classify_pair_delta(info.get("delta"))
            bucket[task_id][own] = bucket[task_id].get(own, 0) + 1
            vs_base = _classify_pair_delta(info.get("vs_baseline"))
            bucket_vs_baseline[task_id][vs_base] = bucket_vs_baseline[task_id].get(vs_base, 0) + 1

    guidance = _reward_guidance_list(cleaned, task_meta)

    logger.info(
        "跨任务对比 tasks={} common={} baseline={}",
        cleaned,
        len(common),
        baseline_id,
    )
    baseline_name = (task_meta.get(baseline_id) or {}).get("task_name") or ""
    return {
        "tasks": [task_meta[task_id] for task_id in cleaned],
        "baseline_task_id": baseline_id,
        "baseline_task_name": baseline_name,
        "n_common": len(common),
        "common_instances": common,
        "summaries": summaries,
        "epoch_curve": curve,
        "instances": instance_rows,
        "focus_instance": focus or "",
        "focus_rows": focus_rows,
        "buckets": {str(task_id): bucket[task_id] for task_id in cleaned},
        "buckets_vs_baseline": {
            str(task_id): bucket_vs_baseline[task_id] for task_id in cleaned
        },
        "reward_guidance": guidance,
        "available_tasks": list_compareable_tasks(),
        "page_note": (
            "共同题目 = 所有选中任务都出现过的题目；第 k 个 epoch = 该题第 k 次出现。"
            "「变好 / 变坏」按各任务自身训练轨迹（首末通过率之差）统计；"
            "「相对基线」列比较末次通过率与基线任务（页面顶部可切换）之差。"
        ),
    }


def build_instance_compare(
    task_ids: Sequence[int],
    instance_id: str,
    baseline_task_id: Optional[int] = None,
) -> Dict[str, Any]:
    """只看一道题在多个任务上的训练过程变化。"""
    name = str(instance_id or "").strip()
    if not name:
        from data.rollout_insight_exception import RolloutInsightException

        raise RolloutInsightException(
            RolloutInsightException.INVALID_ARGUMENT, "instance_id 不能为空"
        )
    return build_cross_task_compare(
        task_ids,
        instance_id=name,
        baseline_task_id=baseline_task_id,
    )
