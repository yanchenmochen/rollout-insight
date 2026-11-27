"""任务详情：按南湖/本地 env 判定作业类型，评测作业附加 TB 评分。"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from loguru import logger

from utils.eval_tb_scores import read_eval_scores_for_rollout_root
from utils.slime_job_type import (
    classify_slime_job_type,
    load_env_from_run_dir,
    normalize_env_map,
)
from utils.tb_run_paths import resolve_run_dir

__all__ = [
    "build_task_job_context",
    "load_merged_task_env",
]


def _merge_env(*parts: Mapping[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in parts:
        if part:
            out.update(part)
    return out


def _load_env_from_nanhu(
    nanhu_job_id: str,
    *,
    cluster: str = "",
    local_user_id: Optional[int] = None,
) -> Dict[str, str]:
    job_id = (nanhu_job_id or "").strip()
    if not job_id:
        return {}

    cluster_name = (cluster or "").strip()
    if not cluster_name:
        try:
            from db.db_nanhu_job_service import DBNanhuJobService

            row = DBNanhuJobService.get_by_job_id(job_id)
            if row is not None:
                cluster_name = (getattr(row, "cluster", None) or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("查 nanhu job 集群失败 job_id={}: {}", job_id, exc)
    if not cluster_name:
        cluster_name = "x10000"

    try:
        from service import nanhu_client
        from utils.nanhu_job_path_resolve import env_map_from_job, extract_job_payload
    except Exception as exc:  # noqa: BLE001
        logger.warning("导入 nanhu_client 失败: {}", exc)
        return {}

    try:
        detail = nanhu_client.get_job_detail(
            cluster=cluster_name,
            job_id=job_id,
            local_user_id=local_user_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "get_job_detail 失败 job_id={} cluster={}: {}",
            job_id,
            cluster_name,
            exc,
        )
        return {}

    if not isinstance(detail, dict):
        return {}
    payload = extract_job_payload(detail)
    if payload is None:
        return {}
    return normalize_env_map(env_map_from_job(payload))


def load_merged_task_env(
    *,
    rollout_root: Optional[str] = None,
    nanhu_job_id: Optional[str] = None,
    cluster: str = "",
    local_user_id: Optional[int] = None,
    fetch_nanhu_env: bool = True,
) -> Dict[str, Any]:
    """合并本地 run 目录与南湖作业的环境变量。

    返回 ``env`` / ``launcher`` / ``env_source`` / ``run_dir`` / 作业类型字段。
    供任务详情与环境变量对比共用，避免平行实现。
    """
    run_dir = resolve_run_dir(rollout_root)
    local_env, launcher, env_source = load_env_from_run_dir(run_dir)

    nanhu_env: Dict[str, str] = {}
    if fetch_nanhu_env and (nanhu_job_id or "").strip():
        # 本地已有 TRAINING_SCRIPT/launcher 足够判评测时可跳过南湖，加速详情页
        typed = classify_slime_job_type(local_env, launcher=launcher)
        need_nanhu = typed["job_type"] in ("unknown",) or not (
            launcher
            or local_env.get("TRAINING_SCRIPT")
            or local_env.get("DEBUG_ROLLOUT_ONLY")
        )
        if need_nanhu:
            nanhu_env = _load_env_from_nanhu(
                str(nanhu_job_id),
                cluster=cluster,
                local_user_id=local_user_id,
            )
            if nanhu_env and not env_source:
                env_source = "nanhu_job_detail"
            elif nanhu_env:
                env_source = f"{env_source}+nanhu" if env_source else "nanhu_job_detail"

    env = _merge_env(local_env, nanhu_env)
    typed = classify_slime_job_type(env, launcher=launcher)
    return {
        "env": env,
        "launcher": launcher or "",
        "env_source": env_source or "",
        "run_dir": str(run_dir) if run_dir else None,
        "job_type": typed["job_type"],
        "job_type_label": typed["job_type_label"],
        "job_type_signals": typed.get("signals") or [],
        "training_script": typed.get("training_script") or "",
    }


def build_task_job_context(
    *,
    rollout_root: Optional[str] = None,
    nanhu_job_id: Optional[str] = None,
    cluster: str = "",
    local_user_id: Optional[int] = None,
    fetch_nanhu_env: bool = True,
    include_env_summary: bool = True,
) -> Dict[str, Any]:
    """组装任务作业上下文：类型 +（评测时）评分 + 可选 env 摘要。

    env 来源优先本地 ``resolved_config.yaml``，不足时再拉南湖详情。
    """
    merged = load_merged_task_env(
        rollout_root=rollout_root,
        nanhu_job_id=nanhu_job_id,
        cluster=cluster,
        local_user_id=local_user_id,
        fetch_nanhu_env=fetch_nanhu_env,
    )
    typed_job = merged["job_type"]
    run_dir = merged.get("run_dir")

    scores: list = []
    event_dir = None
    score_run_dir = None
    if typed_job == "eval":
        tb = read_eval_scores_for_rollout_root(rollout_root)
        scores = tb.get("scores") or []
        event_dir = tb.get("event_dir")
        score_run_dir = tb.get("run_dir")

    out: Dict[str, Any] = {
        "job_type": typed_job,
        "job_type_label": merged["job_type_label"],
        "job_type_signals": merged.get("job_type_signals") or [],
        "training_script": merged.get("training_script") or "",
        "env_source": merged.get("env_source") or "",
        "eval_scores": scores,
        "eval_event_dir": event_dir,
        "eval_run_dir": score_run_dir or run_dir,
        "is_eval_job": typed_job == "eval",
        "run_dir": run_dir,
    }
    if include_env_summary:
        from utils.task_env_catalog import summarize_env_map

        summary = summarize_env_map(merged.get("env") or {})
        out["env_count"] = summary["env_count"]
        out["env_highlight"] = summary["highlight"]
        out["env_highlight_catalog"] = summary["highlight_catalog"]
        out["env_by_category"] = summary["by_category"]
    return out
