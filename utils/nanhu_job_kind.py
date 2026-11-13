"""南湖作业算法库分类（slime / 评测 / 转换 / 其它）。

用于作业列表 UI：仅 slime 开放「链接任务 / 快速创建」；评测与转换只展示信息。
"""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional

__all__ = [
    "SLIME_ALGORITHM_UUIDS",
    "CONVERT_ALGORITHM_UUIDS",
    "EVAL_ALGORITHM_UUIDS",
    "classify_nanhu_job_kind",
    "attach_nanhu_job_kind",
]

# slime 训练算法（rollout-insight 接入）
SLIME_ALGORITHM_UUIDS = frozenset(
    {
        "9ebb43be-7e49-411e-9e39-fb23bea6f3ce",
    }
)

# 权重转换算法
CONVERT_ALGORITHM_UUIDS = frozenset(
    {
        "6575725e-b4f5-47ed-a33e-fb38aa3b4489",
    }
)

# 评测算法（TransEval / 通用 eval）
EVAL_ALGORITHM_UUIDS = frozenset(
    {
        "45856236-51fb-4987-a844-10a991552457",
        "8b7fcc39-c3d2-4894-a003-5110854eb3aa",
    }
)

_KIND_LABELS = {
    "slime": "slime",
    "convert": "转换",
    "eval": "评测",
    "other": "其它",
}


def _norm_uuid(value: Any) -> str:
    return str(value or "").strip().lower()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def classify_nanhu_job_kind(job: Optional[Mapping[str, Any]]) -> dict:
    """按 algorithm_uuid / 路径 / 名称判定算法库。

    优先级：slime → convert → eval → other。

    返回 ``{"algo_kind": ..., "algo_label": ...}``。
    """
    if not isinstance(job, Mapping):
        return {"algo_kind": "other", "algo_label": _KIND_LABELS["other"]}

    algo = _norm_uuid(
        job.get("algorithm_uuid")
        or job.get("algorithmUuid")
        or job.get("algorithm")
    )
    name = _as_text(job.get("name") or "").strip()
    output = _as_text(
        job.get("output_dir")
        or job.get("outputDir")
        or job.get("output")
        or ""
    )
    # outputDir 可能是 dict；规整后仍可能被 str() 落库，启发式对字符串即可
    haystack = f"{output} {name}"

    if algo in SLIME_ALGORITHM_UUIDS or "/slime/" in haystack or "output/slime" in haystack:
        kind = "slime"
    elif algo in CONVERT_ALGORITHM_UUIDS or name.lower().startswith("convert-"):
        kind = "convert"
    elif (
        algo in EVAL_ALGORITHM_UUIDS
        or "TransEval" in haystack
        or "/Eval/" in haystack
    ):
        kind = "eval"
    else:
        kind = "other"

    return {"algo_kind": kind, "algo_label": _KIND_LABELS[kind]}


def attach_nanhu_job_kind(job: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """就地写入 ``algo_kind`` / ``algo_label``，并返回同一 dict。"""
    job.update(classify_nanhu_job_kind(job))
    return job
