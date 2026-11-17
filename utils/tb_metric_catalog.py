"""TensorBoard scalar tag 说明。

只收录代码里能对上的评测 tag，其余一律「未收录」。
不给 loss / kl 等训练曲线编造含义。
"""
from __future__ import annotations

from utils.eval_tb_scores import is_primary_eval_tag

__all__ = [
    "DESC_ATTEMPT_COUNT",
    "DESC_PRIMARY",
    "DESC_TRUNCATED",
    "DESC_UNKNOWN",
    "classify_tb_tag",
    "describe_tb_tag",
]

DESC_PRIMARY = "评测主分数（attempt 平均 reward）"
DESC_TRUNCATED = "截断比例，不是主分数"
DESC_ATTEMPT_COUNT = "尝试次数"
DESC_UNKNOWN = "未收录"

_KIND_PRIMARY = "primary"
_KIND_TRUNCATED = "truncated_ratio"
_KIND_ATTEMPT_COUNT = "attempt_count"
_KIND_UNKNOWN = "unknown"


def classify_tb_tag(tag: str) -> str:
    """返回 ``primary`` / ``truncated_ratio`` / ``attempt_count`` / ``unknown``。

    更具体的子指标先于主分数判断，避免 ``eval/attempt_count`` 被当成主分数。
    """
    text = str(tag or "").strip()
    if text.startswith("eval/") and "-truncated_ratio" in text:
        return _KIND_TRUNCATED
    if text.startswith("eval/") and "attempt_count" in text:
        return _KIND_ATTEMPT_COUNT
    if is_primary_eval_tag(text):
        return _KIND_PRIMARY
    return _KIND_UNKNOWN


def describe_tb_tag(tag: str) -> str:
    """已知评测 tag 的中文说明；对不上的返回「未收录」。"""
    kind = classify_tb_tag(tag)
    if kind == _KIND_TRUNCATED:
        return DESC_TRUNCATED
    if kind == _KIND_ATTEMPT_COUNT:
        return DESC_ATTEMPT_COUNT
    if kind == _KIND_PRIMARY:
        return DESC_PRIMARY
    return DESC_UNKNOWN
