#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""从 .pt sample.metadata 抽取失败 / 超时 / abort 深挖洞察并聚类。

数据仅来自 dump 内字段（evaluation.stdout_tail、driver_error 等），
每条结果附带 ``sources`` 注明出处，便于 UI 展示「信息来源」。
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from utils.rollout_pt_parser import classify_sample_outcome

__all__ = [
    "empty_failure_insights",
    "extract_sample_failure_detail",
    "cluster_failure_insights",
    "parse_pytest_failure",
]

_SNIPPET_MAX = 1200
_SUMMARY_MAX = 180
_EXAMPLES_PER_CLUSTER = 3

_FAILURES_TEST_RE = re.compile(
    r"_{3,}\s+([^\n_][^\n]*?)\s+_{3,}",
)
_PYTEST_E_RE = re.compile(r"^E\s+(.+)$", re.MULTILINE)
_ASSERT_RE = re.compile(r"^\s*(assert\s.+)$", re.MULTILINE)
_DIGIT_RE = re.compile(r"\d+")


def empty_failure_insights() -> Dict[str, Any]:
    """空的失败深挖结构（fail / timeout / abort clusters + by_group）。"""
    return {
        "n_fail": 0,
        "n_timeout": 0,
        "n_abort": 0,
        "fail": {"n_samples": 0, "clusters": []},
        "timeout": {"n_samples": 0, "clusters": []},
        "abort": {"n_samples": 0, "clusters": []},
        "by_group": {},
    }


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _source(field: str, path: str) -> Dict[str, str]:
    return {"field": field, "path": path}


def _sample_attempt_id(sample: Dict[str, Any], index: int) -> Optional[int]:
    aid = sample.get("rollout_id")
    if aid is None:
        aid = index
    try:
        return int(aid)
    except (TypeError, ValueError):
        return None


def _parse_pytest_failure(stdout_tail: str) -> Tuple[Optional[str], Optional[str]]:
    """从 pytest stdout_tail 抽首个失败用例名与错误摘要。"""
    text = stdout_tail or ""
    failed_test: Optional[str] = None
    m = _FAILURES_TEST_RE.search(text)
    if m:
        failed_test = m.group(1).strip() or None

    error_summary: Optional[str] = None
    e_lines = _PYTEST_E_RE.findall(text)
    if e_lines:
        error_summary = e_lines[0].strip()
    else:
        asserts = _ASSERT_RE.findall(text)
        if asserts:
            error_summary = asserts[0].strip()
    return failed_test, error_summary


def parse_pytest_failure(stdout_tail: str) -> Tuple[Optional[str], Optional[str]]:
    """从 pytest stdout_tail 抽 ``(首个失败用例, 错误摘要)``；无法识别时为 None。"""
    return _parse_pytest_failure(stdout_tail)


def _normalize_driver_error(driver_error: str) -> str:
    return _DIGIT_RE.sub("N", (driver_error or "").strip())


def extract_sample_failure_detail(
    sample: Dict[str, Any], *, sample_index: int = 0
) -> Optional[Dict[str, Any]]:
    """抽取单条非通过 sample 的深挖细节；通过样本返回 None。"""
    if not isinstance(sample, dict):
        return None
    outcome = classify_sample_outcome(sample)
    cat = str(outcome.get("category") or "")
    if cat not in {"fail", "timeout", "abort"}:
        return None

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    gi = sample.get("group_index")
    try:
        group_index = int(gi) if gi is not None else None
    except (TypeError, ValueError):
        group_index = None
    attempt_id = _sample_attempt_id(sample, sample_index)
    instance_id = str(metadata.get("instance_id") or sample.get("label") or "").strip()
    sandbox_id = str(metadata.get("sandbox_id") or "").strip()
    reason = str(outcome.get("reason") or "").strip()

    base = {
        "category": cat,
        "reason": reason,
        "label": str(outcome.get("label") or f"{cat}:{reason}"),
        "group_index": group_index,
        "attempt_id": attempt_id,
        "instance_id": instance_id,
        "sandbox_id": sandbox_id,
        "sources": [],
        "snippet": "",
        "failed_test": None,
        "error_summary": None,
        "driver_error": None,
        "driver_error_norm": None,
        "context_prompt_tokens": None,
        "context_limit_tokens": None,
        "attempt_peak_context_tokens": None,
        "attempt_completed_turns": None,
        "abort_reason": None,
        "note": None,
    }

    if cat == "fail":
        diagnosis = str(metadata.get("diagnosis") or metadata.get("failure_class") or reason)
        evaluation = (
            metadata.get("evaluation") if isinstance(metadata.get("evaluation"), dict) else {}
        )
        stdout_tail = str(evaluation.get("stdout_tail") or "")
        stderr_tail = str(evaluation.get("stderr_tail") or "")
        failed_test, error_summary = _parse_pytest_failure(stdout_tail)
        sources: List[Dict[str, str]] = [
            _source("diagnosis", "sample.metadata.diagnosis"),
        ]
        if stdout_tail:
            sources.append(
                _source("stdout_tail", "sample.metadata.evaluation.stdout_tail")
            )
        if stderr_tail:
            sources.append(
                _source("stderr_tail", "sample.metadata.evaluation.stderr_tail")
            )
        if evaluation.get("exit_code") is not None:
            sources.append(
                _source("exit_code", "sample.metadata.evaluation.exit_code")
            )
        if not stdout_tail and not stderr_tail:
            base["note"] = "dump 未含 evaluation.stdout_tail / stderr_tail，仅有 diagnosis"
        snippet_parts = []
        if stdout_tail:
            snippet_parts.append(stdout_tail)
        if stderr_tail:
            snippet_parts.append(f"[stderr]\n{stderr_tail}")
        base.update(
            {
                "reason": diagnosis or reason,
                "failed_test": failed_test,
                "error_summary": _truncate(error_summary or "", _SUMMARY_MAX) or None,
                "snippet": _truncate("\n".join(snippet_parts), _SNIPPET_MAX),
                "sources": sources,
            }
        )
        return base

    if cat == "timeout":
        driver_error = str(metadata.get("driver_error") or "").strip()
        conv = str(metadata.get("conversation_error_code") or "").strip()
        sources = [
            _source(
                "conversation_error_code",
                "sample.metadata.conversation_error_code",
            ),
        ]
        if driver_error:
            sources.append(_source("driver_error", "sample.metadata.driver_error"))
        for fld in (
            "context_prompt_tokens",
            "context_limit_tokens",
            "attempt_peak_context_tokens",
            "attempt_completed_turns",
            "train_truncation_reason",
        ):
            if metadata.get(fld) is not None:
                sources.append(_source(fld, f"sample.metadata.{fld}"))
        if not driver_error:
            base["note"] = "dump 未含 driver_error，仅有 conversation_error_code"
        base.update(
            {
                "reason": conv or reason,
                "driver_error": driver_error or None,
                "driver_error_norm": _normalize_driver_error(driver_error)
                if driver_error
                else (conv or reason or "(unknown)"),
                "error_summary": _truncate(driver_error or conv or reason, _SUMMARY_MAX)
                or None,
                "context_prompt_tokens": metadata.get("context_prompt_tokens"),
                "context_limit_tokens": metadata.get("context_limit_tokens"),
                "attempt_peak_context_tokens": metadata.get(
                    "attempt_peak_context_tokens"
                ),
                "attempt_completed_turns": metadata.get("attempt_completed_turns"),
                "snippet": _truncate(driver_error or conv or reason, _SNIPPET_MAX),
                "sources": sources,
            }
        )
        return base

    # abort
    abort_reason = str(metadata.get("abort_reason") or reason or "").strip()
    sources = [_source("abort_reason", "sample.metadata.abort_reason")]
    phase_bits = []
    for fld in (
        "phase_time/sandbox_boot",
        "phase_time/attempt_total",
        "train_drop_reason",
    ):
        if metadata.get(fld) is not None:
            sources.append(_source(fld, f"sample.metadata.{fld}"))
            phase_bits.append(f"{fld}={metadata.get(fld)}")
    base.update(
        {
            "abort_reason": abort_reason or None,
            "error_summary": abort_reason or None,
            "snippet": _truncate(
                "\n".join([abort_reason] + phase_bits[:3]), _SNIPPET_MAX
            ),
            "sources": sources,
            "note": "dump 未含 exception traceback，仅有 abort_reason",
        }
    )
    return base


def _cluster_key_fail(detail: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(detail.get("reason") or ""),
            str(detail.get("failed_test") or "(no_test)"),
            str(detail.get("error_summary") or "(no_error)"),
        ]
    )


def _cluster_key_timeout(detail: Dict[str, Any]) -> str:
    return str(
        detail.get("driver_error_norm")
        or detail.get("reason")
        or "(unknown)"
    )


def _cluster_key_abort(detail: Dict[str, Any]) -> str:
    return str(detail.get("abort_reason") or detail.get("reason") or "(unknown)")


def _example_from(detail: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "group_index": detail.get("group_index"),
        "attempt_id": detail.get("attempt_id"),
        "instance_id": detail.get("instance_id") or "",
        "sandbox_id": detail.get("sandbox_id") or "",
    }


def _build_category_clusters(
    details: List[Dict[str, Any]], category: str
) -> Dict[str, Any]:
    if not details:
        return {"n_samples": 0, "clusters": []}

    key_fn = {
        "fail": _cluster_key_fail,
        "timeout": _cluster_key_timeout,
        "abort": _cluster_key_abort,
    }[category]

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in details:
        buckets[key_fn(d)].append(d)

    total = len(details)
    clusters: List[Dict[str, Any]] = []
    for key, items in sorted(
        buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        head = items[0]
        examples = [_example_from(x) for x in items[:_EXAMPLES_PER_CLUSTER]]
        # 合并 sources（按 path 去重）
        seen_paths = set()
        sources: List[Dict[str, str]] = []
        for it in items:
            for src in it.get("sources") or []:
                path = src.get("path")
                if path and path not in seen_paths:
                    seen_paths.add(path)
                    sources.append(src)

        prompt_vals = [
            x.get("context_prompt_tokens")
            for x in items
            if isinstance(x.get("context_prompt_tokens"), (int, float))
        ]
        limit_vals = [
            x.get("context_limit_tokens")
            for x in items
            if isinstance(x.get("context_limit_tokens"), (int, float))
        ]
        turns_vals = [
            x.get("attempt_completed_turns")
            for x in items
            if isinstance(x.get("attempt_completed_turns"), (int, float))
        ]

        cluster: Dict[str, Any] = {
            "key": key,
            "category": category,
            "reason": head.get("reason"),
            "failed_test": head.get("failed_test"),
            "error_summary": head.get("error_summary"),
            "driver_error": head.get("driver_error"),
            "abort_reason": head.get("abort_reason"),
            "count": len(items),
            "pct": round(100.0 * len(items) / total, 1) if total else 0.0,
            "examples": examples,
            "snippet": head.get("snippet") or "",
            "sources": sources,
            "note": head.get("note"),
        }
        if category == "timeout":
            cluster["context_prompt_tokens_min"] = (
                int(min(prompt_vals)) if prompt_vals else None
            )
            cluster["context_prompt_tokens_max"] = (
                int(max(prompt_vals)) if prompt_vals else None
            )
            cluster["context_limit_tokens"] = (
                int(limit_vals[0]) if limit_vals else None
            )
            cluster["attempt_completed_turns_min"] = (
                int(min(turns_vals)) if turns_vals else None
            )
            cluster["attempt_completed_turns_max"] = (
                int(max(turns_vals)) if turns_vals else None
            )
        clusters.append(cluster)

    return {"n_samples": total, "clusters": clusters}


def _insights_from_details(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_cat: Dict[str, List[Dict[str, Any]]] = {
        "fail": [],
        "timeout": [],
        "abort": [],
    }
    for d in details:
        cat = d.get("category")
        if cat in by_cat:
            by_cat[cat].append(d)
    return {
        "n_fail": len(by_cat["fail"]),
        "n_timeout": len(by_cat["timeout"]),
        "n_abort": len(by_cat["abort"]),
        "fail": _build_category_clusters(by_cat["fail"], "fail"),
        "timeout": _build_category_clusters(by_cat["timeout"], "timeout"),
        "abort": _build_category_clusters(by_cat["abort"], "abort"),
    }


def cluster_failure_insights(
    dump: Dict[str, Any], *, group_index: Optional[int] = None
) -> Dict[str, Any]:
    """跨组 / 组内失败深挖聚类。

    Returns:
        {
          n_fail, n_timeout, n_abort,
          fail: {n_samples, clusters:[...]},
          timeout: {...}, abort: {...},
          by_group: {gi: 同上不含 by_group}
        }
    """
    empty = empty_failure_insights()
    samples = dump.get("samples") if isinstance(dump, dict) else None
    if not isinstance(samples, list):
        return empty

    all_details: List[Dict[str, Any]] = []
    per_group: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        gi = sample.get("group_index")
        if group_index is not None:
            try:
                if int(gi) != int(group_index):
                    continue
            except (TypeError, ValueError):
                continue
        detail = extract_sample_failure_detail(sample, sample_index=i)
        if not detail:
            continue
        all_details.append(detail)
        if isinstance(detail.get("group_index"), int):
            per_group[int(detail["group_index"])].append(detail)

    result = _insights_from_details(all_details)
    by_group: Dict[int, Dict[str, Any]] = {}
    for gi in sorted(per_group.keys()):
        by_group[gi] = _insights_from_details(per_group[gi])
    result["by_group"] = by_group
    return result
