#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""任务详情页 .pt 注解加载：内存缓存 + sidecar + 单 rollout API 载荷。

读路径：sidecar（与 .pt 同目录 ``{pt}.ri_ann.json``）→ 进程内缓存 → ``load_pt``。
成功解析后写 sidecar，供进程重启后秒开类型柱。
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

__all__ = [
    "SIDECAR_SUFFIX",
    "empty_annotations",
    "load_pt_annotations",
    "warm_pt_annotation_sidecar",
    "sidecar_path_for",
    "list_task_annotation_meta",
    "get_rollout_annotation_payload",
    "assign_stack_type_colors",
]

SIDECAR_SUFFIX = ".ri_ann.json"
_SIDECAR_VERSION = 1

_PT_ANNOTATION_CACHE: Dict[str, Dict[str, Any]] = {}

_STACK_PASS_COLOR = "#2e7d32"
_STACK_OTHER_COLOR = "#546e7a"
_STACK_FAIL_PALETTE = (
    "#6a1b9a",
    "#8e24aa",
    "#ab47bc",
    "#ce93d8",
    "#4a148c",
    "#9c27b0",
    "#7b1fa2",
    "#ba68c8",
)
_STACK_TIMEOUT_RW1_PALETTE = (
    "#ef6c00",
    "#ff9800",
    "#f57c00",
    "#ffa726",
    "#fb8c00",
)
_STACK_TIMEOUT_RW0_PALETTE = (
    "#8d6e63",
    "#a1887f",
    "#6d4c41",
    "#bcaaa4",
    "#5d4037",
)
_STACK_TIMEOUT_NONE_PALETTE = ("#90a4ae", "#78909c", "#607d8b")
_STACK_ABORT_PALETTE = (
    "#c62828",
    "#e53935",
    "#ef5350",
    "#b71c1c",
    "#e57373",
    "#ff8a80",
    "#d32f2f",
)


def empty_failure_clusters() -> Dict[str, Any]:
    return {
        "n_attempts": 0,
        "n_pass": 0,
        "n_abort": 0,
        "n_timeout": 0,
        "n_fail": 0,
        "n_other": 0,
        "n_failure": 0,
        "n_groups_with_abort": 0,
        "n_groups_with_failure": 0,
        "by_category": [],
        "by_reason": [],
        "by_group": {},
        "group_trend": [],
    }


def empty_type_breakdown() -> Dict[str, Any]:
    return {
        "n_attempts": 0,
        "n_pass": 0,
        "n_fail": 0,
        "n_timeout": 0,
        "n_abort": 0,
        "n_other": 0,
        "n_timeout_rw1": 0,
        "n_timeout_rw0": 0,
        "n_timeout_none": 0,
        "segments": [],
    }


def empty_train_sample_stats() -> Dict[str, Any]:
    return {
        "n_samples": 0,
        "n_train_kept": 0,
        "n_train_dropped": 0,
        "keep_pct": 0.0,
        "by_drop_reason": [],
    }


def empty_annotations() -> Dict[str, Any]:
    return {
        "group_briefs": {},
        "failure_clusters": empty_failure_clusters(),
        "abort_clusters": {
            "total": 0,
            "n_groups_with_abort": 0,
            "by_reason": [],
            "by_group": {},
        },
        "attempt_outcomes": {},
        "type_breakdown": empty_type_breakdown(),
        "train_sample_stats": empty_train_sample_stats(),
    }


def sidecar_path_for(pt_path: str) -> str:
    return f"{pt_path}{SIDECAR_SUFFIX}"


def _prompt_text_of(sample: Dict[str, Any]) -> str:
    prompt = sample.get("prompt") or ""
    if isinstance(prompt, list):
        parts: List[str] = []
        for item in prompt:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    if isinstance(prompt, dict):
        return str(prompt.get("content") or prompt.get("text") or "").strip()
    return str(prompt or "").strip()


def _pt_fingerprint(pt_path: str) -> Optional[Tuple[float, int]]:
    try:
        st = os.stat(pt_path)
        return (float(st.st_mtime), int(st.st_size))
    except OSError:
        return None


def _serialize_attempt_outcomes(
    outcomes: Dict[Any, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in (outcomes or {}).items():
        if isinstance(key, tuple) and len(key) == 2:
            out[f"{key[0]}:{key[1]}"] = val
        else:
            out[str(key)] = val
    return out


def _deserialize_attempt_outcomes(
    raw: Dict[str, Any],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        if isinstance(key, str) and ":" in key:
            left, right = key.split(":", 1)
            try:
                out[(int(left), int(right))] = val
                continue
            except (TypeError, ValueError):
                pass
        if isinstance(key, (list, tuple)) and len(key) == 2:
            try:
                out[(int(key[0]), int(key[1]))] = val
            except (TypeError, ValueError):
                continue
    return out


def _normalize_group_briefs(raw: Any) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    if not isinstance(raw, dict):
        return out
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        try:
            gi = int(key)
        except (TypeError, ValueError):
            continue
        out[gi] = {
            "title": str(val.get("title") or ""),
            "prompt": str(val.get("prompt") or ""),
        }
    return out


def _annotations_complete(cached: Dict[str, Any]) -> bool:
    return (
        isinstance(cached, dict)
        and "type_breakdown" in cached
        and "train_sample_stats" in cached
        and "failure_clusters" in cached
    )


def _read_sidecar(pt_path: str) -> Optional[Dict[str, Any]]:
    path = sidecar_path_for(pt_path)
    if not os.path.isfile(path):
        return None
    fp = _pt_fingerprint(pt_path)
    if fp is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("读取注解 sidecar 失败 path={} err={}", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version") or 0) != _SIDECAR_VERSION:
        return None
    if float(payload.get("pt_mtime") or -1) != fp[0]:
        return None
    if int(payload.get("pt_size") or -1) != fp[1]:
        return None
    ann = payload.get("annotations")
    if not _annotations_complete(ann or {}):
        return None
    assert isinstance(ann, dict)
    result = empty_annotations()
    result["group_briefs"] = _normalize_group_briefs(ann.get("group_briefs"))
    result["failure_clusters"] = ann.get("failure_clusters") or empty_failure_clusters()
    result["abort_clusters"] = ann.get("abort_clusters") or result["abort_clusters"]
    result["attempt_outcomes"] = _deserialize_attempt_outcomes(
        ann.get("attempt_outcomes") or {}
    )
    result["type_breakdown"] = ann.get("type_breakdown") or empty_type_breakdown()
    result["train_sample_stats"] = (
        ann.get("train_sample_stats") or empty_train_sample_stats()
    )
    return result


def _write_sidecar(pt_path: str, annotations: Dict[str, Any]) -> None:
    fp = _pt_fingerprint(pt_path)
    if fp is None:
        return
    path = sidecar_path_for(pt_path)
    payload = {
        "version": _SIDECAR_VERSION,
        "pt_mtime": fp[0],
        "pt_size": fp[1],
        "annotations": {
            "group_briefs": {
                str(k): v for k, v in (annotations.get("group_briefs") or {}).items()
            },
            "failure_clusters": annotations.get("failure_clusters")
            or empty_failure_clusters(),
            "abort_clusters": annotations.get("abort_clusters")
            or empty_annotations()["abort_clusters"],
            "attempt_outcomes": _serialize_attempt_outcomes(
                annotations.get("attempt_outcomes") or {}
            ),
            "type_breakdown": annotations.get("type_breakdown")
            or empty_type_breakdown(),
            "train_sample_stats": annotations.get("train_sample_stats")
            or empty_train_sample_stats(),
        },
    }
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("写入注解 sidecar 失败 path={} err={}", path, exc)
        try:
            if os.path.isfile(f"{path}.tmp"):
                os.remove(f"{path}.tmp")
        except OSError:
            pass


def _compute_annotations(pt_path: str) -> Dict[str, Any]:
    from utils.rollout_pt_parser import (
        cluster_abort_reasons,
        cluster_attempt_outcomes,
        index_attempt_outcomes,
        load_pt,
        summarize_stack_type_breakdown,
        summarize_train_sample_stats,
    )

    dump = load_pt(pt_path)
    briefs: Dict[int, Dict[str, str]] = {}
    for sample in dump.get("samples") or []:
        if not isinstance(sample, dict):
            continue
        gi = sample.get("group_index")
        if not isinstance(gi, int) or gi in briefs:
            continue
        md = sample.get("metadata") or {}
        title = str(md.get("instance_id") or sample.get("label") or "").strip()
        briefs[gi] = {
            "title": title,
            "prompt": _prompt_text_of(sample),
        }
    failure = cluster_attempt_outcomes(dump)
    return {
        "group_briefs": briefs,
        "failure_clusters": failure,
        "abort_clusters": cluster_abort_reasons(dump),
        "attempt_outcomes": index_attempt_outcomes(dump),
        "type_breakdown": summarize_stack_type_breakdown(dump),
        "train_sample_stats": summarize_train_sample_stats(dump),
    }


def load_pt_annotations(pt_path: Optional[str]) -> Dict[str, Any]:
    """一次读 pt：group_briefs + 失败聚类 + 类型柱 + 进入训练统计。

    命中顺序：内存缓存 → sidecar → ``load_pt``（成功后写 sidecar）。
    """
    empty = empty_annotations()
    if not pt_path:
        return empty
    cached = _PT_ANNOTATION_CACHE.get(pt_path)
    if cached is not None and _annotations_complete(cached):
        return cached

    sidecar_hit = _read_sidecar(pt_path)
    if sidecar_hit is not None:
        _PT_ANNOTATION_CACHE[pt_path] = sidecar_hit
        return sidecar_hit

    try:
        result = _compute_annotations(pt_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取 pt 注解失败 pt={}: {}", pt_path, exc)
        result = empty
        _PT_ANNOTATION_CACHE[pt_path] = result
        return result

    _PT_ANNOTATION_CACHE[pt_path] = result
    _write_sidecar(pt_path, result)
    return result


def warm_pt_annotation_sidecar(pt_path: Optional[str]) -> bool:
    """扫描成功后预写 sidecar；失败仅记日志，不抛出。"""
    if not pt_path:
        return False
    try:
        ann = load_pt_annotations(pt_path)
        return _annotations_complete(ann)
    except Exception as exc:  # noqa: BLE001
        logger.warning("预热注解 sidecar 失败 pt={} err={}", pt_path, exc)
        return False


def assign_stack_type_colors(key_totals: Counter) -> Dict[str, Dict[str, str]]:
    """按全局频次为类型柱 segment key 分配稳定颜色与图例文案。"""
    fail_keys: List[Tuple[str, int]] = []
    timeout_rw1: List[Tuple[str, int]] = []
    timeout_rw0: List[Tuple[str, int]] = []
    timeout_none: List[Tuple[str, int]] = []
    abort_keys: List[Tuple[str, int]] = []
    other_keys: List[Tuple[str, int]] = []
    for key, cnt in key_totals.most_common():
        sk = str(key)
        if sk == "pass":
            continue
        if sk.startswith("fail:"):
            fail_keys.append((sk, int(cnt)))
        elif sk.startswith("timeout:") and sk.endswith(":rw1"):
            timeout_rw1.append((sk, int(cnt)))
        elif sk.startswith("timeout:") and sk.endswith(":rw0"):
            timeout_rw0.append((sk, int(cnt)))
        elif sk.startswith("timeout:"):
            timeout_none.append((sk, int(cnt)))
        elif sk.startswith("abort:"):
            abort_keys.append((sk, int(cnt)))
        else:
            other_keys.append((sk, int(cnt)))

    out: Dict[str, Dict[str, str]] = {
        "pass": {"color": _STACK_PASS_COLOR, "label": "通过"},
    }

    def _paint(
        items: List[Tuple[str, int]],
        palette: Tuple[str, ...],
        label_of,
    ) -> None:
        for i, (key, _) in enumerate(items):
            color = palette[i] if i < len(palette) else palette[-1]
            out[key] = {"color": color, "label": label_of(key)}

    def _fail_label(key: str) -> str:
        return f"失败·{key[len('fail:'):]}"

    def _abort_label(key: str) -> str:
        return f"abort·{key[len('abort:'):]}"

    def _timeout_label(key: str, rw_zh: str) -> str:
        body = key[len("timeout:") :]
        if body.endswith(":rw1") or body.endswith(":rw0") or body.endswith(":none"):
            reason = body.rsplit(":", 1)[0]
        else:
            reason = body
        return f"超时·{rw_zh}·{reason}"

    _paint(fail_keys, _STACK_FAIL_PALETTE, _fail_label)
    _paint(
        timeout_rw1,
        _STACK_TIMEOUT_RW1_PALETTE,
        lambda k: _timeout_label(k, "有奖"),
    )
    _paint(
        timeout_rw0,
        _STACK_TIMEOUT_RW0_PALETTE,
        lambda k: _timeout_label(k, "无奖"),
    )
    _paint(
        timeout_none,
        _STACK_TIMEOUT_NONE_PALETTE,
        lambda k: _timeout_label(k, "无测"),
    )
    _paint(abort_keys, _STACK_ABORT_PALETTE, _abort_label)
    for key, _ in other_keys:
        out[key] = {
            "color": _STACK_OTHER_COLOR,
            "label": key if ":" not in key else key,
        }
    return out


def _type_segments_from_ann(ann: Dict[str, Any]) -> List[Dict[str, Any]]:
    breakdown = ann.get("type_breakdown") or {}
    segs: List[Dict[str, Any]] = []
    for seg in breakdown.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        key = str(seg.get("key") or "")
        segs.append(
            {
                "key": key,
                "category": seg.get("category"),
                "reason": seg.get("reason"),
                "reward_flag": seg.get("reward_flag"),
                "count": int(seg.get("count") or 0),
                "label": seg.get("label") or key,
            }
        )
    return segs


def _failure_summary(ann: Dict[str, Any]) -> Dict[str, Any]:
    fc = ann.get("failure_clusters") or {}
    return {
        "n_attempts": int(fc.get("n_attempts") or 0),
        "n_pass": int(fc.get("n_pass") or 0),
        "n_abort": int(fc.get("n_abort") or 0),
        "n_timeout": int(fc.get("n_timeout") or 0),
        "n_fail": int(fc.get("n_fail") or 0),
        "n_other": int(fc.get("n_other") or 0),
        "by_reason": list(fc.get("by_reason") or []),
    }


def list_task_annotation_meta(task_id: int) -> Dict[str, Any]:
    """返回待渐进加载的 rollout 有序列表（按业务 rollout_id）。"""
    from db.db_element import Session
    from db.db_rollout import DBRollout
    from db.db_task import DBTask

    with Session() as session:
        task = session.query(DBTask).filter(DBTask.primary_id == task_id).first()
        if task is None:
            from data.rollout_insight_exception import RolloutInsightException

            raise RolloutInsightException(
                RolloutInsightException.TASK_NOT_FOUND, f"任务 #{task_id} 不存在"
            )
        rows = (
            session.query(DBRollout)
            .filter(DBRollout.task_id == task_id)
            .order_by(DBRollout.rollout_id.asc(), DBRollout.primary_id.asc())
            .all()
        )
        items = [
            {
                "rollout_pk": int(r.primary_id),
                "rollout_id": r.rollout_id,
                "pt_path": r.pt_path or "",
            }
            for r in rows
        ]
    return {"task_id": task_id, "rollouts": items}


def get_rollout_annotation_payload(task_id: int, rollout_pk: int) -> Dict[str, Any]:
    """加载单个 rollout 的类型柱 / 训练统计 / 失败摘要。"""
    from db.db_element import Session
    from db.db_rollout import DBRollout
    from data.rollout_insight_exception import RolloutInsightException

    with Session() as session:
        obj = (
            session.query(DBRollout)
            .filter(DBRollout.primary_id == rollout_pk)
            .first()
        )
        if obj is None or int(obj.task_id) != int(task_id):
            raise RolloutInsightException(
                RolloutInsightException.PARSE_DUMP_NOT_FOUND,
                f"任务 #{task_id} 下无 rollout_pk={rollout_pk}",
            )
        pt_path = obj.pt_path
        rollout_id = obj.rollout_id

    ann = load_pt_annotations(pt_path)
    ts = ann.get("train_sample_stats") or empty_train_sample_stats()
    return {
        "task_id": task_id,
        "rollout_pk": int(rollout_pk),
        "rollout_id": rollout_id,
        "pt_path": pt_path or "",
        "type_segments": _type_segments_from_ann(ann),
        "train_sample_stats": {
            "n_samples": int(ts.get("n_samples") or 0),
            "n_train_kept": int(ts.get("n_train_kept") or 0),
            "n_train_dropped": int(ts.get("n_train_dropped") or 0),
            "keep_pct": float(ts.get("keep_pct") or 0.0),
            "by_drop_reason": list(ts.get("by_drop_reason") or []),
        },
        "failure_clusters": _failure_summary(ann),
    }


def clear_annotation_cache_for_tests() -> None:
    """仅供单测清空进程内缓存。"""
    _PT_ANNOTATION_CACHE.clear()
