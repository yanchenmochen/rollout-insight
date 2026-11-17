#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight segment 检索 service（v0.2.0）。

按 (task_id, min_reward) 过滤 segment；keyword 非空时走 transcript LIKE 模糊匹配，
并同步把未 fetched 的段解析写回（最多前 50 条，避免阻塞）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import and_ as sql_and

from db.db_attempt import DBAttempt
from db.db_element import Session
from db.db_segment import DBSegment
from db.db_segment_transcript import DBSegmentTranscript
from db.db_segment_transcript_service import DBSegmentTranscriptService
from service.segment_transcript_service import ensure_transcript


__all__ = ["search_segments"]


_SNIPPET_RADIUS = 80  # 关键词命中前后保留的字符数
_DEFAULT_PRELOAD = 50  # 关键词触发时同步写回的最大条数


def search_segments(
    task_id: Optional[int] = None,
    min_reward: Optional[float] = None,
    keyword: str = "",
    limit: int = 50,
    offset: int = 0,
    group_index: Optional[int] = None,
    attempt_id: Optional[int] = None,
) -> Dict[str, Any]:
    """按 task / reward / keyword，以及可选的 group_index、业务 attempt_id 检索 segment。

    ``attempt_id`` 参数是 ``DBAttempt.attempt_id``（失败表里那一列），精确匹配。
    返回 item 里的 ``attempt_id`` 仍是 attempt 表主键（CLI 在用）；
    业务 id 放在 ``attempt_biz_id``，组号放在 ``group_index``。

    ``min_reward`` 未传时，关键字检索不按 reward 过滤，reward 为空的 segment 仍会出现。
    无关键字列表维持历史默认：未传时按 ``reward >= 0`` 过滤。显式传入下限后，两种检索
    都要求 ``reward >= min_reward``（空 reward 不满足比较）。

    返回：
        {
            "total": int,
            "items": [
                {
                    "segment_id": int,
                    "attempt_id": int,
                    "group_index": int | None,
                    "attempt_biz_id": int | None,
                    "reward": float | None,
                    "sandbox_id": str,
                    "snippet": str,
                    "full_text_fetched": int,
                },
                ...
            ],
        }

    命中关键字的 segment 来自 DBSegmentTranscriptService.search_by_keyword；无关键字时
    走纯 segment 表过滤。LIMIT/MAX 支持 limit + offset 分页。
    """
    keyword = (keyword or "").strip()
    limit = max(1, int(limit or 50))
    offset = max(0, int(offset or 0))

    if keyword:
        return _search_with_keyword(
            task_id=task_id, min_reward=min_reward, keyword=keyword,
            limit=limit, offset=offset,
            group_index=group_index, attempt_id=attempt_id,
        )
    # 无关键字列表的历史默认是 reward >= 0；关键字路径不能套这个默认，否则空 reward 会被剔掉。
    listed_min = 0.0 if min_reward is None else float(min_reward)
    return _search_no_keyword(
        task_id=task_id, min_reward=listed_min, limit=limit, offset=offset,
        group_index=group_index, attempt_id=attempt_id,
    )


def _apply_attempt_filters(
    query,
    task_id: Optional[int],
    group_index: Optional[int],
    attempt_id: Optional[int],
    min_reward: Optional[float],
):
    """在已 join DBAttempt 与 DBSegment 的查询上追加可选过滤。"""
    if task_id is not None:
        query = query.filter(DBAttempt.task_id == int(task_id))
    if group_index is not None:
        query = query.filter(DBAttempt.group_index == int(group_index))
    if attempt_id is not None:
        query = query.filter(DBAttempt.attempt_id == int(attempt_id))
    if min_reward is not None:
        query = query.filter(DBSegment.reward >= float(min_reward))
    return query


def _item_from_pair(seg: DBSegment, attempt: Optional[DBAttempt], snippet: str, fetched: int) -> Dict[str, Any]:
    """segment + attempt 行 → 检索 item。attempt_id 保持主键语义。"""
    return {
        "segment_id": int(seg.primary_id),
        "attempt_id": int(seg.attempt_id) if seg.attempt_id is not None else 0,
        "group_index": int(attempt.group_index) if attempt is not None else None,
        "attempt_biz_id": int(attempt.attempt_id) if attempt is not None else None,
        "reward": (float(seg.reward) if seg.reward is not None else None),
        "sandbox_id": seg.sandbox_id or "",
        "snippet": snippet,
        "full_text_fetched": int(fetched),
    }


# ---------- 纯 segment 检索（无 keyword） ----------


def _search_no_keyword(
    task_id: Optional[int],
    min_reward: float,
    limit: int,
    offset: int,
    group_index: Optional[int] = None,
    attempt_id: Optional[int] = None,
) -> Dict[str, Any]:
    """无 keyword：按 task / reward / group / 业务 attempt_id 过滤 segment，并 JOIN attempt。"""
    with Session() as session:
        query = session.query(DBSegment, DBAttempt).join(
            DBAttempt, DBSegment.attempt_id == DBAttempt.primary_id
        )
        query = _apply_attempt_filters(query, task_id, group_index, attempt_id, min_reward)
        total = query.count()
        rows = (
            query.order_by(
                DBAttempt.rollout_id.asc(),
                DBAttempt.group_index.asc(),
                DBAttempt.attempt_id.asc(),
                DBSegment.segment_index.asc(),
            )
            .offset(offset)
            .limit(limit)
            .all()
        )

        # 一次批量查 transcript full_text_fetched 状态，避免 N+1
        seg_ids = [int(seg.primary_id) for seg, _att in rows]
        fetched_map = _bulk_fetched_map(session, seg_ids)

        items: List[Dict[str, Any]] = []
        for seg, att in rows:
            items.append(
                _item_from_pair(
                    seg,
                    att,
                    "",
                    int(fetched_map.get(int(seg.primary_id), 0)),
                )
            )
        return {"total": int(total), "items": items}


# ---------- 关键字检索 ----------


def _search_with_keyword(
    task_id: Optional[int],
    min_reward: Optional[float],
    keyword: str,
    limit: int,
    offset: int,
    group_index: Optional[int] = None,
    attempt_id: Optional[int] = None,
) -> Dict[str, Any]:
    """关键字检索：先查已 fetched 的 transcript LIKE 命中，再同步写回未 fetched 的段。

    流程：
      1. 找出 task 内未 fetched 的 segment，按 reward 过滤；
      2. 前 _DEFAULT_PRELOAD 条调 ensure_transcript 写回（v0.2.0 同步）；
      3. 再走 DBSegmentTranscriptService.search_by_keyword 拿到 LIKE 命中；
      4. 返回 (snippet, full_text_fetched) 信息。
    """
    pending_segments = _list_unfetched_segments(
        task_id=task_id, min_reward=min_reward, limit=_DEFAULT_PRELOAD,
        group_index=group_index, attempt_id=attempt_id,
    )
    if pending_segments:
        logger.info(
            "search_segments: 预热 {} 个未 fetched 的 segment (task_id={}, min_reward={})",
            len(pending_segments),
            task_id,
            min_reward,
        )
        for seg_id in pending_segments:
            try:
                ensure_transcript(seg_id, tokenizer_dir="", force=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "search_segments: segment_id={} ensure_transcript 失败: {}",
                    seg_id,
                    exc,
                )

    # 真正的 LIKE 检索
    rows = DBSegmentTranscriptService.search_by_keyword(
        keyword=keyword, task_id=task_id, limit=limit, offset=offset,
        group_index=group_index, attempt_id=attempt_id, min_reward=min_reward,
    )
    seg_ids = [int(r.segment_id) for r in rows]
    items = _build_keyword_items(seg_ids, rows, keyword)

    # total 用 LIKE 命中 + 一次 count 近似
    total = len(rows) if not rows else _count_keyword_hits(
        keyword, task_id, group_index, attempt_id, min_reward,
    )
    return {"total": int(total), "items": items}


def _count_keyword_hits(
    keyword: str,
    task_id: Optional[int],
    group_index: Optional[int] = None,
    attempt_id: Optional[int] = None,
    min_reward: Optional[float] = None,
) -> int:
    """统计 LIKE 命中总条数（含 task / group / 业务 attempt / reward），不实际拉行。"""
    with Session() as session:
        query = (
            session.query(DBSegmentTranscript)
            .filter(
                sql_and(
                    DBSegmentTranscript.full_text_fetched == 1,
                    DBSegmentTranscript.response_text.like(f"%{keyword}%"),
                )
            )
            .join(DBSegment, DBSegment.primary_id == DBSegmentTranscript.segment_id)
            .join(DBAttempt, DBAttempt.primary_id == DBSegment.attempt_id)
        )
        query = _apply_attempt_filters(query, task_id, group_index, attempt_id, min_reward)
        return int(query.count())


def _list_unfetched_segments(
    task_id: Optional[int],
    min_reward: Optional[float],
    limit: int,
    group_index: Optional[int] = None,
    attempt_id: Optional[int] = None,
) -> List[int]:
    """列出 (task_id, min_reward) 下尚未 lazy-load 的 segment_ids。"""
    with Session() as session:
        # LEFT JOIN ri_segment_transcript_table 过滤 full_text_fetched=0/缺
        from sqlalchemy import or_

        query = (
            session.query(DBSegment)
            .outerjoin(
                DBSegmentTranscript,
                DBSegmentTranscript.segment_id == DBSegment.primary_id,
            )
            .join(DBAttempt, DBSegment.attempt_id == DBAttempt.primary_id)
            .filter(
                or_(
                    DBSegmentTranscript.segment_id.is_(None),
                    DBSegmentTranscript.full_text_fetched == 0,
                )
            )
        )
        query = _apply_attempt_filters(query, task_id, group_index, attempt_id, min_reward)
        rows = (
            query.order_by(
                DBAttempt.rollout_id.asc(),
                DBAttempt.group_index.asc(),
                DBAttempt.attempt_id.asc(),
                DBSegment.segment_index.asc(),
            )
            .limit(max(1, limit))
            .all()
        )
        return [int(r.primary_id) for r in rows]


def _build_keyword_items(
    seg_ids: List[int],
    rows: List[DBSegmentTranscript],
    keyword: str,
) -> List[Dict[str, Any]]:
    """把 LIKE 命中行转成 search 返回 item。"""
    if not seg_ids:
        return []

    # 补齐 attempt / reward / sandbox 信息
    with Session() as session:
        pairs = (
            session.query(DBSegment, DBAttempt)
            .join(DBAttempt, DBSegment.attempt_id == DBAttempt.primary_id)
            .filter(DBSegment.primary_id.in_(seg_ids))
            .all()
        )
        pair_map = {int(seg.primary_id): (seg, att) for seg, att in pairs}

    items: List[Dict[str, Any]] = []
    for row in rows:
        seg_id = int(row.segment_id)
        pair = pair_map.get(seg_id)
        snippet = _extract_snippet(row.response_text or "", keyword)
        fetched = int(row.full_text_fetched or 0)
        if pair is None:
            items.append(
                {
                    "segment_id": seg_id,
                    "attempt_id": 0,
                    "group_index": None,
                    "attempt_biz_id": None,
                    "reward": None,
                    "sandbox_id": "",
                    "snippet": snippet,
                    "full_text_fetched": fetched,
                }
            )
            continue
        seg, att = pair
        items.append(_item_from_pair(seg, att, snippet, fetched))
    return items


def _extract_snippet(response_text: str, keyword: str) -> str:
    """围绕 keyword 命中位置截取前后 _SNIPPET_RADIUS 字符的 snippet。"""
    if not response_text:
        return ""
    idx = response_text.lower().find(keyword.lower())
    if idx < 0:
        return response_text[: _SNIPPET_RADIUS * 2]
    s = max(0, idx - _SNIPPET_RADIUS)
    e = min(len(response_text), idx + len(keyword) + _SNIPPET_RADIUS)
    return response_text[s:e]


def _bulk_fetched_map(session: Any, seg_ids: List[int]) -> Dict[int, int]:
    """批量查 transcript full_text_fetched 状态，避免 N+1。"""
    if not seg_ids:
        return {}
    rows = (
        session.query(DBSegmentTranscript)
        .filter(DBSegmentTranscript.segment_id.in_(seg_ids))
        .all()
    )
    return {int(r.segment_id): int(r.full_text_fetched or 0) for r in rows}