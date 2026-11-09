#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight v0.4.0 —— .pt dump 解析 service。

封装 utils.rollout_pt_parser，提供：
- ``parse_rollout(pt_path, force=False)``：把单份 .pt 的 attempts / segments / turns
  解析出来，并把 turn 行 upsert 到 ``ri_segment_turn_table``（T4 新表）。
- 复用 DBSegmentService / DBAttemptService 已有的 upsert 路径补全 summary 字段。
- v0.5.0 T2：``parse_attempt_token_stats(attempt_pk)`` —— attempt 维度聚合
  （n_tool_calls / n_diff_lines / total_resp_len / total_mask_sum /
  total_trainable_tokens / reward_bucket）

设计：
- 业务错误用 RolloutInsightException 抛出
- 调 loguru logger
- 任何 service/db 导入在函数内延迟，避免循环 import
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from loguru import logger

from data.rollout_insight_exception import RolloutInsightException
from utils.rollout_pt_parser import (
    load_pt,
    parse_attempts_and_segments,
    split_response_to_turns,
    split_response_to_turns_with_tokenizer,
    token_estimate,
)


__all__ = [
    "parse_rollout",
    "RolloutParseService",
    "parse_attempt_token_stats",  # v0.5.0 T2
]


# 阈值常量（v0.5.0 T2 reward_bucket 用）
_REWARD_EPS = 1e-9


class RolloutParseService:
    """.pt dump 解析 service。"""

    def __init__(self) -> None:
        # 不在 __init__ 里 import db，避免循环
        pass

    # ---------- 公共入口 ----------

    def parse_rollout(self, pt_path: str, force: bool = False) -> Dict[str, Any]:
        """解析单份 .pt dump，返回 attempts/segments 摘要，并落库 turns。

        Returns:
            {
                "pt_path": str,
                "rollout_id": int,
                "n_attempts": int,
                "n_segments": int,
                "attempts": [{key, segment_count, mean_reward, segments: [...]}],
                "all_segments": [segment_dict, ...],
                "persisted_turns": int,
            }
        """
        if not pt_path:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "pt_path 不能为空"
            )
        if not os.path.isfile(pt_path):
            raise RolloutInsightException(
                RolloutInsightException.PARSE_DUMP_NOT_FOUND,
                f".pt dump 文件不存在: {pt_path}",
            )

        logger.info("[parse_rollout] start pt={} force={}", pt_path, force)
        dump = load_pt(pt_path)
        parsed = parse_attempts_and_segments(dump)
        rollout_id = parsed.get("rollout_id")
        attempts = parsed.get("attempts", [])
        all_segments = parsed.get("all_segments", [])

        # 落库 turns：每个 segment 解析 turn 后 upsert
        persisted_turns = self._persist_turns(all_segments, force=force)

        logger.info(
            "[parse_rollout] done pt={} rollout_id={} attempts={} segments={} persisted_turns={}",
            pt_path,
            rollout_id,
            len(attempts),
            len(all_segments),
            persisted_turns,
        )

        return {
            "pt_path": pt_path,
            "rollout_id": rollout_id,
            "n_attempts": len(attempts),
            "n_segments": len(all_segments),
            "attempts": attempts,
            "all_segments": all_segments,
            "persisted_turns": persisted_turns,
        }

    # ---------- 内部 ----------

    def _persist_turns(self, segments: List[Dict[str, Any]], force: bool = False) -> int:
        """解析每个 segment 的 turns，落到 ``ri_segment_turn_table``。

        流程：
          1. 通过 sample 的 (rollout_id, group_index, instance_id, segment_index) 反查
             ``ri_segment_table.primary_id``；
          2. 调 ``split_response_to_turns`` 切 turn；
          3. 调 ``DBSegmentTurnService.upsert_segment_turns``（T4 提供）落库。
        """
        if not segments:
            return 0

        # 延迟导入 db 服务
        try:
            from db.db_segment_turn_service import DBSegmentTurnService
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_persist_turns: DBSegmentTurnService 不可用（T4 尚未落地？）: {}", exc
            )
            return 0

        # 按 (rollout_id, group_index, instance_id, segment_index) → segment_pk 索引
        # 一次拉出本次解析涉及的所有 segment 记录
        segment_pk_map = self._build_segment_pk_map(segments)
        if not segment_pk_map:
            logger.warning("_persist_turns: 找不到对应 ri_segment_table 行，跳过落库")
            return 0

        total = 0
        for seg in segments:
            pk = self._lookup_segment_pk(segment_pk_map, seg)
            if pk is None:
                continue
            response_text = seg.get("response_text") or ""
            if not response_text:
                continue
            turns = split_response_to_turns(response_text)
            if not turns:
                continue
            try:
                n = DBSegmentTurnService.upsert_segment_turns(
                    segment_id=int(pk), turns=turns, force=force
                )
                total += n
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "_persist_turns: 落库 segment_pk={} 失败: {}", pk, exc
                )
        return total

    @staticmethod
    def _build_segment_pk_map(
        segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """拉出本次解析涉及的所有 segment 行（仅 metadata）。"""
        try:
            from db.db_element import Session
            from db.db_attempt import DBAttempt
            from db.db_segment import DBSegment
        except Exception as exc:  # noqa: BLE001
            logger.warning("_build_segment_pk_map: 不可用: {}", exc)
            return []

        rollout_ids = sorted(
            {s.get("rollout_id") for s in segments if s.get("rollout_id") is not None}
        )
        if not rollout_ids:
            return []

        with Session() as session:
            # 一次拉齐所有 (rollout_id, group_index, instance_id, segment_index) 的 segment
            rows = (
                session.query(
                    DBSegment.primary_id,
                    DBAttempt.rollout_id,
                    DBAttempt.group_index,
                    DBAttempt.instance_id,
                    DBSegment.segment_index,
                )
                .join(DBAttempt, DBSegment.attempt_id == DBAttempt.primary_id)
                .filter(DBAttempt.rollout_id.in_(rollout_ids))
                .all()
            )
        return [
            {
                "primary_id": int(r[0]),
                "rollout_id": r[1],
                "group_index": r[2],
                "instance_id": r[3] or "",
                "segment_index": r[4],
            }
            for r in rows
        ]

    @staticmethod
    def _lookup_segment_pk(
        segment_pk_map: List[Dict[str, Any]],
        seg: Dict[str, Any],
    ) -> Optional[int]:
        for row in segment_pk_map:
            if (
                row.get("rollout_id") == seg.get("rollout_id")
                and row.get("group_index") == seg.get("group_index")
                and row.get("instance_id") == seg.get("instance_id")
                and row.get("segment_index") == seg.get("segment_index")
            ):
                return row.get("primary_id")
        return None


# 函数式快捷入口
def parse_rollout(pt_path: str, force: bool = False) -> Dict[str, Any]:
    """``RolloutParseService().parse_rollout`` 的快捷封装。"""
    return RolloutParseService().parse_rollout(pt_path, force=force)


# ---------------------------------------------------------------------------
# v0.5.0 T2：attempt 维度摘要聚合
# ---------------------------------------------------------------------------


_ABORT_STATUSES = frozenset(
    {"aborted", "abort", "cancelled", "canceled", "invalid", "failed_infra"}
)


def _is_abort_status(status: Any) -> bool:
    text = str(status or "").split(".")[-1].lower()
    return text in _ABORT_STATUSES


def _classify_reward_bucket(
    rewards: List[float], *, aborted: bool = False
) -> str:
    """把 attempts 的 segment reward 列表归类为 reward_bucket 标签。

    - 'aborted' —— attempt / segment 为 abort（不可测；即使 reward 被写成 0.0）
    - 'none'  —— 没有任何 segment 有数字 reward
    - 'all_zero' —— 所有 segment 都是 0（不含 None）
    - 'all_one' —— 所有 segment 都是 1（不含 None）
    - 'mixed' —— 既有非 0 又非 1，或 reward 落在 0~1 之间
    """
    if aborted:
        return "aborted"
    if not rewards:
        return "none"
    if all(abs(r) <= _REWARD_EPS for r in rewards):
        return "all_zero"
    if all(abs(r - 1.0) <= _REWARD_EPS for r in rewards):
        return "all_one"
    if any(0.0 < r < 1.0 for r in rewards):
        return "mixed"
    # 出现 >1 或 <0 的值都视作 mixed（避免错把奖励解码错误的样本归为 all_zero）
    if any(r > 1.0 + _REWARD_EPS or r < 0.0 - _REWARD_EPS for r in rewards):
        return "mixed"
    return "mixed"


def parse_attempt_token_stats(attempt_pk: int) -> Dict[str, Any]:
    """聚合某个 attempt 下所有 segment 的 token / mask / reward 统计，供 UI 用。

    Returns:
        {
            "attempt_id": int,
            "segment_count": int,
            "n_tool_calls": int,
            "n_diff_lines": int,
            "total_resp_len": int,
            "total_mask_sum": int,
            "total_trainable_tokens": int,
            "reward_bucket": str ('aborted' | 'all_zero' | 'all_one' | 'mixed' | 'none'),
            "rewards": [float, ...],      # 各 segment 可测 reward（不含 None / abort）
        }

    Raises:
        RolloutInsightException.DB_ERROR：DB 查询失败
        RolloutInsightException.NOT_IMPLEMENTED：模型未就绪
    """
    if attempt_pk is None:
        raise RolloutInsightException(
            RolloutInsightException.INVALID_ARGUMENT, "attempt_pk 不能为空"
        )
    try:
        from db.db_element import Session
        from db.db_attempt import DBAttempt
        from db.db_segment import DBSegment
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "parse_attempt_token_stats: 延迟导入失败: {}", exc
        )
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            f"db 模型未就绪: {exc}",
        ) from exc

    out: Dict[str, Any] = {
        "attempt_id": int(attempt_pk),
        "segment_count": 0,
        "n_tool_calls": 0,
        "n_diff_lines": 0,
        "total_resp_len": 0,
        "total_mask_sum": 0,
        "total_trainable_tokens": 0,
        "reward_bucket": "none",
        "rewards": [],
    }

    try:
        with Session() as session:
            # 1) attempt 行校验
            att = (
                session.query(DBAttempt)
                .filter(DBAttempt.primary_id == int(attempt_pk))
                .first()
            )
            if att is None:
                logger.warning(
                    "parse_attempt_token_stats: attempt_pk={} 不存在",
                    attempt_pk,
                )
                return out
            # 2) segments 聚合
            segs = (
                session.query(DBSegment)
                .filter(DBSegment.attempt_id == int(attempt_pk))
                .order_by(DBSegment.segment_index.asc())
                .all()
            )

            # n_tool_calls / n_diff_lines 取 attempt 行（DBAttempt 上已有聚合列）
            out["n_tool_calls"] = int(getattr(att, "tool_calls", 0) or 0)
            out["n_diff_lines"] = int(getattr(att, "diff_line_count", 0) or 0)

            attempt_aborted = _is_abort_status(getattr(att, "status", None))
            any_seg_aborted = False
            for s in segs:
                out["total_resp_len"] += int(getattr(s, "response_length", 0) or 0)
                out["total_mask_sum"] += int(getattr(s, "mask_sum", 0) or 0)
                out["total_trainable_tokens"] += int(
                    getattr(s, "trainable_tokens", 0) or 0
                )
                if _is_abort_status(getattr(s, "status", None)):
                    any_seg_aborted = True
                    continue  # abort 的兜底 0.0 不能进可测 rewards
                r = getattr(s, "reward", None)
                if isinstance(r, (int, float)):
                    out["rewards"].append(float(r))

            # n_diff_lines 缺 attempt 行（0）时退化为 segments sum
            if not out["n_diff_lines"]:
                for s in segs:
                    # segment 表本身没 diff_line_count 字段（与 attempt.metadata 重）
                    pass  # 占位：实际数据由 attempt 表提供

            out["segment_count"] = len(segs)
            out["reward_bucket"] = _classify_reward_bucket(
                out["rewards"], aborted=attempt_aborted or any_seg_aborted
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "parse_attempt_token_stats: DB 操作失败 attempt_pk={} err={}",
            attempt_pk,
            exc,
        )
        raise RolloutInsightException(
            RolloutInsightException.DB_ERROR,
            "attempt #{} 聚合失败: {}".format(attempt_pk, exc),
        ) from exc

    return out
