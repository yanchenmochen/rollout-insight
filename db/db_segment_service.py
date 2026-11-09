#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight DBSegmentService —— ri_segment_table 的 CRUD。

唯一键 (attempt_id, segment_index)；upsert 由应用层主动以唯一键定位行，
存在则覆盖可更新字段，不存在则 INSERT。
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, func

from db.db_element import Session
from db.db_segment import DBSegment
from data.rollout_insight_exception import RolloutInsightException


class DBSegmentService:
    """segment 表的读写服务。"""

    @staticmethod
    def upsert(**fields: Any) -> DBSegment:
        """按 (attempt_id, segment_index) upsert 一个 segment。

        必填：attempt_id, segment_index。
        """
        attempt_id = fields.get("attempt_id")
        segment_index = fields.get("segment_index")
        if attempt_id is None or segment_index is None:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT,
                "upsert segment 必填 attempt_id 与 segment_index",
            )
        with Session() as session:
            existing: Optional[DBSegment] = (
                session.query(DBSegment)
                .filter(
                    and_(
                        DBSegment.attempt_id == attempt_id,
                        DBSegment.segment_index == segment_index,
                    )
                )
                .first()
            )
            if existing is None:
                seg = DBSegment(**fields)
                session.add(seg)
                session.commit()
                session.refresh(seg)
                return seg
            # 已存在：合并可更新字段
            for field_name in (
                "segment_count",
                "status",
                "reward",
                "response_length",
                "trainable_tokens",
                "mask_sum",
                "sandbox_id",
            ):
                if field_name in fields:
                    setattr(existing, field_name, fields[field_name])
            session.commit()
            session.refresh(existing)
            return existing

    @staticmethod
    def list_by_attempt(
        attempt_id: int,
        limit: int = 500,
        offset: int = 0,
    ) -> List[DBSegment]:
        """列出某个 attempt 下所有 segment，按 segment_index 升序。"""
        with Session() as session:
            return (
                session.query(DBSegment)
                .filter(DBSegment.attempt_id == attempt_id)
                .order_by(DBSegment.segment_index.asc())
                .offset(offset)
                .limit(limit)
                .all()
            )

    @staticmethod
    def get_by_id(primary_id: int) -> Optional[DBSegment]:
        """按主键查 segment。"""
        with Session() as session:
            return (
                session.query(DBSegment)
                .filter(DBSegment.primary_id == primary_id)
                .first()
            )

    @staticmethod
    def list_by_task(
        task_id: int,
        limit: int = 2000,
        offset: int = 0,
    ) -> List[DBSegment]:
        """列出某个 task 下所有 segment（通过 JOIN attempt 进行过滤）。"""
        # 局部 import 避免循环
        from db.db_attempt import DBAttempt

        with Session() as session:
            q = session.query(DBSegment).join(
                DBAttempt, DBSegment.attempt_id == DBAttempt.primary_id
            )
            q = q.filter(DBAttempt.task_id == task_id)
            q = q.order_by(
                DBAttempt.rollout_id.asc(),
                DBAttempt.group_index.asc(),
                DBAttempt.attempt_id.asc(),
                DBSegment.segment_index.asc(),
            )
            return q.offset(offset).limit(limit).all()

    @staticmethod
    def count_by_rollout_for_task(task_id: int) -> Dict[int, int]:
        """按业务键 rollout_id 统计某 task 下各 rollout 的 segment 总数。

        返回 ``{rollout_id: n_segments}``；无 segment 的 rollout 不会出现在结果中。
        """
        from db.db_attempt import DBAttempt

        with Session() as session:
            rows = (
                session.query(
                    DBAttempt.rollout_id,
                    func.count(DBSegment.primary_id),
                )
                .join(DBSegment, DBSegment.attempt_id == DBAttempt.primary_id)
                .filter(DBAttempt.task_id == task_id)
                .group_by(DBAttempt.rollout_id)
                .all()
            )
            return {
                int(rollout_id): int(count)
                for rollout_id, count in rows
                if rollout_id is not None
            }

    @staticmethod
    def delete_by_attempt(attempt_id: int) -> int:
        """删除某个 attempt 下所有 segment，返回受影响行数。"""
        with Session() as session:
            count = (
                session.query(DBSegment)
                .filter(DBSegment.attempt_id == attempt_id)
                .delete(synchronize_session=False)
            )
            session.commit()
            return count

    @staticmethod
    def delete_by_task(task_id: int) -> int:
        """按 task_id 级联删除所有 segment（经 JOIN attempt）。"""
        from db.db_attempt import DBAttempt

        with Session() as session:
            q = session.query(DBSegment).join(
                DBAttempt, DBSegment.attempt_id == DBAttempt.primary_id
            )
            q = q.filter(DBAttempt.task_id == task_id)
            count = q.delete(synchronize_session=False)
            session.commit()
            return count
