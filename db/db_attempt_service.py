#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight DBAttemptService —— ri_attempt_table 的 CRUD。

唯一键 (task_id, rollout_id, group_index, attempt_id)；upsert 走 session.merge()，
重复写入会覆盖已有行的可更新字段。
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_

from db.db_element import Session
from db.db_attempt import DBAttempt
from data.rollout_insight_exception import RolloutInsightException


class DBAttemptService:
    """attempt 主表的读写服务。"""

    @staticmethod
    def _key_dict(
        task_id: int,
        rollout_id: int,
        group_index: int,
        attempt_id: int,
    ) -> Dict[str, int]:
        return {
            "task_id": task_id,
            "rollout_id": rollout_id,
            "group_index": group_index,
            "attempt_id": attempt_id,
        }

    @classmethod
    def upsert(cls, **fields: Any) -> DBAttempt:
        """按唯一键 upsert 一个 attempt 行。

        必填：task_id, rollout_id, group_index, attempt_id。
        """
        key = cls._key_dict(
            task_id=fields["task_id"],
            rollout_id=fields["rollout_id"],
            group_index=fields["group_index"],
            attempt_id=fields["attempt_id"],
        )
        with Session() as session:
            existing: Optional[DBAttempt] = (
                session.query(DBAttempt)
                .filter(
                    and_(
                        DBAttempt.task_id == key["task_id"],
                        DBAttempt.rollout_id == key["rollout_id"],
                        DBAttempt.group_index == key["group_index"],
                        DBAttempt.attempt_id == key["attempt_id"],
                    )
                )
                .first()
            )
            if existing is None:
                attempt = DBAttempt(**fields)
                session.add(attempt)
                session.commit()
                session.refresh(attempt)
                return attempt
            # 已存在：合并可更新字段
            for field_name in (
                "sandbox_id",
                "instance_id",
                "status",
                "reward",
                "agent_turns",
                "tool_calls",
                "diff_line_count",
                "response_length",
                "trainable_tokens",
                "finalized",
            ):
                if field_name in fields:
                    setattr(existing, field_name, fields[field_name])
            session.commit()
            session.refresh(existing)
            return existing

    @staticmethod
    def list_by_rollout(
        task_id: int,
        rollout_id: int,
        limit: int = 500,
        offset: int = 0,
    ) -> List[DBAttempt]:
        """列出某个 rollout 下所有 attempt，按 (group_index, attempt_id) 升序。"""
        with Session() as session:
            return (
                session.query(DBAttempt)
                .filter(
                    and_(
                        DBAttempt.task_id == task_id,
                        DBAttempt.rollout_id == rollout_id,
                    )
                )
                .order_by(
                    DBAttempt.group_index.asc(),
                    DBAttempt.attempt_id.asc(),
                )
                .offset(offset)
                .limit(limit)
                .all()
            )

    @staticmethod
    def list_by_task(
        task_id: int,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[DBAttempt]:
        """列出某个 task 下所有 attempt。"""
        with Session() as session:
            return (
                session.query(DBAttempt)
                .filter(DBAttempt.task_id == task_id)
                .order_by(
                    DBAttempt.rollout_id.asc(),
                    DBAttempt.group_index.asc(),
                    DBAttempt.attempt_id.asc(),
                )
                .offset(offset)
                .limit(limit)
                .all()
            )

    @staticmethod
    def get_by_id(primary_id: int) -> Optional[DBAttempt]:
        """按主键查 attempt。"""
        with Session() as session:
            return (
                session.query(DBAttempt)
                .filter(DBAttempt.primary_id == primary_id)
                .first()
            )

    @staticmethod
    def delete_by_rollout(task_id: int, rollout_id: int) -> int:
        """删除某个 rollout 下所有 attempt，返回受影响行数。"""
        with Session() as session:
            count = (
                session.query(DBAttempt)
                .filter(
                    and_(
                        DBAttempt.task_id == task_id,
                        DBAttempt.rollout_id == rollout_id,
                    )
                )
                .delete(synchronize_session=False)
            )
            session.commit()
            return count

    @staticmethod
    def delete_by_task(task_id: int) -> int:
        """删除某个 task 下所有 attempt。"""
        with Session() as session:
            count = (
                session.query(DBAttempt)
                .filter(DBAttempt.task_id == task_id)
                .delete(synchronize_session=False)
            )
            session.commit()
            return count
