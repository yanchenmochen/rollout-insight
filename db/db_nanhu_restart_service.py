#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight DBNanhuRestartService —— ri_nanhu_restart_table 的 CRUD。

唯一键 (nanhu_job_id, restart_index)；新增一条重启记录前会先取
MAX(restart_index) + 1 作为下一次序号，保证 append-only 语义。
"""
from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import and_, desc, func

from db.db_element import Session
from db.db_nanhu_restart import DBNanhuRestart
from data.rollout_insight_exception import RolloutInsightException


class DBNanhuRestartService:
    """ri_nanhu_restart_table 的读写服务。"""

    @staticmethod
    def next_restart_index(nanhu_job_id: str) -> int:
        """计算下一个 restart_index：MAX + 1；表为空则返回 0。

        注意：用 `is None` 而非 `or -1` 兜底，避免 max_idx=0 时被误判为 falsy。
        """
        nanhu_job_id = (nanhu_job_id or "").strip()
        if not nanhu_job_id:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "nanhu_job_id 不能为空"
            )
        with Session() as session:
            max_idx: Optional[int] = (
                session.query(func.max(DBNanhuRestart.restart_index))
                .filter(DBNanhuRestart.nanhu_job_id == nanhu_job_id)
                .scalar()
            )
        return 0 if max_idx is None else int(max_idx) + 1

    @classmethod
    def create_record(
        cls,
        nanhu_job_id: str,
        restart_status: str = "REQUESTED",
        related_task_id: Optional[int] = None,
        restart_at: Optional[datetime] = None,
        restart_index: Optional[int] = None,
    ) -> DBNanhuRestart:
        """追加一条重启记录。

        restart_index 为 None 时由 next_restart_index 推算；
        非 None 时直接使用传入值（用于外部指定序号，例如 link-task 时附带）。
        """
        nanhu_job_id = (nanhu_job_id or "").strip()
        if not nanhu_job_id:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "nanhu_job_id 不能为空"
            )
        if restart_index is None:
            restart_index = cls.next_restart_index(nanhu_job_id)
        with Session() as session:
            row = DBNanhuRestart(
                nanhu_job_id=nanhu_job_id,
                restart_index=restart_index,
                restart_at=restart_at,
                restart_status=restart_status or "REQUESTED",
                related_task_id=related_task_id,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    @staticmethod
    def list_by_job(
        nanhu_job_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DBNanhuRestart]:
        """按 nanhu_job_id 列出重启记录，按 restart_index 倒序。"""
        nanhu_job_id = (nanhu_job_id or "").strip()
        if not nanhu_job_id:
            return []
        with Session() as session:
            return (
                session.query(DBNanhuRestart)
                .filter(DBNanhuRestart.nanhu_job_id == nanhu_job_id)
                .order_by(desc(DBNanhuRestart.restart_index))
                .offset(offset)
                .limit(limit)
                .all()
            )

    @staticmethod
    def latest_for_job(nanhu_job_id: str) -> Optional[DBNanhuRestart]:
        """返回某 job 的最近一次重启记录；不存在返回 None。"""
        nanhu_job_id = (nanhu_job_id or "").strip()
        if not nanhu_job_id:
            return None
        with Session() as session:
            return (
                session.query(DBNanhuRestart)
                .filter(DBNanhuRestart.nanhu_job_id == nanhu_job_id)
                .order_by(desc(DBNanhuRestart.restart_index))
                .first()
            )

    @staticmethod
    def update_status(
        nanhu_job_id: str,
        restart_index: int,
        restart_status: str,
    ) -> bool:
        """按 (nanhu_job_id, restart_index) 更新 restart_status；找不到返回 False。"""
        nanhu_job_id = (nanhu_job_id or "").strip()
        if not nanhu_job_id or restart_index is None or not restart_status:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT,
                "update_status 需要非空的 nanhu_job_id / restart_index / restart_status",
            )
        with Session() as session:
            row = (
                session.query(DBNanhuRestart)
                .filter(
                    and_(
                        DBNanhuRestart.nanhu_job_id == nanhu_job_id,
                        DBNanhuRestart.restart_index == restart_index,
                    )
                )
                .first()
            )
            if row is None:
                return False
            row.restart_status = restart_status
            session.commit()
            return True

    @staticmethod
    def count_by_job(nanhu_job_id: str) -> int:
        """统计某 job 的累计重启次数。"""
        nanhu_job_id = (nanhu_job_id or "").strip()
        if not nanhu_job_id:
            return 0
        with Session() as session:
            return (
                session.query(func.count(DBNanhuRestart.primary_id))
                .filter(DBNanhuRestart.nanhu_job_id == nanhu_job_id)
                .scalar()
                or 0
            )