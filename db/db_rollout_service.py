#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight DBRolloutService —— ri_rollout_table 的 CRUD。

核心是 upsert_with_fingerprint：以 (pt_path, pt_size_bytes, pt_mtime) 为指纹定位
历史记录；如果不存在则 INSERT；存在则刷新 mean_reward / n_samples / n_attempts 等摘要。
"""
from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import and_, desc

from db.db_element import Session
from db.db_rollout import DBRollout
from db.db_task import DBTask
from data.rollout_insight_exception import RolloutInsightException


class DBRolloutService:
    """rollout 主表的读写服务。"""

    @staticmethod
    def upsert_with_fingerprint(
        task_id: int,
        rollout_id: int,
        pt_path: str,
        pt_size_bytes: int,
        pt_mtime: float,
        mean_reward: float = 0.0,
        n_samples: int = 0,
        n_attempts: int = 0,
        instance_count: int = 0,
    ) -> DBRollout:
        """按指纹 (pt_path, pt_size_bytes, pt_mtime) upsert。

        流程：
          1. 按指纹查找 DBRollout 是否存在；
          2. 存在则更新 mean_reward / n_samples / n_attempts / instance_count / scanned_at；
          3. 不存在则 INSERT（同时校验 task 存在）。
        """
        if not pt_path:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "pt_path 不能为空"
            )
        with Session() as session:
            existing: Optional[DBRollout] = (
                session.query(DBRollout)
                .filter(
                    and_(
                        DBRollout.pt_path == pt_path,
                        DBRollout.pt_size_bytes == pt_size_bytes,
                        DBRollout.pt_mtime == pt_mtime,
                    )
                )
                .first()
            )
            if existing is not None:
                existing.mean_reward = mean_reward
                existing.n_samples = n_samples
                existing.n_attempts = n_attempts
                existing.instance_count = instance_count
                existing.scanned_at = datetime.now()
                session.commit()
                session.refresh(existing)
                return existing

            # 验证 task 存在，避免悬挂外键
            task = (
                session.query(DBTask)
                .filter(DBTask.primary_id == task_id)
                .first()
            )
            if task is None:
                raise RolloutInsightException(
                    RolloutInsightException.TASK_NOT_FOUND,
                    f"task_id={task_id} 不存在，无法创建 rollout",
                )

            rollout = DBRollout(
                task_id=task_id,
                rollout_id=rollout_id,
                pt_path=pt_path,
                pt_size_bytes=pt_size_bytes,
                pt_mtime=pt_mtime,
                mean_reward=mean_reward,
                n_samples=n_samples,
                n_attempts=n_attempts,
                instance_count=instance_count,
                scanned_at=datetime.now(),
            )
            session.add(rollout)
            session.commit()
            session.refresh(rollout)
            return rollout

    @staticmethod
    def list_by_task(
        task_id: int,
        limit: int = 200,
        offset: int = 0,
    ) -> List[DBRollout]:
        """按 task_id 列出 rollout，按 rollout_id 升序。"""
        with Session() as session:
            return (
                session.query(DBRollout)
                .filter(DBRollout.task_id == task_id)
                .order_by(DBRollout.rollout_id.asc())
                .offset(offset)
                .limit(limit)
                .all()
            )

    @staticmethod
    def get_by_pt_path(pt_path: str) -> Optional[DBRollout]:
        """按 .pt 绝对路径查 rollout；用于重扫/调试定位。"""
        if not pt_path:
            return None
        with Session() as session:
            return (
                session.query(DBRollout)
                .filter(DBRollout.pt_path == pt_path)
                .order_by(desc(DBRollout.scanned_at))
                .first()
            )

    @staticmethod
    def get_by_id(primary_id: int) -> Optional[DBRollout]:
        """按主键查 rollout。"""
        with Session() as session:
            return (
                session.query(DBRollout)
                .filter(DBRollout.primary_id == primary_id)
                .first()
            )

    @staticmethod
    def delete_by_task(task_id: int) -> int:
        """按 task_id 删除所有 rollout，返回受影响行数。"""
        with Session() as session:
            count = (
                session.query(DBRollout)
                .filter(DBRollout.task_id == task_id)
                .delete(synchronize_session=False)
            )
            session.commit()
            return count
