#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight DBTaskService —— ri_task_table 的 CRUD。

所有方法 @staticmethod；所有写操作使用 `with Session() as session:` 短事务。
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import desc

from db.db_element import Session
from db.db_task import DBTask
from data.rollout_insight_exception import RolloutInsightException


class DBTaskService:
    """任务主表的读写服务。"""

    @staticmethod
    def create_task(**fields: Any) -> DBTask:
        """创建新任务。

        必填字段：task_name。
        其它可选字段按 db/db_task.py 的列约定走。
        """
        task_name = (fields.get("task_name") or "").strip()
        if not task_name:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "task_name 不能为空"
            )
        fields["task_name"] = task_name
        # 给 status 一个默认值，确保 NOT NULL 命中
        fields.setdefault("status", "PENDING")
        task = DBTask(**fields)
        with Session() as session:
            # task_name unique：若重复，merge 会覆盖（这里期望新插入失败）
            existing = (
                session.query(DBTask)
                .filter(DBTask.task_name == task_name)
                .first()
            )
            if existing is not None:
                raise RolloutInsightException(
                    RolloutInsightException.TASK_CREATE_FAILED,
                    f"task_name={task_name} 已存在",
                )
            session.add(task)
            session.commit()
            session.refresh(task)
            return task

    @staticmethod
    def get_by_id(primary_id: int) -> Optional[DBTask]:
        """按主键查任务；不存在返回 None。"""
        with Session() as session:
            return (
                session.query(DBTask)
                .filter(DBTask.primary_id == primary_id)
                .first()
            )

    @staticmethod
    def get_by_name(task_name: str) -> Optional[DBTask]:
        """按唯一名称查任务；不存在返回 None。"""
        task_name = (task_name or "").strip()
        if not task_name:
            return None
        with Session() as session:
            return (
                session.query(DBTask)
                .filter(DBTask.task_name == task_name)
                .first()
            )

    @staticmethod
    def list_tasks(
        status: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DBTask]:
        """分页列出任务；可选 status / 名称模糊过滤。"""
        with Session() as session:
            query = session.query(DBTask)
            if status:
                query = query.filter(DBTask.status == status.strip())
            kw = (keyword or "").strip()
            if kw:
                # SQLite 无 ILIKE，用 lower() 兼容大小写
                from sqlalchemy import func

                query = query.filter(
                    func.lower(DBTask.task_name).like(f"%{kw.lower()}%")
                )
            return (
                query.order_by(desc(DBTask.create_time), desc(DBTask.primary_id))
                .offset(offset)
                .limit(limit)
                .all()
            )

    @staticmethod
    def list_by_nanhu_job_ids(
        job_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """按 ``nanhu_job_id`` 批量查关联任务。

        同一 job_id 多条任务时取 ``primary_id`` 最大的一条。
        返回 ``{nanhu_job_id: {"primary_id", "task_name", "status"}}``。
        """
        cleaned = [str(j).strip() for j in (job_ids or []) if str(j or "").strip()]
        if not cleaned:
            return {}
        unique_ids = list(dict.fromkeys(cleaned))
        with Session() as session:
            rows = (
                session.query(DBTask)
                .filter(DBTask.nanhu_job_id.in_(unique_ids))
                .order_by(desc(DBTask.primary_id))
                .all()
            )
            out: Dict[str, Dict[str, Any]] = {}
            for task in rows:
                key = (task.nanhu_job_id or "").strip()
                if not key or key in out:
                    continue
                out[key] = {
                    "primary_id": task.primary_id,
                    "task_name": task.task_name or "",
                    "status": task.status or "",
                }
            return out

    @staticmethod
    def update_status(primary_id: int, status: str) -> bool:
        """更新任务状态；找不到任务返回 False。"""
        if not status:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "status 不能为空"
            )
        with Session() as session:
            task = (
                session.query(DBTask)
                .filter(DBTask.primary_id == primary_id)
                .first()
            )
            if not task:
                return False
            task.status = status
            task.update_time = datetime.now()
            session.commit()
            return True

    @staticmethod
    def update_fields(primary_id: int, **fields: Any) -> bool:
        """按字段名更新任务，未识别的字段被忽略。"""
        if not fields:
            return True
        with Session() as session:
            task = (
                session.query(DBTask)
                .filter(DBTask.primary_id == primary_id)
                .first()
            )
            if not task:
                return False
            for key, value in fields.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.update_time = datetime.now()
            session.commit()
            return True

    @staticmethod
    def delete_task(primary_id: int) -> bool:
        """按主键删除任务；外键级联由应用层负责（见 v0.2 扫描服务）。

        返回：True 已删除；False 任务不存在。
        """
        with Session() as session:
            task = (
                session.query(DBTask)
                .filter(DBTask.primary_id == primary_id)
                .first()
            )
            if not task:
                return False
            session.delete(task)
            session.commit()
            return True
