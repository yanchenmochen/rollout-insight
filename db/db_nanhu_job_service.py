#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight DBNanhuJobService —— ri_nanhu_job_table 的 CRUD。

每行 job_id unique；首次 list_train_jobs 后批量入库，二次 list 时按 job_id 命中
已存在行只刷 status/name/output_dir 等可变字段，避免重复 INSERT。
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from db.db_element import Session
from db.db_nanhu_job import DBNanhuJob
from data.rollout_insight_exception import RolloutInsightException


class DBNanhuJobService:
    """ri_nanhu_job_table 的读写服务。"""

    @staticmethod
    def upsert(**fields: Any) -> DBNanhuJob:
        """按 job_id upsert 一个南湖作业缓存行。

        必填：job_id。
        """
        try:
            from utils.nanhu_job_schema import ensure_nanhu_job_columns

            ensure_nanhu_job_columns()
        except Exception:  # noqa: BLE001
            pass
        job_id = (fields.get("job_id") or "").strip()
        if not job_id:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "job_id 不能为空"
            )
        fields["job_id"] = job_id
        with Session() as session:
            existing: Optional[DBNanhuJob] = (
                session.query(DBNanhuJob)
                .filter(DBNanhuJob.job_id == job_id)
                .first()
            )
            if existing is None:
                row = DBNanhuJob(**fields)
                session.add(row)
                session.commit()
                session.refresh(row)
                return row
            # 已存在：合并可更新字段
            for field_name in (
                "cluster",
                "name",
                "status",
                "project_uuid",
                "algorithm_uuid",
                "algorithm_name",
                "creator",
                "creator_name",
                "algo_creator",
                "algo_creator_name",
                "shared_users_json",
                "output_dir",
                "mounted_storages_json",
            ):
                if field_name in fields:
                    setattr(existing, field_name, fields[field_name] or "")
            if "is_public" in fields:
                try:
                    existing.is_public = 1 if int(fields.get("is_public") or 0) else 0
                except (TypeError, ValueError):
                    existing.is_public = 0
            existing.update_time = datetime.now()
            session.commit()
            session.refresh(existing)
            return existing

    @staticmethod
    def get_by_job_id(job_id: str) -> Optional[DBNanhuJob]:
        """按南湖侧 job_id 查一行；不存在返回 None。"""
        job_id = (job_id or "").strip()
        if not job_id:
            return None
        with Session() as session:
            return (
                session.query(DBNanhuJob)
                .filter(DBNanhuJob.job_id == job_id)
                .first()
            )

    @staticmethod
    def get_by_id(primary_id: int) -> Optional[DBNanhuJob]:
        """按主键查一行；不存在返回 None。"""
        with Session() as session:
            return (
                session.query(DBNanhuJob)
                .filter(DBNanhuJob.primary_id == primary_id)
                .first()
            )

    @staticmethod
    def list_jobs(
        cluster: Optional[str] = None,
        status: Optional[str] = None,
        keyword: Optional[str] = None,
        creator: Optional[str] = None,
        algorithm_name: Optional[str] = None,
        job_ids: Optional[List[str]] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[DBNanhuJob]:
        """分页列出 nanhu job；可选 cluster / status / 任务名 / 创建人 / 算法库 / job_ids 过滤。

        ``job_ids`` 非空时按精确 UUID 集合过滤（批量导入共享作业后按组检索用），
        不再与 ``keyword`` 模糊条件叠加。
        """
        with Session() as session:
            from sqlalchemy import func, or_

            query = session.query(DBNanhuJob)
            if cluster:
                query = query.filter(DBNanhuJob.cluster == cluster.strip())
            if status:
                query = query.filter(DBNanhuJob.status == status.strip())
            id_list = [
                str(x).strip()
                for x in (job_ids or [])
                if str(x).strip()
            ]
            if id_list:
                query = query.filter(DBNanhuJob.job_id.in_(id_list))
            else:
                kw = (keyword or "").strip()
                if kw:
                    like = f"%{kw.lower()}%"
                    query = query.filter(
                        or_(
                            func.lower(DBNanhuJob.name).like(like),
                            func.lower(DBNanhuJob.job_id).like(like),
                        )
                    )
            creator_q = (creator or "").strip()
            if creator_q:
                like_c = f"%{creator_q.lower()}%"
                query = query.filter(
                    or_(
                        func.lower(DBNanhuJob.creator).like(like_c),
                        func.lower(DBNanhuJob.creator_name).like(like_c),
                    )
                )
            algo_q = (algorithm_name or "").strip()
            if algo_q:
                like_a = f"%{algo_q.lower()}%"
                query = query.filter(
                    or_(
                        func.lower(DBNanhuJob.algorithm_name).like(like_a),
                        func.lower(DBNanhuJob.algorithm_uuid).like(like_a),
                    )
                )
            return (
                query.order_by(DBNanhuJob.update_time.desc(), DBNanhuJob.primary_id.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

    @staticmethod
    def update_status(job_id: str, status: str) -> bool:
        """按 job_id 更新 status；找不到返回 False。"""
        job_id = (job_id or "").strip()
        if not job_id or not status:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT,
                "update_status 需要非空的 job_id 与 status",
            )
        with Session() as session:
            row = (
                session.query(DBNanhuJob)
                .filter(DBNanhuJob.job_id == job_id)
                .first()
            )
            if row is None:
                return False
            row.status = status
            row.update_time = datetime.now()
            session.commit()
            return True

    @staticmethod
    def delete_by_job_id(job_id: str) -> bool:
        """按 job_id 删除一行；不存在返回 False。"""
        job_id = (job_id or "").strip()
        if not job_id:
            return False
        with Session() as session:
            row = (
                session.query(DBNanhuJob)
                .filter(DBNanhuJob.job_id == job_id)
                .first()
            )
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True