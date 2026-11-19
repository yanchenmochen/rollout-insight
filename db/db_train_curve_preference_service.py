#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight DBTrainCurvePreferenceService —— ri_train_curve_preference_table 的 CRUD。

所有方法 @staticmethod；写操作用 `with Session() as session:` 短事务。
"""
import json
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, or_

from db.db_element import Session
from db.db_train_curve_preference import DBTrainCurvePreference
from data.rollout_insight_exception import RolloutInsightException

__all__ = ["DBTrainCurvePreferenceService"]

SCOPE_GLOBAL = "global"
SCOPE_TASK = "task"


class DBTrainCurvePreferenceService:
    """训练曲线指标偏好的读写服务。"""

    @staticmethod
    def _dump_list(values: Optional[List[str]]) -> str:
        return json.dumps([str(v) for v in (values or [])], ensure_ascii=False)

    @staticmethod
    def get_global(user_id: int) -> Optional[DBTrainCurvePreference]:
        """取用户的全局默认偏好；没有返回 None。"""
        if user_id is None:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "user_id 不能为空"
            )
        with Session() as session:
            return (
                session.query(DBTrainCurvePreference)
                .filter(
                    and_(
                        DBTrainCurvePreference.user_id == int(user_id),
                        DBTrainCurvePreference.scope == SCOPE_GLOBAL,
                    )
                )
                .first()
            )

    @staticmethod
    def get_for_task(user_id: int, task_id: int) -> Optional[DBTrainCurvePreference]:
        """取用户在某任务上的追加偏好；没有返回 None。"""
        if user_id is None:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "user_id 不能为空"
            )
        with Session() as session:
            return (
                session.query(DBTrainCurvePreference)
                .filter(
                    and_(
                        DBTrainCurvePreference.user_id == int(user_id),
                        DBTrainCurvePreference.scope == SCOPE_TASK,
                        DBTrainCurvePreference.task_id == int(task_id),
                    )
                )
                .first()
            )

    @staticmethod
    def upsert_global(
        user_id: int,
        *,
        keys: Optional[List[str]] = None,
        custom_keys: Optional[List[str]] = None,
    ) -> DBTrainCurvePreference:
        """upsert 用户全局默认偏好。"""
        return DBTrainCurvePreferenceService._upsert(
            user_id=int(user_id),
            scope=SCOPE_GLOBAL,
            task_id=None,
            keys=keys,
            custom_keys=custom_keys,
        )

    @staticmethod
    def upsert_for_task(
        user_id: int,
        task_id: int,
        *,
        keys: Optional[List[str]] = None,
        custom_keys: Optional[List[str]] = None,
    ) -> DBTrainCurvePreference:
        """upsert 用户在某任务上的追加偏好。"""
        return DBTrainCurvePreferenceService._upsert(
            user_id=int(user_id),
            scope=SCOPE_TASK,
            task_id=int(task_id),
            keys=keys,
            custom_keys=custom_keys,
        )

    @staticmethod
    def delete_global(user_id: int) -> bool:
        """删除用户全局默认偏好，回到内置默认按钮组。"""
        with Session() as session:
            count = (
                session.query(DBTrainCurvePreference)
                .filter(
                    and_(
                        DBTrainCurvePreference.user_id == int(user_id),
                        DBTrainCurvePreference.scope == SCOPE_GLOBAL,
                    )
                )
                .delete(synchronize_session=False)
            )
            session.commit()
            return int(count or 0) > 0

    @staticmethod
    def delete_for_task(user_id: int, task_id: int) -> bool:
        """删除用户在某任务上的追加偏好。"""
        with Session() as session:
            count = (
                session.query(DBTrainCurvePreference)
                .filter(
                    and_(
                        DBTrainCurvePreference.user_id == int(user_id),
                        DBTrainCurvePreference.scope == SCOPE_TASK,
                        DBTrainCurvePreference.task_id == int(task_id),
                    )
                )
                .delete(synchronize_session=False)
            )
            session.commit()
            return int(count or 0) > 0

    @staticmethod
    def list_all_for_user(user_id: int) -> List[Dict[str, Any]]:
        """列出用户的全部偏好（全局 + 各任务追加）。"""
        if user_id is None:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "user_id 不能为空"
            )
        with Session() as session:
            rows = (
                session.query(DBTrainCurvePreference)
                .filter(DBTrainCurvePreference.user_id == int(user_id))
                .order_by(
                    DBTrainCurvePreference.scope.asc(),
                    DBTrainCurvePreference.task_id.asc(),
                )
                .all()
            )
        return [row.to_dict() for row in rows]

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _upsert(
        *,
        user_id: int,
        scope: str,
        task_id: Optional[int],
        keys: Optional[List[str]],
        custom_keys: Optional[List[str]],
    ) -> DBTrainCurvePreference:
        if user_id is None:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "user_id 不能为空"
            )
        with Session() as session:
            conditions = [
                DBTrainCurvePreference.user_id == int(user_id),
                DBTrainCurvePreference.scope == scope,
            ]
            if task_id is None:
                conditions.append(DBTrainCurvePreference.task_id.is_(None))
            else:
                conditions.append(DBTrainCurvePreference.task_id == int(task_id))
            row = session.query(DBTrainCurvePreference).filter(and_(*conditions)).first()

            if row is None:
                row = DBTrainCurvePreference(
                    user_id=int(user_id),
                    scope=scope,
                    task_id=task_id,
                    keys_json=DBTrainCurvePreferenceService._dump_list(keys),
                    custom_keys_json=DBTrainCurvePreferenceService._dump_list(custom_keys),
                )
                session.add(row)
            else:
                if keys is not None:
                    row.keys_json = DBTrainCurvePreferenceService._dump_list(keys)
                if custom_keys is not None:
                    row.custom_keys_json = DBTrainCurvePreferenceService._dump_list(
                        custom_keys
                    )
            session.commit()
            session.refresh(row)
            return row
