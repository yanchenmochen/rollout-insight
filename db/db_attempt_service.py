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
            # 已存在：合并可更新字段。
            # 同一 attempt 的多 segment 会反复 upsert；response_length / trainable_tokens 累加，
            # agent_turns / tool_calls 等会话级字段保持覆盖（各 segment 上数值相同）。
            for field_name in (
                "sandbox_id",
                "instance_id",
                "status",
                "reward",
                "agent_turns",
                "tool_calls",
                "diff_line_count",
                "finalized",
            ):
                if field_name in fields:
                    setattr(existing, field_name, fields[field_name])
            if "response_length" in fields:
                existing.response_length = int(existing.response_length or 0) + int(
                    fields.get("response_length") or 0
                )
            if "trainable_tokens" in fields:
                existing.trainable_tokens = int(existing.trainable_tokens or 0) + int(
                    fields.get("trainable_tokens") or 0
                )
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

    # abort / 不可测：与 rollout_parse_service / summarize_groups 对齐
    _ABORT_STATUSES = frozenset(
        {"aborted", "abort", "cancelled", "canceled", "invalid", "failed_infra"}
    )
    _TIMEOUT_STATUSES = frozenset({"truncated", "timeout"})
    _REWARD_EPS = 1e-9

    @classmethod
    def _status_token(cls, status: Any) -> str:
        return str(status or "").split(".")[-1].lower()

    @classmethod
    def _classify_attempt_row(cls, status: Any, reward: Any) -> str:
        """把一行 attempt 归为 pass / fail / abort / timeout / other。"""
        token = cls._status_token(status)
        if token in cls._ABORT_STATUSES:
            return "abort"
        if token in cls._TIMEOUT_STATUSES:
            return "timeout"
        if isinstance(reward, (int, float)):
            rv = float(reward)
            if abs(rv - 1.0) <= cls._REWARD_EPS:
                return "pass"
            if abs(rv) <= cls._REWARD_EPS:
                return "fail"
        return "other"

    @classmethod
    def summarize_reward_metrics_by_rollout(
        cls, task_id: int
    ) -> Dict[int, Dict[str, int]]:
        """按业务键 ``rollout_id`` 聚合 attempt 分布（对齐 TB generated/paired/filtered）。

        返回 ``{rollout_id: {n_passed, n_aborted, n_timeout, n_failed, n_other,
        n_valid_groups, n_attempts, n_reward_1, n_reward_0, n_no_reward}}``：

        - ``n_passed``：可测且 reward≈1（= generated 分子）
        - ``n_aborted``：abort / 不可测
        - ``n_timeout``：``status`` 为 truncated/timeout（优先于 reward）
        - ``n_failed``：可测且 reward≈0 的正常失败
        - ``n_valid_groups``：去掉 abort/timeout 后仍「有过有挂」的 group 数
        - ``n_reward_1`` / ``n_reward_0``：含超时上的可测奖励（超时也可能有奖）
        - ``n_no_reward``：abort 等不可测（置灰段）
        """
        if task_id is None:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT, "task_id 不能为空"
            )

        with Session() as session:
            rows = (
                session.query(
                    DBAttempt.rollout_id,
                    DBAttempt.group_index,
                    DBAttempt.status,
                    DBAttempt.reward,
                )
                .filter(DBAttempt.task_id == int(task_id))
                .all()
            )

        by_rollout: Dict[int, Dict[str, Any]] = {}
        for rollout_id, group_index, status, reward in rows:
            if rollout_id is None:
                continue
            rid = int(rollout_id)
            bucket = by_rollout.setdefault(
                rid,
                {
                    "n_passed": 0,
                    "n_aborted": 0,
                    "n_timeout": 0,
                    "n_failed": 0,
                    "n_other": 0,
                    "n_attempts": 0,
                    "n_reward_1": 0,
                    "n_reward_0": 0,
                    "n_no_reward": 0,
                    "groups": {},
                },
            )
            bucket["n_attempts"] += 1
            cat = cls._classify_attempt_row(status, reward)
            if cat == "abort":
                bucket["n_aborted"] += 1
                bucket["n_no_reward"] += 1
                continue
            if cat == "timeout":
                bucket["n_timeout"] += 1
                if isinstance(reward, (int, float)):
                    rv = float(reward)
                    if abs(rv - 1.0) <= cls._REWARD_EPS:
                        bucket["n_reward_1"] += 1
                    elif abs(rv) <= cls._REWARD_EPS:
                        bucket["n_reward_0"] += 1
                    else:
                        bucket["n_no_reward"] += 1
                else:
                    bucket["n_no_reward"] += 1
                continue
            gmap = bucket["groups"].setdefault(
                int(group_index) if group_index is not None else -1,
                {"has_one": False, "has_zero": False},
            )
            if cat == "pass":
                bucket["n_passed"] += 1
                bucket["n_reward_1"] += 1
                gmap["has_one"] = True
            elif cat == "fail":
                bucket["n_failed"] += 1
                bucket["n_reward_0"] += 1
                gmap["has_zero"] = True
            else:
                bucket["n_other"] += 1
                if isinstance(reward, (int, float)):
                    rv = float(reward)
                    if abs(rv - 1.0) <= cls._REWARD_EPS:
                        bucket["n_reward_1"] += 1
                    elif abs(rv) <= cls._REWARD_EPS:
                        bucket["n_reward_0"] += 1
                    else:
                        bucket["n_no_reward"] += 1
                else:
                    bucket["n_no_reward"] += 1

        out: Dict[int, Dict[str, int]] = {}
        for rid, bucket in by_rollout.items():
            n_valid = sum(
                1
                for g in bucket["groups"].values()
                if g["has_one"] and g["has_zero"]
            )
            out[rid] = {
                "n_passed": int(bucket["n_passed"]),
                "n_aborted": int(bucket["n_aborted"]),
                "n_timeout": int(bucket["n_timeout"]),
                "n_failed": int(bucket["n_failed"]),
                "n_other": int(bucket["n_other"]),
                "n_valid_groups": int(n_valid),
                "n_attempts": int(bucket["n_attempts"]),
                "n_reward_1": int(bucket["n_reward_1"]),
                "n_reward_0": int(bucket["n_reward_0"]),
                "n_no_reward": int(bucket["n_no_reward"]),
            }
        return out

    @classmethod
    def summarize_reward_metrics_by_group(
        cls, task_id: int, rollout_id: int
    ) -> Dict[int, Dict[str, Any]]:
        """按 ``group_index`` 聚合单次 rollout 的 attempt 分布。

        返回 ``{group_index: {n_passed, n_aborted, n_timeout, n_failed, n_other,
        n_attempts, is_valid, mean_reward, instance_id}}``。
        """
        if task_id is None or rollout_id is None:
            raise RolloutInsightException(
                RolloutInsightException.INVALID_ARGUMENT,
                "task_id / rollout_id 不能为空",
            )

        with Session() as session:
            rows = (
                session.query(
                    DBAttempt.group_index,
                    DBAttempt.status,
                    DBAttempt.reward,
                    DBAttempt.instance_id,
                )
                .filter(
                    and_(
                        DBAttempt.task_id == int(task_id),
                        DBAttempt.rollout_id == int(rollout_id),
                    )
                )
                .all()
            )

        by_group: Dict[int, Dict[str, Any]] = {}
        for group_index, status, reward, instance_id in rows:
            if group_index is None:
                continue
            gi = int(group_index)
            bucket = by_group.setdefault(
                gi,
                {
                    "n_passed": 0,
                    "n_aborted": 0,
                    "n_timeout": 0,
                    "n_failed": 0,
                    "n_other": 0,
                    "n_attempts": 0,
                    "reward_sum": 0.0,
                    "reward_n": 0,
                    "has_one": False,
                    "has_zero": False,
                    "instance_id": "",
                },
            )
            bucket["n_attempts"] += 1
            if not bucket["instance_id"] and instance_id:
                bucket["instance_id"] = str(instance_id)
            cat = cls._classify_attempt_row(status, reward)
            if cat == "abort":
                bucket["n_aborted"] += 1
                continue
            if cat == "timeout":
                bucket["n_timeout"] += 1
                continue
            if isinstance(reward, (int, float)):
                bucket["reward_sum"] += float(reward)
                bucket["reward_n"] += 1
            if cat == "pass":
                bucket["n_passed"] += 1
                bucket["has_one"] = True
            elif cat == "fail":
                bucket["n_failed"] += 1
                bucket["has_zero"] = True
            else:
                bucket["n_other"] += 1

        out: Dict[int, Dict[str, Any]] = {}
        for gi, bucket in by_group.items():
            mean_reward = (
                bucket["reward_sum"] / bucket["reward_n"]
                if bucket["reward_n"]
                else None
            )
            out[gi] = {
                "n_passed": int(bucket["n_passed"]),
                "n_aborted": int(bucket["n_aborted"]),
                "n_timeout": int(bucket["n_timeout"]),
                "n_failed": int(bucket["n_failed"]),
                "n_other": int(bucket["n_other"]),
                "n_attempts": int(bucket["n_attempts"]),
                "is_valid": bool(bucket["has_one"] and bucket["has_zero"]),
                "mean_reward": mean_reward,
                "instance_id": bucket["instance_id"],
            }
        return out
