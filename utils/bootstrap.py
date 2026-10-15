#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: rollout-insight v0.5.0 启动期 bootstrap 工具。

启动期一次性执行：
1. ``DELETE FROM ri_segment_turn_table``（p17_a 决定：丢弃旧 turn 行）
2. ``ensure_admin_seed()``：若 ``ri_user_table`` 为空则插入 admin / bcrypt /
   role=admin（启动期 WARNING 一次）
3. ``Base.metadata.create_all(engine)`` 已在 main.py 里调过，这里**不**重复

用法（main.py 启动期）：
    from utils.bootstrap import bootstrap_on_startup
    bootstrap_on_startup()
"""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger


__all__ = ["bootstrap_on_startup"]


def _clear_segment_turn_table() -> int:
    """清空 ``ri_segment_turn_table``（p17_a：启动期丢弃旧 turn 行）。

    Returns:
        删除的行数；失败时返回 -1。
    """
    try:
        from db.db_element import Session
        from db.db_segment_turn import DBSegmentTurn
        from sqlalchemy import delete
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "bootstrap._clear_segment_turn_table: 不可用: {}", exc
        )
        return -1
    try:
        with Session() as session:
            stmt = delete(DBSegmentTurn)
            result = session.execute(stmt)
            session.commit()
            count = int(result.rowcount or 0)
            logger.info(
                "[bootstrap] DELETE FROM ri_segment_turn_table: rows={}",
                count,
            )
            return count
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "bootstrap._clear_segment_turn_table: DELETE 失败: {}", exc
        )
        return -1


def _ensure_admin_seed() -> Dict[str, Any]:
    """调用 ``DBUserService.ensure_admin_seed``。"""
    try:
        from db.db_user_service import ensure_admin_seed
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "bootstrap._ensure_admin_seed: db_user_service 不可用: {}", exc
        )
        return {"seeded": False, "user_count": 0, "error": str(exc)}
    try:
        result = ensure_admin_seed(log_warning=True)
        if result.get("seeded"):
            logger.info(
                "[bootstrap] ensure_admin_seed: seeded admin (rows={})",
                result.get("user_count", 0),
            )
        else:
            logger.info(
                "[bootstrap] ensure_admin_seed: 跳过（user_count={}）",
                result.get("user_count", 0),
            )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "bootstrap._ensure_admin_seed: 调用失败: {}", exc
        )
        return {"seeded": False, "user_count": 0, "error": str(exc)}


def bootstrap_on_startup() -> Dict[str, Any]:
    """v0.5.0 启动期入口：

    1. ``DELETE FROM ri_segment_turn_table`` 丢旧数据
    2. ``ensure_admin_seed()`` 若 user 表空则插 admin

    Returns:
        {
            "deleted_turn_rows": int,   # p17_a 实际清理的行数
            "admin_seeded": bool,
            "user_count": int,
        }
    """
    logger.info("[bootstrap] v0.5.0 startup bootstrap 开始")
    deleted = _clear_segment_turn_table()
    admin_result = _ensure_admin_seed()
    summary = {
        "deleted_turn_rows": int(deleted) if deleted is not None else -1,
        "admin_seeded": bool(admin_result.get("seeded")),
        "user_count": int(admin_result.get("user_count", 0) or 0),
        "username": admin_result.get("username"),
    }
    logger.info("[bootstrap] v0.5.0 startup bootstrap 结束: {}", summary)
    return summary
