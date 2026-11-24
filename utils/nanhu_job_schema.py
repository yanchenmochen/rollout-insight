#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""为 ``ri_nanhu_job_table`` 补齐增量列（SQLite create_all 不会 ALTER）。"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from loguru import logger
from sqlalchemy import text

from db.db_element import engine

__all__ = ["ensure_nanhu_job_columns"]

# (column_name, sqlite_type_with_default)
_NANHU_JOB_EXTRA_COLUMNS: Sequence[Tuple[str, str]] = (
    ("algorithm_name", "VARCHAR DEFAULT ''"),
    ("creator", "VARCHAR DEFAULT ''"),
    ("creator_name", "VARCHAR DEFAULT ''"),
    ("algo_creator", "VARCHAR DEFAULT ''"),
    ("algo_creator_name", "VARCHAR DEFAULT ''"),
    ("is_public", "INTEGER DEFAULT 0"),
    ("shared_users_json", "TEXT DEFAULT ''"),
)


def _existing_columns(table: str) -> set:
    with engine.connect() as conn:
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    # row: cid, name, type, notnull, dflt_value, pk
    return {str(r[1]) for r in rows}


def ensure_nanhu_job_columns(
    extra_columns: Iterable[Tuple[str, str]] = _NANHU_JOB_EXTRA_COLUMNS,
) -> list:
    """若表已存在但缺列，则 ``ALTER TABLE ... ADD COLUMN``。返回新加列名。"""
    table = "ri_nanhu_job_table"
    try:
        existing = _existing_columns(table)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure_nanhu_job_columns: 读表结构失败: {}", exc)
        return []
    if not existing:
        # 表尚不存在，交给 create_all
        return []

    added: list = []
    with engine.begin() as conn:
        for col, col_type in extra_columns:
            if col in existing:
                continue
            conn.execute(
                text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            )
            added.append(col)
            logger.info("ensure_nanhu_job_columns: 已添加 {}.{}", table, col)
    return added
