#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: SQLAlchemy engine / Session 全局入口，支持 SQLite 与 MySQL（database_url）。

仿 Shell-dsv2/automate/db/db_element.py 风格。v0.1.0 仅保留 SQLite 主路径，
MySQL / 只读模式等扩展留给后续阶段。
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from utils.config_loader import config

__all__ = ["engine", "Session", "Base", "db_path"]


def _resolve_database_uri() -> str:
    """根据 config 拼接 SQLAlchemy URI。

    优先级：
    1. config.database_url 非空 → 直接使用（MySQL 等）
    2. config.db_path → sqlite:///<db_path>
    """
    url = (config.database_url or "").strip()
    if url:
        return url
    db_path = config.db_path
    if db_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return f"sqlite:///{db_path}"
    # 默认落到 program_root/data/rollout_insight.db
    fallback = os.path.join(config.program_root, "data", "rollout_insight.db")
    os.makedirs(os.path.dirname(fallback), exist_ok=True)
    return f"sqlite:///{fallback}"


db_path = config.db_path
database_uri = _resolve_database_uri()

is_sqlite = database_uri.startswith("sqlite")

engine_kwargs = {}
if is_sqlite:
    # SQLite 单文件最大便携；check_same_thread=False 让 FastAPI 多线程可用
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(database_uri, **engine_kwargs)
Session = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)
Base = declarative_base()
