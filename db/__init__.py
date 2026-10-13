"""db 包：SQLAlchemy 模型与会话入口。

v0.2.0 起 5 张 ORM 全部暴露在外面，方便 app 启动时 Base.metadata.create_all 一把建表。
"""
from db.db_element import Base, Session, engine
from db.db_task import DBTask
from db.db_rollout import DBRollout
from db.db_attempt import DBAttempt
from db.db_segment import DBSegment
from db.db_segment_transcript import DBSegmentTranscript

__all__ = [
    "Base",
    "Session",
    "engine",
    # ORM 模型
    "DBTask",
    "DBRollout",
    "DBAttempt",
    "DBSegment",
    "DBSegmentTranscript",
]
