"""db 包：SQLAlchemy 模型与会话入口。

v0.2.0 起 5 张 ORM 全部暴露在外面，方便 app 启动时 Base.metadata.create_all 一把建表。
v0.3.0 起新增 2 张 nanhu 相关表（job / restart）。
v0.4.0 起新增 1 张 turn 表（ri_segment_turn_table）—— 规范化 turn 行。
"""
from db.db_element import Base, Session, engine
from db.db_task import DBTask
from db.db_rollout import DBRollout
from db.db_attempt import DBAttempt
from db.db_segment import DBSegment
from db.db_segment_transcript import DBSegmentTranscript
from db.db_segment_turn import DBSegmentTurn
from db.db_nanhu_job import DBNanhuJob
from db.db_nanhu_restart import DBNanhuRestart

# 别名：让 controller 通过 db.models.Task / db.Task 也能找到（兼容 v0.2 阶段 2C 的延迟导入）
Task = DBTask
Rollout = DBRollout
Attempt = DBAttempt
Segment = DBSegment
SegmentTranscript = DBSegmentTranscript
SegmentTurn = DBSegmentTurn
NanhuJob = DBNanhuJob
NanhuRestart = DBNanhuRestart

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
    "DBSegmentTurn",
    "DBNanhuJob",
    "DBNanhuRestart",
    # 兼容别名
    "Task",
    "Rollout",
    "Attempt",
    "Segment",
    "SegmentTranscript",
    "SegmentTurn",
    "NanhuJob",
    "NanhuRestart",
]
