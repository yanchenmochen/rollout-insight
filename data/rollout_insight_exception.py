#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: RolloutInsight 业务异常类。
错误码分段约定：

- 50xxx 通用异常
- 51xxx 任务（rollout/任务生命周期）
- 52xxx 解析（.pt dump / transcript / tokenizer）
- 53xxx 南湖（nanhu 调度、作业、镜像）
- 54xxx 系统（数据库、IO、配置等）
"""


class RolloutInsightException(Exception):
    """RolloutInsight 业务异常，支持 (code, name, message) 三元组作为 error 参数。"""

    # ----- 50xxx 通用 -----
    GENERAL_ERROR = (50000, "GENERAL_ERROR", "RolloutInsight 业务异常")
    INVALID_ARGUMENT = (50001, "INVALID_ARGUMENT", "参数非法或缺失")
    NOT_IMPLEMENTED = (50002, "NOT_IMPLEMENTED", "功能尚未实现")

    # ----- 51xxx 任务 -----
    TASK_ERROR = (51000, "TASK_ERROR", "任务处理异常")
    TASK_NOT_FOUND = (51001, "TASK_NOT_FOUND", "任务不存在")
    TASK_VALIDATION = (51002, "TASK_VALIDATION", "任务参数校验失败")
    TASK_STATE_INVALID = (51003, "TASK_STATE_INVALID", "任务状态非法，不允许该操作")
    TASK_DELETE_FORBIDDEN = (51004, "TASK_DELETE_FORBIDDEN", "任务处于运行中，禁止删除")
    TASK_CREATE_FAILED = (51005, "TASK_CREATE_FAILED", "创建任务失败")

    # ----- 52xxx 解析 -----
    PARSE_ERROR = (52000, "PARSE_ERROR", "解析处理异常")
    PARSE_DUMP_NOT_FOUND = (52001, "PARSE_DUMP_NOT_FOUND", "rollout dump 文件不存在")
    PARSE_DUMP_CORRUPTED = (52002, "PARSE_DUMP_CORRUPTED", "rollout dump 文件损坏")
    PARSE_TRANSCRIPT_FAILED = (52003, "PARSE_TRANSCRIPT_FAILED", "transcript 渲染失败")
    PARSE_SAMPLE_OUT_OF_RANGE = (52004, "PARSE_SAMPLE_OUT_OF_RANGE", "sample 索引越界")
    PARSE_TOKENIZER_FAILED = (52005, "PARSE_TOKENIZER_FAILED", "tokenizer 加载失败")

    # ----- 53xxx 南湖 -----
    NANHU_ERROR = (53000, "NANHU_ERROR", "南湖调度异常")
    NANHU_VALIDATION = (53001, "NANHU_VALIDATION", "南湖作业参数校验失败")
    NANHU_JOB_CREATE_FAILED = (53002, "NANHU_JOB_CREATE_FAILED", "南湖创建作业失败")
    NANHU_JOB_STOP_FAILED = (53003, "NANHU_JOB_STOP_FAILED", "南湖终止作业失败")
    NANHU_JOB_RESTART_FAILED = (53004, "NANHU_JOB_RESTART_FAILED", "南湖重启作业失败")
    NANHU_RESOURCE_QUERY_FAILED = (53005, "NANHU_RESOURCE_QUERY_FAILED", "南湖资源查询失败")
    NANHU_ALGORITHM_NOT_FOUND = (53006, "NANHU_ALGORITHM_NOT_FOUND", "南湖算法镜像未找到")

    # ----- 54xxx 系统 -----
    SYSTEM_ERROR = (54000, "SYSTEM_ERROR", "系统异常")
    DB_ERROR = (54001, "DB_ERROR", "数据库异常")
    DB_CONNECT_FAILED = (54002, "DB_CONNECT_FAILED", "数据库连接失败")
    IO_ERROR = (54003, "IO_ERROR", "IO 异常")
    CONFIG_ERROR = (54004, "CONFIG_ERROR", "配置异常")
    CONFIG_NOT_FOUND = (54005, "CONFIG_NOT_FOUND", "配置项未找到")

    def __init__(self, error=GENERAL_ERROR, message=None, *args: object) -> None:
        if isinstance(error, str):
            message = error
            error = self.GENERAL_ERROR
        self.code, self.name, default_message = self.get_error_info(error)
        self.message = message or default_message
        super().__init__(self.message, *args)

    def __str__(self):
        return f"[{self.code}][{self.name}] {self.message}"

    @classmethod
    def get_error_info(cls, error):
        if cls.is_error_info(error):
            return error[0], error[1], error[2]
        if isinstance(error, int):
            return cls.get_error_map().get(error, cls.GENERAL_ERROR)
        return cls.GENERAL_ERROR

    @classmethod
    def is_error_info(cls, value) -> bool:
        return (
            isinstance(value, tuple)
            and len(value) == 3
            and isinstance(value[0], int)
            and isinstance(value[1], str)
            and isinstance(value[2], str)
        )

    @classmethod
    def get_error_map(cls) -> dict:
        error_map = {}
        for error_info in cls.__dict__.values():
            if cls.is_error_info(error_info):
                error_map[error_info[0]] = error_info
        return error_map

    @classmethod
    def get_code(cls, error) -> int:
        return cls.get_error_info(error)[0]

    @classmethod
    def get_name(cls, error) -> str:
        return cls.get_error_info(error)[1]

    @classmethod
    def get_default_message(cls, error) -> str:
        return cls.get_error_info(error)[2]
