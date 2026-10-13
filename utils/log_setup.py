"""基于 loguru 的统一日志配置。"""
import logging
import sys
from pathlib import Path

from loguru import logger

from utils.config_loader import config

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{file.name}:{function}:{line} | pid={process.id} tid={thread.id} | {message}"
)
LOG_ROTATION = "5 MB"
LOG_RETENTION = "30 days"
ROTATION_BYTES = 5 * 1024 * 1024

_LOGGING_ENV_FLAG = "ROLLOUT_INSIGHT_LOGGING_CONFIGURED"


class InterceptHandler(logging.Handler):
    """桥接 stdlib logging 到 loguru。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_rollout_insight_logging(log_base: str = "") -> None:
    """配置 rollout-insight 的 loguru 输出。

    - 日志目录：config.log_dir（默认 <program_root>/log）
    - stderr + 文件双 sink；文件按大小轮转
    """
    log_dir = Path(log_base) if log_base else Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    active_log = log_dir / "rollout-insight.log"

    logger.remove()
    logger.add(
        sys.stderr,
        format=LOG_FORMAT,
        level="INFO",
        enqueue=False,
    )
    logger.add(
        str(active_log),
        format=LOG_FORMAT,
        level="INFO",
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
        compression=None,
        encoding="utf-8",
        enqueue=True,
    )

    # 桥接 stdlib logging（uvicorn、sqlalchemy 等）
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(logging.INFO)


def ensure_logging_configured() -> None:
    """幂等地初始化 logging，避免被反复执行。"""
    import os

    if os.environ.get(_LOGGING_ENV_FLAG) == "1":
        return
    setup_rollout_insight_logging()
    os.environ[_LOGGING_ENV_FLAG] = "1"
