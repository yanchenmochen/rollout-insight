"""
配置加载工具：基于 program_root 计算所有相对路径，并提供全局单例 Config。

仿 Shell-dsv2/automate/utils/config_loader.py 风格，但只暴露 rollout-insight
v0.1.0 所需的字段。后续阶段可继续向 Config 类追加字段。
"""
import json
import os
from typing import Dict, Optional


_DEFAULT_CONFIG_ENV = "ROLLOUT_INSIGHT_CONFIG_PATH"


def _resolve_config_path(config_path: Optional[str] = None) -> str:
    """解析配置文件路径优先级：
    1. 显式传入的 config_path
    2. 环境变量 ROLLOUT_INSIGHT_CONFIG_PATH
    3. 当前模块同目录上两级下的 config.json（即 <program_root>/config.json）
    """
    if config_path:
        return config_path
    env_path = os.environ.get(_DEFAULT_CONFIG_ENV)
    if env_path:
        return env_path
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "config.json")


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    加载配置文件，并基于 program_root 计算其他路径。
    如果未指定 config_path，将按 _resolve_config_path 的策略寻找默认路径。
    """
    cfg_path = _resolve_config_path(config_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 获取 program_root
    program_root = config.get("program_root")
    if not program_root:
        raise ValueError("program_root is required in config.json")

    # 确保 program_root 是绝对路径
    if not os.path.isabs(program_root):
        program_root = os.path.abspath(program_root)
    config["program_root"] = program_root

    # 基于 program_root 推导常用路径
    config.setdefault("data_dir", os.path.join(program_root, "data"))
    config.setdefault("log_dir", os.path.join(program_root, "log"))
    config.setdefault("static_dir", os.path.join(program_root, "static"))
    config.setdefault("templates_dir", os.path.join(program_root, "templates"))

    # SQLite db_path：如果未设置，给一个 data 下的默认路径
    if not config.get("db_path"):
        config["db_path"] = os.path.join(program_root, "data", "rollout_insight.db")

    # 配置 rollout dump 扫描的默认根目录
    config.setdefault("rollout_dump_root", "")
    config.setdefault("rollout_dump_pattern", "**/rollout_dumps/rollout_*.pt")
    config.setdefault("rollout_insight_version", "0.1.0")
    config.setdefault("rollout_insight_title", "Rollout Insight")

    # v0.3.0 nanhu-skill 集成默认配置
    config.setdefault("nanhu_cluster", "wuxi")
    config.setdefault("nanhu_user_id", "")

    return config


class Config:
    """全局配置单例。

    使用方式：
        from utils.config_loader import config
        db_path = config.db_path
    """

    _instance: Optional["Config"] = None

    def __init__(self, data: Dict):
        self._data = data

    @classmethod
    def instance(cls, config_path: Optional[str] = None) -> "Config":
        """获取全局 Config 实例。"""
        if cls._instance is None or config_path:
            data = load_config(config_path)
            cls._instance = cls(data)
        return cls._instance

    # 通用访问
    def get(self, key: str, default: Optional[str] = None):
        return self._data.get(key, default)

    def get_int(self, key: str, default: int) -> int:
        raw = self._data.get(key, default)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return int(default)

    def get_float(self, key: str, default: float) -> float:
        raw = self._data.get(key, default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(default)

    @property
    def program_root(self) -> str:
        return self._data["program_root"]

    @property
    def db_path(self) -> str:
        return self._data.get("db_path", "")

    @property
    def database_url(self) -> str:
        return self._data.get("database_url", "")

    @property
    def data_dir(self) -> str:
        return self._data.get("data_dir", "")

    @property
    def log_dir(self) -> str:
        return self._data.get("log_dir", "")

    @property
    def static_dir(self) -> str:
        return self._data.get("static_dir", "")

    @property
    def templates_dir(self) -> str:
        return self._data.get("templates_dir", "")

    @property
    def rollout_dump_root(self) -> str:
        return self._data.get("rollout_dump_root", "")

    @property
    def rollout_dump_pattern(self) -> str:
        return self._data.get("rollout_dump_pattern", "**/rollout_dumps/rollout_*.pt")

    @property
    def rollout_insight_version(self) -> str:
        return self._data.get("rollout_insight_version", "0.1.0")

    @property
    def rollout_insight_title(self) -> str:
        return self._data.get("rollout_insight_title", "Rollout Insight")

    @property
    def nanhu_cluster(self) -> str:
        return self._data.get("nanhu_cluster", "wuxi")

    @property
    def nanhu_user_id(self) -> str:
        return self._data.get("nanhu_user_id", "")


config = Config.instance()


if __name__ == "__main__":
    print("program_root: ", config.program_root)
    print("db_path: ", config.db_path)
    print("log_dir: ", config.log_dir)
    print("rollout_dump_root: ", config.rollout_dump_root)
    print("rollout_dump_pattern: ", config.rollout_dump_pattern)
