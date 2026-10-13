"""rollout-insight v0.1.0 骨架基础 import smoke 测试。

不依赖外部服务、不写数据库，仅验证：
- 各子模块 import 路径无副作用
- Config 单例核心字段可读
- 错误码段定义完整
"""
import unittest


class TestImportOnly(unittest.TestCase):
    def test_import_utils(self) -> None:
        from utils import analyze_dump  # noqa: F401
        from utils.config_loader import Config, config  # noqa: F401
        from utils.log_setup import ensure_logging_configured, setup_rollout_insight_logging  # noqa: F401
        self.assertTrue(hasattr(analyze_dump, "load_dump_meta"))
        self.assertTrue(callable(ensure_logging_configured))
        self.assertTrue(callable(setup_rollout_insight_logging))

    def test_import_data(self) -> None:
        from data import Response, RolloutInsightException  # noqa: F401
        self.assertTrue(hasattr(Response, "success"))
        self.assertTrue(hasattr(Response, "error"))
        self.assertTrue(issubclass(RolloutInsightException, Exception))

    def test_import_db(self) -> None:
        from db import Base, Session, engine  # noqa: F401
        self.assertTrue(hasattr(Base, "metadata"))

    def test_import_controller(self) -> None:
        import controller  # noqa: F401
        from controller import task_controller  # noqa: F401
        self.assertTrue(hasattr(task_controller, "router"))


class TestConfig(unittest.TestCase):
    def test_config_basics(self) -> None:
        from utils.config_loader import config
        self.assertTrue(config.program_root)
        self.assertTrue(config.db_path)
        # v0.1.0 阶段版本号
        self.assertEqual(config.rollout_insight_version, "0.1.0")


class TestExceptionCodes(unittest.TestCase):
    def test_code_segments_present(self) -> None:
        """所有约定的错误码段都至少有一个示例错误码。"""
        from data.rollout_insight_exception import RolloutInsightException

        # 50xxx
        self.assertEqual(RolloutInsightException.GENERAL_ERROR[0], 50000)
        self.assertEqual(RolloutInsightException.INVALID_ARGUMENT[0], 50001)

        # 51xxx
        self.assertEqual(RolloutInsightException.TASK_ERROR[0], 51000)
        self.assertEqual(RolloutInsightException.TASK_NOT_FOUND[0], 51001)

        # 52xxx
        self.assertEqual(RolloutInsightException.PARSE_ERROR[0], 52000)
        self.assertEqual(RolloutInsightException.PARSE_DUMP_NOT_FOUND[0], 52001)

        # 53xxx
        self.assertEqual(RolloutInsightException.NANHU_ERROR[0], 53000)
        self.assertEqual(RolloutInsightException.NANHU_JOB_CREATE_FAILED[0], 53002)

        # 54xxx
        self.assertEqual(RolloutInsightException.SYSTEM_ERROR[0], 54000)
        self.assertEqual(RolloutInsightException.DB_ERROR[0], 54001)

    def test_error_info_tuple(self) -> None:
        from data.rollout_insight_exception import RolloutInsightException

        exc = RolloutInsightException(
            RolloutInsightException.TASK_NOT_FOUND, "找不到 task=42"
        )
        self.assertEqual(exc.code, 51001)
        self.assertEqual(exc.name, "TASK_NOT_FOUND")
        self.assertEqual(exc.message, "找不到 task=42")


if __name__ == "__main__":
    unittest.main()
