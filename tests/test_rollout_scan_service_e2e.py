"""端到端测试：使用真实的 DB ORM + Service 跑 RolloutScanService。

使用临时 SQLite 文件，避免污染默认数据库；mock view_trajectory 让 fake .pt 能跑通。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


_TMP_DIR = tempfile.mkdtemp(prefix="rollout-insight-e2e-")
_TMP_DB_PATH = os.path.join(_TMP_DIR, "e2e.db")
_TMP_DUMP_ROOT = tempfile.mkdtemp(prefix="e2e-dump-root-")
os.environ.setdefault("ROLLOUT_INSIGHT_CONFIG_PATH", "")

# 切到临时 DB 与临时 dump root
from utils.config_loader import config  # noqa: E402

config._data["db_path"] = _TMP_DB_PATH  # type: ignore[attr-defined]
config._data["database_url"] = f"sqlite:///{_TMP_DB_PATH}"  # type: ignore[attr-defined]
config._data["rollout_dump_root"] = _TMP_DUMP_ROOT  # type: ignore[attr-defined]


# 现在 import ORM + Service，让它们绑定到临时 DB
from db.db_element import Base, Session, engine  # noqa: E402
from db.db_attempt import DBAttempt  # noqa: E402
from db.db_attempt_service import DBAttemptService  # noqa: E402
from db.db_rollout import DBRollout  # noqa: E402
from db.db_rollout_service import DBRolloutService  # noqa: E402
from db.db_segment import DBSegment  # noqa: E402
from db.db_segment_service import DBSegmentService  # noqa: E402
from db.db_task import DBTask  # noqa: E402
from db.db_task_service import DBTaskService  # noqa: E402

# 建表
Base.metadata.create_all(bind=engine)


# mock analyze_dump 工具
def _fake_meta(p):
    rollout_id = int(Path(p).stem.split("_")[-1])
    return {
        "rollout_id": rollout_id,
        "n_samples": 3,
        "n_attempts": 3,
        "mean_reward": 0.7,
        "pt_path": str(p),
    }


def _fake_rows(p):
    rollout_id = int(Path(p).stem.split("_")[-1])
    return [
        {
            "idx": i,
            "group_index": i,
            "rollout_id": rollout_id,
            "segment_index": i,
            "segment_count": 3,
            "instance_id": f"inst-{rollout_id}-{i}",
            "reward": 0.7,
            "status": "completed",
            "agent_turns": 3,
            "tool_calls": 2,
            "diff_line_count": 10,
            "response_length": 100,
            "mask_sum": 50,
            "sandbox_id": f"sx-{rollout_id}",
        }
        for i in range(3)
    ]


class TestEnd2End(unittest.TestCase):
    """用真实的 SQLAlchemy ORM + 临时 SQLite 跑端到端。"""

    @classmethod
    def setUpClass(cls) -> None:
        # mock analyze_dump
        cls._patcher = mock.patch.multiple(
            "service.rollout_scan_service",
            load_dump_meta=_fake_meta,
            summarize_dump=_fake_rows,
        )
        cls._patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._patcher.stop()

    def setUp(self) -> None:
        # 清表
        with Session() as s:
            s.query(DBSegment).delete()
            s.query(DBAttempt).delete()
            s.query(DBRollout).delete()
            s.query(DBTask).delete()
            s.commit()

        # 创建 task
        self.task = DBTaskService.create_task(
            task_name=f"e2e_task_{os.getpid()}_{id(self)}",
            status="PENDING",
        )

        # 准备临时 .pt：dump_root/<task_name>/rollout_dumps/rollout_<i>.pt
        dumps_dir = Path(_TMP_DUMP_ROOT) / self.task.task_name / "rollout_dumps"
        dumps_dir.mkdir(parents=True, exist_ok=True)
        for r in [0, 1]:
            (dumps_dir / f"rollout_{r}.pt").write_bytes(b"FAKE")

        from service.rollout_scan_service import RolloutScanService  # noqa: WPS433

        self.SvcCls = RolloutScanService

    def tearDown(self) -> None:
        with Session() as s:
            s.query(DBSegment).delete()
            s.query(DBAttempt).delete()
            s.query(DBRollout).delete()
            s.query(DBTask).delete()
            s.commit()
        # 清掉 dumps
        import shutil

        shutil.rmtree(
            Path(_TMP_DUMP_ROOT) / self.task.task_name, ignore_errors=True
        )

    def test_full_scan(self) -> None:
        svc = self.SvcCls()
        result = svc.scan_task(self.task.primary_id)
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["scanned"], 2)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["rollouts"], 2)
        self.assertEqual(result["attempts"], 6)
        self.assertEqual(result["segments"], 6)
        self.assertEqual(result["errors"], [])

        # 验证 ORM 里真的被填了数据
        with Session() as s:
            self.assertEqual(s.query(DBRollout).count(), 2)
            self.assertEqual(s.query(DBAttempt).count(), 6)
            self.assertEqual(s.query(DBSegment).count(), 6)

    def test_rescan_purges(self) -> None:
        svc = self.SvcCls()
        # 第一次扫
        svc.scan_task(self.task.primary_id)
        with Session() as s:
            self.assertEqual(s.query(DBRollout).count(), 2)
        # 重扫
        result = svc.rescan_task(self.task.primary_id)
        self.assertEqual(result["status"], "READY")
        with Session() as s:
            self.assertEqual(s.query(DBRollout).count(), 2)
            self.assertEqual(s.query(DBAttempt).count(), 6)
            self.assertEqual(s.query(DBSegment).count(), 6)

    def test_get_progress_after_scan(self) -> None:
        svc = self.SvcCls()
        svc.scan_task(self.task.primary_id)
        progress = svc.get_progress(self.task.primary_id)
        self.assertEqual(progress["rollouts"], 2)
        self.assertEqual(progress["attempts"], 6)
        self.assertEqual(progress["segments"], 6)
        self.assertEqual(progress["last_rollout_id"], 1)


if __name__ == "__main__":
    unittest.main()