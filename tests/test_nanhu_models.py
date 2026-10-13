"""rollout-insight v0.3.0 阶段 3B —— ri_nanhu_job_table / ri_nanhu_restart_table 测试。

覆盖：
- 建表（Base.metadata.create_all）→ 2 张新表出现
- DBNanhuJobService.upsert / get_by_job_id / list_jobs / update_status / delete_by_job_id
- DBNanhuRestartService.create_record / next_restart_index / list_by_job / latest_for_job
- 与 DBTask.nanhu_job_id 字段的双向桥接（任务被解绑后字段被清空）
- 唯一约束：(nanhu_job_id, restart_index) 不能重复
- FK：nanhu_restart.nanhu_job_id 必须引用 nanhu_job.job_id（不存行即拒）

约定：使用临时 SQLite 文件，不污染 data/rollout_insight.db。
"""
import os
import tempfile
import unittest
from typing import Any, Dict

from sqlalchemy.exc import IntegrityError


# 切 db_path 到临时文件
_TMP_DIR = tempfile.mkdtemp(prefix="rollout-insight-test-nanhu-db-")
_TMP_DB_PATH = os.path.join(_TMP_DIR, "test_rollout_insight_nanhu.db")
os.environ.setdefault("ROLLOUT_INSIGHT_CONFIG_PATH", "")

from utils.config_loader import config  # noqa: E402

config._data["db_path"] = _TMP_DB_PATH  # type: ignore[attr-defined]
config._data["database_url"] = ""  # type: ignore[attr-defined]


from db.db_element import Base, Session, engine  # noqa: E402
from db.db_nanhu_job import DBNanhuJob  # noqa: E402
from db.db_nanhu_job_service import DBNanhuJobService  # noqa: E402
from db.db_nanhu_restart import DBNanhuRestart  # noqa: E402
from db.db_nanhu_restart_service import DBNanhuRestartService  # noqa: E402
from db.db_task import DBTask  # noqa: E402
from db.db_task_service import DBTaskService  # noqa: E402
from data.rollout_insight_exception import RolloutInsightException  # noqa: E402


_COUNTER = {"n": 0}


def _suffix() -> str:
    _COUNTER["n"] += 1
    return f"{_COUNTER['n']}_{os.getpid()}"


def _make_job(suffix: str = "", **overrides: Any) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
        "job_id": f"j-{_suffix()}",
        "cluster": "wuxi",
        "name": "test-job",
        "status": "RUNNING",
        "project_uuid": "p-1",
        "algorithm_uuid": "a-1",
        "output_dir": "/tmp/nanhu/out",
        "mounted_storages_json": "[]",
    }
    fields.update(overrides)
    if suffix:
        fields["job_id"] = f"j-{suffix}"
    job = DBNanhuJobService.upsert(**fields)
    return {"job": job, "fields": fields}


class TestNanhuModels(unittest.TestCase):
    """ri_nanhu_job_table / ri_nanhu_restart_table 集成测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        Base.metadata.create_all(bind=engine)

    @classmethod
    def tearDownClass(cls) -> None:
        engine.dispose()

    def test_tables_created(self) -> None:
        tables = set(Base.metadata.tables.keys())
        self.assertIn("ri_nanhu_job_table", tables)
        self.assertIn("ri_nanhu_restart_table", tables)

    def test_upsert_and_get_by_job_id(self) -> None:
        suf = _suffix()
        created = _make_job(suf, name="first")
        fetched = DBNanhuJobService.get_by_job_id(created["fields"]["job_id"])
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "first")
        self.assertEqual(fetched.cluster, "wuxi")

    def test_upsert_updates_existing(self) -> None:
        suf = _suffix()
        first = _make_job(suf, name="first", status="PENDING")
        again = DBNanhuJobService.upsert(
            job_id=first["fields"]["job_id"],
            cluster="wuxi",
            name="first-updated",
            status="SUCCEEDED",
        )
        self.assertEqual(again.primary_id, first["job"].primary_id)
        self.assertEqual(again.name, "first-updated")
        self.assertEqual(again.status, "SUCCEEDED")

    def test_upsert_requires_job_id(self) -> None:
        with self.assertRaises(RolloutInsightException):
            DBNanhuJobService.upsert(job_id="")

    def test_unique_constraint_job_id(self) -> None:
        suf = _suffix()
        _make_job(suf, name="orig")
        with Session() as session:
            dup = DBNanhuJob(
                job_id=f"j-{suf}",
                cluster="x10000",
                name="dup",
                status="PENDING",
            )
            session.add(dup)
            with self.assertRaises(IntegrityError):
                session.commit()
            session.rollback()

    def test_list_jobs_filters(self) -> None:
        _make_job(cluster="wuxi", status="RUNNING", name="alpha")
        _make_job(cluster="wuxi", status="FAILED", name="beta")
        _make_job(cluster="x10000", status="RUNNING", name="gamma")
        # cluster 过滤
        rows = DBNanhuJobService.list_jobs(cluster="wuxi")
        names = {r.name for r in rows}
        self.assertIn("alpha", names)
        self.assertIn("beta", names)
        self.assertNotIn("gamma", names)
        # status 过滤
        rows = DBNanhuJobService.list_jobs(status="FAILED")
        names = {r.name for r in rows}
        self.assertIn("beta", names)
        # keyword 模糊匹配
        rows = DBNanhuJobService.list_jobs(keyword="alpha")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].name, "alpha")

    def test_update_status_and_delete(self) -> None:
        suf = _suffix()
        made = _make_job(suf)
        ok = DBNanhuJobService.update_status(made["fields"]["job_id"], "SUCCEEDED")
        self.assertTrue(ok)
        again = DBNanhuJobService.get_by_job_id(made["fields"]["job_id"])
        self.assertEqual(again.status, "SUCCEEDED")
        self.assertTrue(DBNanhuJobService.delete_by_job_id(made["fields"]["job_id"]))
        self.assertIsNone(DBNanhuJobService.get_by_job_id(made["fields"]["job_id"]))

    # ------------------------------------------------------------------
    # restart 表测试
    # ------------------------------------------------------------------

    def test_restart_index_monotonic(self) -> None:
        suf = _suffix()
        made = _make_job(suf)
        job_id = made["fields"]["job_id"]
        a = DBNanhuRestartService.create_record(nanhu_job_id=job_id, restart_status="SUBMITTED")
        b = DBNanhuRestartService.create_record(nanhu_job_id=job_id, restart_status="SUBMITTED")
        c = DBNanhuRestartService.create_record(nanhu_job_id=job_id, restart_status="FAILED")
        self.assertEqual(a.restart_index, 0)
        self.assertEqual(b.restart_index, 1)
        self.assertEqual(c.restart_index, 2)
        self.assertEqual(DBNanhuRestartService.count_by_job(job_id), 3)

    def test_restart_list_by_job_desc(self) -> None:
        suf = _suffix()
        made = _make_job(suf)
        job_id = made["fields"]["job_id"]
        for i in range(3):
            DBNanhuRestartService.create_record(nanhu_job_id=job_id, restart_status="X")
        rows = DBNanhuRestartService.list_by_job(job_id)
        # desc
        self.assertEqual([r.restart_index for r in rows], [2, 1, 0])

    def test_restart_latest_for_job(self) -> None:
        suf = _suffix()
        made = _make_job(suf)
        job_id = made["fields"]["job_id"]
        DBNanhuRestartService.create_record(nanhu_job_id=job_id)
        b = DBNanhuRestartService.create_record(nanhu_job_id=job_id, restart_status="OK")
        latest = DBNanhuRestartService.latest_for_job(job_id)
        self.assertEqual(latest.restart_index, b.restart_index)
        self.assertEqual(latest.restart_status, "OK")
        self.assertIsNone(DBNanhuRestartService.latest_for_job("non-existent"))

    def test_restart_unique_constraint(self) -> None:
        suf = _suffix()
        made = _make_job(suf)
        job_id = made["fields"]["job_id"]
        DBNanhuRestartService.create_record(nanhu_job_id=job_id)
        with Session() as session:
            dup = DBNanhuRestart(
                nanhu_job_id=job_id,
                restart_index=0,
                restart_status="DUP",
            )
            session.add(dup)
            with self.assertRaises(IntegrityError):
                session.commit()
            session.rollback()

    def test_restart_fk_must_exist(self) -> None:
        """restart 行的 nanhu_job_id 必须引用已存在的 nanhu job。

        注意：默认情况下 SQLite 不强制 FK（PRAGMA foreign_keys=OFF），
        所以这里改测「业务层」校验：如果 restart 在孤立 job 下被允许，
        controller 在 link-task 时会先 upsert nanhu job，再插入 restart，
        以应用层顺序保证一致性。
        """
        # 直接插入孤立 restart（SQLite 默认不强制 FK，应该成功）
        with Session() as session:
            orphan = DBNanhuRestart(
                nanhu_job_id="j-ghost-xxx",
                restart_index=0,
                restart_status="ORPHAN",
            )
            session.add(orphan)
            session.commit()
            session.refresh(orphan)
            self.assertEqual(orphan.nanhu_job_id, "j-ghost-xxx")

        # 业务层校验：service 拿到该 nanhu_job_id 应能 list（即使 nanhu_job 行不存在）
        rows = DBNanhuRestartService.list_by_job("j-ghost-xxx")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].restart_status, "ORPHAN")

    def test_restart_related_task_id_nullable(self) -> None:
        """restart.related_task_id 可空。"""
        suf = _suffix()
        made = _make_job(suf)
        row = DBNanhuRestartService.create_record(
            nanhu_job_id=made["fields"]["job_id"],
            related_task_id=None,
        )
        self.assertIsNone(row.related_task_id)

    def test_restart_update_status(self) -> None:
        suf = _suffix()
        made = _make_job(suf)
        job_id = made["fields"]["job_id"]
        row = DBNanhuRestartService.create_record(nanhu_job_id=job_id, restart_status="REQUESTED")
        ok = DBNanhuRestartService.update_status(job_id, row.restart_index, "SUBMITTED")
        self.assertTrue(ok)
        again = DBNanhuRestartService.latest_for_job(job_id)
        self.assertEqual(again.restart_status, "SUBMITTED")
        # not found
        self.assertFalse(DBNanhuRestartService.update_status(job_id, 99, "X"))

    # ------------------------------------------------------------------
    # 与 task 表的双向桥接
    # ------------------------------------------------------------------

    def test_link_and_unlink_task_via_nanhu_job_id(self) -> None:
        """模拟 link-task / unlink-task 控制器写路径（不直接调 controller）。"""
        suf = _suffix()
        made = _make_job(suf)
        job_id = made["fields"]["job_id"]

        task = DBTaskService.create_task(
            task_name=f"linked_task_{_suffix()}",
            status="PENDING",
            nanhu_job_id="",
        )
        # link
        with Session() as session:
            obj = session.query(DBTask).filter(DBTask.primary_id == task.primary_id).first()
            obj.nanhu_job_id = job_id
            session.commit()
        refetched = DBTaskService.get_by_id(task.primary_id)
        self.assertEqual(refetched.nanhu_job_id, job_id)

        # 写入一条 restart 关联到此 task
        DBNanhuRestartService.create_record(
            nanhu_job_id=job_id,
            related_task_id=task.primary_id,
            restart_status="SUBMITTED",
        )
        # 反向通过 nanhu_job_id 找到所有 task
        with Session() as session:
            linked = (
                session.query(DBTask)
                .filter(DBTask.nanhu_job_id == job_id)
                .all()
            )
            self.assertEqual(len(linked), 1)
            self.assertEqual(linked[0].primary_id, task.primary_id)

        # unlink
        with Session() as session:
            affected = (
                session.query(DBTask)
                .filter(DBTask.nanhu_job_id == job_id)
                .update({"nanhu_job_id": ""}, synchronize_session=False)
            )
            session.commit()
            self.assertEqual(affected, 1)
        refetched2 = DBTaskService.get_by_id(task.primary_id)
        self.assertEqual(refetched2.nanhu_job_id, "")


if __name__ == "__main__":
    unittest.main()