"""rollout-insight v0.3.0 阶段 3B —— nanhu_controller 路由测试。

使用 FastAPI 的 TestClient；nanhu_client 模块未就绪时（3A 未提交），相关路由仍
应能 import 并返回友好错误（dry-run 路径 + NOT_IMPLEMENTED 错误码）。

测试覆盖：
- import & 路由注册
- GET /api/nanhu/jobs/page 渲染模板
- GET /api/nanhu/jobs 空数据返回
- POST /api/nanhu/jobs/{job_id}/link-task（dry-run / submit=true）
- POST /api/nanhu/jobs/{job_id}/unlink-task
- POST /api/nanhu/jobs/restart（dry-run / submit=true，nanhu_client 缺失时返回占位）
"""
import importlib
import json
import os
import sys
import tempfile
import unittest
from typing import Any, Dict
from unittest import mock

from fastapi.testclient import TestClient


# 切 db_path 到临时文件（沿用 v0.2 测试约定）
_TMP_DIR = tempfile.mkdtemp(prefix="rollout-insight-test-nanhu-ctrl-")
_TMP_DB_PATH = os.path.join(_TMP_DIR, "test_nanhu_ctrl.db")
os.environ.setdefault("ROLLOUT_INSIGHT_CONFIG_PATH", "")

from utils.config_loader import config  # noqa: E402

config._data["db_path"] = _TMP_DB_PATH  # type: ignore[attr-defined]
config._data["database_url"] = ""  # type: ignore[attr-defined]


# 让 service.nanhu_client 不存在时也能 import
# （3A worker 提交前 service.nanhu_client 尚未存在，3B 控制器要兼容）
def _purge_nanhu_client() -> None:
    for mod in list(sys.modules.keys()):
        if mod == "service.nanhu_client" or mod.startswith("service.nanhu_client."):
            del sys.modules[mod]


_purge_nanhu_client()


# 让 nanhu_client 模块确实不存在
if "service.nanhu_client" not in sys.modules:
    # 不主动 import，验证 controller 的延迟导入能容忍缺失
    pass


# import FastAPI app
from db.db_element import Base, Session, engine  # noqa: E402
from db.db_nanhu_job import DBNanhuJob  # noqa: E402
from db.db_nanhu_job_service import DBNanhuJobService  # noqa: E402
from db.db_task import DBTask  # noqa: E402
from db.db_task_service import DBTaskService  # noqa: E402
from controller import nanhu_controller as _ctrl  # noqa: E402
from utils.auth_session import require_login  # noqa: E402

# 重新构造一个最小的 FastAPI app，把 nanhu_controller router 挂上去（不依赖 main）
from fastapi import FastAPI  # noqa: E402
from data.response import Response  # noqa: E402
from data.rollout_insight_exception import RolloutInsightException  # noqa: E402

app = FastAPI()


class _FakeUser:
    primary_id = 1
    username = "test"


def _fake_require_login():
    return _FakeUser()


app.dependency_overrides[require_login] = _fake_require_login
app.include_router(_ctrl.router)


@app.exception_handler(RolloutInsightException)
async def _exc_handler(_request: Any, exc: RolloutInsightException):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=200,
        content=Response.error(msg=exc.message, code=exc.code).model_dump(),
    )


client = TestClient(app)


class TestNanhuController(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        Base.metadata.create_all(bind=engine)

    @classmethod
    def tearDownClass(cls) -> None:
        engine.dispose()

    def _make_task(self) -> DBTask:
        """造一个 task 用于 link/unlink 测试。"""
        import time
        suffix = f"{int(time.time() * 1000)}_{os.getpid()}"
        return DBTaskService.create_task(
            task_name=f"ctrl_task_{suffix}",
            status="PENDING",
        )

    def _make_job(self, suffix: str) -> DBNanhuJob:
        return DBNanhuJobService.upsert(
            job_id=f"j-ctrl-{suffix}",
            cluster="wuxi",
            name="ctrl-job",
            status="RUNNING",
            output_dir="/tmp/ctrl",
        )

    # ------------------------------------------------------------------
    # import / 路由
    # ------------------------------------------------------------------

    def test_router_routes(self):
        paths = [r.path for r in _ctrl.router.routes]
        self.assertIn("/api/nanhu/jobs/page", paths)
        self.assertIn("/api/nanhu/jobs", paths)
        self.assertIn("/api/nanhu/jobs/{job_id}/link-task", paths)
        self.assertIn("/api/nanhu/jobs/{job_id}/unlink-task", paths)
        self.assertIn("/api/nanhu/jobs/restart", paths)

    # ------------------------------------------------------------------
    # GET /api/nanhu/jobs/page
    # ------------------------------------------------------------------

    def test_page_nanhu_jobs(self):
        resp = client.get("/api/nanhu/jobs/page")
        self.assertEqual(resp.status_code, 200)
        body = resp.text
        self.assertIn("南湖作业", body)
        self.assertIn("filter-panel", body)

    def test_page_nanhu_jobs_with_filter(self):
        suf = "filter_x"
        self._make_job(suf)
        resp = client.get(
            "/api/nanhu/jobs/page",
            params={
                "cluster": "wuxi",
                "status": "RUNNING",
                "algo_kind": "all",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(f"j-ctrl-{suf}", resp.text)
        self.assertIn("algoKindFilter", resp.text)

    # ------------------------------------------------------------------
    # GET /api/nanhu/jobs
    # ------------------------------------------------------------------

    def test_list_nanhu_jobs_empty(self):
        resp = client.get("/api/nanhu/jobs")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["code"], 0)
        self.assertIn("jobs", data["data"])
        self.assertIn("synced", data["data"])
        self.assertIn("algo_kind", data["data"])

    def test_list_enriches_related_task_and_algo_kind(self):
        """列表应回填 related_task_* 与 algo_kind。"""
        from utils.nanhu_job_kind import SLIME_ALGORITHM_UUIDS

        suf = f"enrich_{int(__import__('time').time() * 1000)}"
        job_id = f"j-ctrl-{suf}"
        DBNanhuJobService.upsert(
            job_id=job_id,
            cluster="wuxi",
            name="slime-enrich",
            status="RUNNING",
            algorithm_uuid=next(iter(SLIME_ALGORITHM_UUIDS)),
            output_dir="/mnt/x/output/slime/runs/demo",
        )
        task = DBTaskService.create_task(
            task_name=f"ctrl_task_{suf}",
            status="PENDING",
            nanhu_job_id=job_id,
        )
        resp = client.get(
            "/api/nanhu/jobs",
            params={"algo_kind": "slime", "cluster": "wuxi"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["code"], 0)
        jobs = data["data"]["jobs"]
        hit = next((j for j in jobs if j["job_id"] == job_id), None)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["algo_kind"], "slime")
        self.assertEqual(hit["related_task_id"], task.primary_id)
        self.assertEqual(hit["related_task_name"], f"ctrl_task_{suf}")

    def test_list_nanhu_jobs_refresh_without_client(self):
        """refresh=1 时若 nanhu_client 未就绪，应返回 sync_error 提示，但仍能列已有数据。

        通过 monkey-patch controller 的 _nanhu_client_module 函数返回 None
        来模拟 3A 未提交状态。
        """
        suf = "refresh_y"
        self._make_job(suf)
        original = _ctrl._nanhu_client_module
        _ctrl._nanhu_client_module = lambda: None
        try:
            resp = client.get("/api/nanhu/jobs", params={"refresh": "1"})
        finally:
            _ctrl._nanhu_client_module = original
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["code"], 0)
        # 3A 尚未提交 → sync_error 应有提示
        self.assertIn("sync_error", data["data"])
        self.assertIn("nanhu", (data["data"]["sync_error"] or "").lower())

    # ------------------------------------------------------------------
    # POST /api/nanhu/jobs/{job_id}/link-task
    # ------------------------------------------------------------------

    def test_link_task_dry_run(self):
        suf = "lr_dry"
        self._make_job(suf)
        task = self._make_task()
        resp = client.post(
            f"/api/nanhu/jobs/j-ctrl-{suf}/link-task",
            params={"submit": "0"},
            json={"task_id": task.primary_id, "cluster": "wuxi"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["code"], 0)
        self.assertTrue(data["data"]["dry_run"])

        # DB 中 task.nanhu_job_id 应仍是 ""
        again = DBTaskService.get_by_id(task.primary_id)
        self.assertEqual(again.nanhu_job_id, "")

    def test_link_task_submit(self):
        suf = "lr_sub"
        self._make_job(suf)
        task = self._make_task()
        resp = client.post(
            f"/api/nanhu/jobs/j-ctrl-{suf}/link-task",
            params={"submit": "1"},
            json={"task_id": task.primary_id, "cluster": "wuxi"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["code"], 0, msg=data)
        self.assertFalse(data["data"]["dry_run"])
        again = DBTaskService.get_by_id(task.primary_id)
        self.assertEqual(again.nanhu_job_id, f"j-ctrl-{suf}")

    def test_link_task_submit_with_restart(self):
        """submit=true 且传 restart_index，应同时写 ri_nanhu_restart_table。"""
        suf = "lr_sub2"
        self._make_job(suf)
        task = self._make_task()
        resp = client.post(
            f"/api/nanhu/jobs/j-ctrl-{suf}/link-task",
            params={"submit": "1"},
            json={"task_id": task.primary_id, "restart_index": 5},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["code"], 0)
        self.assertIsNotNone(data["data"].get("restart"))
        # 应用层按业务约定使用 append-only 序号；这里 controller 透传了用户传的 5
        self.assertEqual(data["data"]["restart"]["restart_index"], 5)

    def test_link_task_missing_job(self):
        """nanhu job 未入库时 submit=true 应抛 NANHU_VALIDATION。"""
        task = self._make_task()
        resp = client.post(
            "/api/nanhu/jobs/j-NONEXISTENT/link-task",
            params={"submit": "1"},
            json={"task_id": task.primary_id},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertNotEqual(data["code"], 0)
        # 错误码 NANHU_VALIDATION = 53001
        self.assertEqual(data["code"], RolloutInsightException.NANHU_VALIDATION[0])

    def test_link_task_missing_task(self):
        suf = "lr_mt"
        self._make_job(suf)
        resp = client.post(
            f"/api/nanhu/jobs/j-ctrl-{suf}/link-task",
            params={"submit": "1"},
            json={"task_id": 99999999},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertNotEqual(data["code"], 0)
        self.assertEqual(data["code"], RolloutInsightException.TASK_NOT_FOUND[0])

    def test_link_task_invalid_task_id(self):
        resp = client.post(
            "/api/nanhu/jobs/j-anything/link-task",
            params={"submit": "1"},
            json={"task_id": "not-a-number"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertNotEqual(data["code"], 0)
        self.assertEqual(data["code"], 50001)

    # ------------------------------------------------------------------
    # POST /api/nanhu/jobs/{job_id}/unlink-task
    # ------------------------------------------------------------------

    def test_unlink_task_dry_run(self):
        """dry-run 不应该真的清空。"""
        suf = "ul_dry"
        self._make_job(suf)
        task = self._make_task()
        with Session() as s:
            obj = s.query(DBTask).filter(DBTask.primary_id == task.primary_id).first()
            obj.nanhu_job_id = f"j-ctrl-{suf}"
            s.commit()

        resp = client.post(
            f"/api/nanhu/jobs/j-ctrl-{suf}/unlink-task",
            params={"submit": "0"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["data"]["dry_run"])
        again = DBTaskService.get_by_id(task.primary_id)
        self.assertEqual(again.nanhu_job_id, f"j-ctrl-{suf}")

    def test_unlink_task_submit(self):
        suf = "ul_sub"
        self._make_job(suf)
        task = self._make_task()
        with Session() as s:
            obj = s.query(DBTask).filter(DBTask.primary_id == task.primary_id).first()
            obj.nanhu_job_id = f"j-ctrl-{suf}"
            s.commit()

        resp = client.post(
            f"/api/nanhu/jobs/j-ctrl-{suf}/unlink-task",
            params={"submit": "1"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["code"], 0)
        self.assertEqual(data["data"]["affected_tasks"], 1)
        again = DBTaskService.get_by_id(task.primary_id)
        self.assertEqual(again.nanhu_job_id, "")

    # ------------------------------------------------------------------
    # POST /api/nanhu/jobs/restart
    # ------------------------------------------------------------------

    def test_restart_dry_run_without_client(self):
        """nanhu_client 缺失时 dry-run 应仍能返回占位结果。"""
        original = _ctrl._nanhu_client_module
        _ctrl._nanhu_client_module = lambda: None
        try:
            resp = client.post(
                "/api/nanhu/jobs/restart",
                params={"submit": "0"},
                json={"job_id": "j-NOOP", "cluster": "wuxi"},
            )
        finally:
            _ctrl._nanhu_client_module = original
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["code"], 0)
        self.assertTrue(data["data"]["dry_run"])
        self.assertFalse(data["data"]["client_ready"])

    def test_restart_submit_without_client_raises(self):
        """nanhu_client 缺失时 submit=true 应抛 NANHU_ERROR。"""
        original = _ctrl._nanhu_client_module
        _ctrl._nanhu_client_module = lambda: None
        try:
            resp = client.post(
                "/api/nanhu/jobs/restart",
                params={"submit": "1"},
                json={"job_id": "j-NOOP", "cluster": "wuxi"},
            )
        finally:
            _ctrl._nanhu_client_module = original
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertNotEqual(data["code"], 0)
        self.assertEqual(data["code"], RolloutInsightException.NANHU_ERROR[0])

    def test_restart_missing_job_id(self):
        resp = client.post(
            "/api/nanhu/jobs/restart",
            params={"submit": "0"},
            json={"cluster": "wuxi"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertNotEqual(data["code"], 0)
        self.assertEqual(data["code"], 50001)

    # ------------------------------------------------------------------
    # 与真实 nanhu_client 集成（dry-run 始终不调 CLI）
    # ------------------------------------------------------------------

    def test_restart_dry_run_with_real_nanhu_client(self):
        """当 nanhu_client 模块已就绪时，dry-run 仍然不真正提交。

        不期望触发任何 _run_cli 调用；restart_job 模块自身的 dry-run 分支
        应被命中。
        """
        import service.nanhu_client as nc
        with mock.patch.object(nc, "_run_cli") as run_cli_spy:
            resp = client.post(
                "/api/nanhu/jobs/restart",
                params={"submit": "0"},
                json={"job_id": "j-NOOP", "cluster": "wuxi"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["code"], 0)
            self.assertTrue(data["data"]["dry_run"])
            self.assertTrue(data["data"]["client_ready"])
            # dry-run 不应触发 _run_cli
            run_cli_spy.assert_not_called()

    def test_list_jobs_refresh_with_real_nanhu_client(self):
        """refresh=1 时若 nanhu_client 已在，应该调 list_train_jobs 并 upsert。

        即使 nanhu CLI 不可用 / 无凭据，list_train_jobs 应返回 []，
        controller 不应抛异常。
        """
        import service.nanhu_client as nc
        with mock.patch.object(nc, "list_train_jobs", return_value=[
            {
                    "job_id": "j-MOCK-1",
                    "name": "mock-job",
                    "status": "RUNNING",
                    "cluster": "wuxi",
                    "project_uuid": "p-mock",
                    "algorithm_uuid": "a-mock",
                    "output_dir": "/tmp/mock",
                    "mounted_storages": [{"name": "data"}],
                }
            ]) as list_spy:
            resp = client.get(
                "/api/nanhu/jobs",
                params={"refresh": "1", "cluster": "wuxi"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["code"], 0)
            list_spy.assert_called_once()
            self.assertGreaterEqual(data["data"]["synced"], 1)
            # upsert 后能查到
            job_ids = [j["job_id"] for j in data["data"]["jobs"]]
            self.assertIn("j-MOCK-1", job_ids)


if __name__ == "__main__":
    unittest.main()