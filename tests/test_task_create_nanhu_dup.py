"""创建任务时 nanhu_job_id 已关联应返回 51006。"""
import os
import tempfile
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

_TMP = tempfile.mkdtemp(prefix="rollout-insight-test-task-dup-")
_TMP_DB = os.path.join(_TMP, "test.db")
os.environ.setdefault("ROLLOUT_INSIGHT_CONFIG_PATH", "")

from utils.config_loader import config  # noqa: E402

config._data["db_path"] = _TMP_DB  # type: ignore[attr-defined]
config._data["database_url"] = ""  # type: ignore[attr-defined]

from data.response import Response  # noqa: E402
from data.rollout_insight_exception import RolloutInsightException  # noqa: E402
from db.db_element import Base, Session, engine  # noqa: E402
from db.db_task import DBTask  # noqa: E402
from controller import task_controller as tc  # noqa: E402
from utils.auth_session import require_login  # noqa: E402


class _FakeUser:
    primary_id = 1
    username = "tester"


app = FastAPI()
app.dependency_overrides[require_login] = lambda: _FakeUser()
app.include_router(tc.router)


@app.exception_handler(RolloutInsightException)
async def _exc_handler(_request, exc: RolloutInsightException):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=200,
        content=Response.error(msg=exc.message, code=exc.code).model_dump(),
    )


client = TestClient(app)


class TestTaskCreateNanhuDup(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        Base.metadata.create_all(bind=engine)

    def test_duplicate_nanhu_job_id_returns_51006(self):
        jid = "jb-aitrain-dup-create-1"
        with Session() as session:
            session.add(
                DBTask(
                    task_name="already",
                    rollout_root="/tmp/r",
                    nanhu_job_id=jid,
                    source_type="nanhu_job",
                    status="PENDING",
                )
            )
            session.commit()

        resp = client.post(
            "/api/tasks",
            data={
                "nanhu_job_id": jid,
                "source_type": "nanhu_job",
                "cluster": "x10000",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["code"], 51006)
        self.assertIn("已关联", data["msg"])

    def test_create_new_job_ok_when_not_linked(self):
        jid = "jb-aitrain-new-create-9"
        with mock.patch.object(
            tc,
            "_resolve_create_fields_from_nanhu",
            return_value={
                "task_name": "auto-name",
                "rollout_root": "/tmp/rollout",
                "tokenizer_dir": "/tmp/tok",
            },
        ), mock.patch.object(tc, "current_user", return_value=_FakeUser()):
            resp = client.post(
                "/api/tasks",
                data={
                    "nanhu_job_id": jid,
                    "source_type": "nanhu_job",
                    "cluster": "x10000",
                },
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["code"], 0, data)
        self.assertTrue(data["data"].get("id") or data["data"].get("task_id"))

    def test_nanhu_create_auto_suffix_on_task_name_conflict(self):
        """南湖自动解析时，若作业名已占用，应自动加 job_id 尾缀而不是抛 UNIQUE。"""
        jid = "jb-aitrain-165272097743459392"
        with Session() as session:
            session.add(
                DBTask(
                    task_name="c021-e34-20261123-093932",
                    rollout_root="/tmp/old",
                    nanhu_job_id="",
                    source_type="manual",
                    status="PENDING",
                )
            )
            session.commit()

        with mock.patch.object(
            tc,
            "_resolve_create_fields_from_nanhu",
            return_value={
                "task_name": "c021-e34-20261123-093932",
                "rollout_root": "/tmp/rollout",
                "tokenizer_dir": "/tmp/tok",
            },
        ), mock.patch.object(tc, "current_user", return_value=_FakeUser()):
            resp = client.post(
                "/api/tasks",
                data={
                    "nanhu_job_id": jid,
                    "source_type": "nanhu_job",
                    "cluster": "x10000",
                },
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["code"], 0, data)
        self.assertIn("165272097743459392"[:12], data["data"].get("task_name", ""))
        self.assertTrue(data["data"].get("name_note"))


if __name__ == "__main__":
    unittest.main()
