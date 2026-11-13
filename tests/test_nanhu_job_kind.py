"""``utils.nanhu_job_kind`` 与列表 enrichment 单测。"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TMP_DIR = tempfile.mkdtemp(prefix="rollout-insight-test-nanhu-kind-")
_TMP_DB_PATH = os.path.join(_TMP_DIR, "test_nanhu_kind.db")
os.environ.setdefault("ROLLOUT_INSIGHT_CONFIG_PATH", "")

from utils.config_loader import config  # noqa: E402

config._data["db_path"] = _TMP_DB_PATH  # type: ignore[attr-defined]
config._data["database_url"] = ""  # type: ignore[attr-defined]

from db.db_element import Base, engine  # noqa: E402
from db.db_task_service import DBTaskService  # noqa: E402
from utils.nanhu_job_kind import (  # noqa: E402
    CONVERT_ALGORITHM_UUIDS,
    EVAL_ALGORITHM_UUIDS,
    SLIME_ALGORITHM_UUIDS,
    attach_nanhu_job_kind,
    classify_nanhu_job_kind,
)
from controller.nanhu_controller import (  # noqa: E402
    enrich_jobs_with_task_links,
    prepare_nanhu_job_list,
)


class _FakeJob:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestNanhuJobKind(unittest.TestCase):
    def test_slime_by_uuid(self) -> None:
        uuid = next(iter(SLIME_ALGORITHM_UUIDS))
        out = classify_nanhu_job_kind({"algorithm_uuid": uuid, "name": "x"})
        self.assertEqual(out["algo_kind"], "slime")
        self.assertEqual(out["algo_label"], "slime")

    def test_slime_by_output_path(self) -> None:
        out = classify_nanhu_job_kind(
            {
                "algorithm_uuid": "",
                "output_dir": "/mnt/seed/output/slime/runs/demo",
                "name": "demo",
            }
        )
        self.assertEqual(out["algo_kind"], "slime")

    def test_convert_by_uuid_and_name(self) -> None:
        uuid = next(iter(CONVERT_ALGORITHM_UUIDS))
        self.assertEqual(
            classify_nanhu_job_kind({"algorithm_uuid": uuid})["algo_kind"],
            "convert",
        )
        self.assertEqual(
            classify_nanhu_job_kind({"name": "convert-hf2torch-demo"})["algo_kind"],
            "convert",
        )
        self.assertEqual(
            classify_nanhu_job_kind({"name": "convert-hf2torch-demo"})["algo_label"],
            "转换",
        )

    def test_eval_by_uuid_and_path(self) -> None:
        for uuid in EVAL_ALGORITHM_UUIDS:
            self.assertEqual(
                classify_nanhu_job_kind({"algorithm_uuid": uuid})["algo_kind"],
                "eval",
                msg=uuid,
            )
        self.assertEqual(
            classify_nanhu_job_kind(
                {"output_dir": "/data/TransEval/mmlu", "name": "mmlu"}
            )["algo_kind"],
            "eval",
        )
        self.assertEqual(
            classify_nanhu_job_kind(
                {"output_dir": "/mnt/foo/Eval/bar", "name": "bar"}
            )["algo_kind"],
            "eval",
        )
        self.assertEqual(
            classify_nanhu_job_kind({"output_dir": "/data/TransEval/x"})["algo_label"],
            "评测",
        )

    def test_other_and_attach(self) -> None:
        out = classify_nanhu_job_kind({"name": "random-job", "output_dir": "/tmp"})
        self.assertEqual(out["algo_kind"], "other")
        job = {"name": "random-job", "output_dir": "/tmp"}
        attach_nanhu_job_kind(job)
        self.assertEqual(job["algo_kind"], "other")
        self.assertEqual(job["algo_label"], "其它")

    def test_slime_priority_over_convert_name(self) -> None:
        """路径含 /slime/ 时优先判为 slime（即使名称像 convert）。"""
        out = classify_nanhu_job_kind(
            {
                "name": "convert-lookalike",
                "output_dir": "/mnt/x/output/slime/runs/a",
            }
        )
        self.assertEqual(out["algo_kind"], "slime")


class TestEnrichRelatedTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        Base.metadata.create_all(bind=engine)

    def test_list_by_nanhu_job_ids_and_enrich(self) -> None:
        suf = f"{int(time.time() * 1000)}_{os.getpid()}"
        job_id = f"jb-kind-{suf}"
        task = DBTaskService.create_task(
            task_name=f"kind_task_{suf}",
            status="PENDING",
            nanhu_job_id=job_id,
        )
        mapping = DBTaskService.list_by_nanhu_job_ids([job_id, "missing"])
        self.assertIn(job_id, mapping)
        self.assertEqual(mapping[job_id]["primary_id"], task.primary_id)
        self.assertEqual(mapping[job_id]["task_name"], f"kind_task_{suf}")

        jobs = [
            {
                "job_id": job_id,
                "name": "slime-demo",
                "algorithm_uuid": next(iter(SLIME_ALGORITHM_UUIDS)),
                "output_dir": "/tmp",
            },
            {
                "job_id": "unlinked-job",
                "name": "other",
                "algorithm_uuid": "",
                "output_dir": "/tmp",
            },
        ]
        enrich_jobs_with_task_links(jobs)
        self.assertEqual(jobs[0]["related_task_id"], task.primary_id)
        self.assertEqual(jobs[0]["related_task_name"], f"kind_task_{suf}")
        self.assertIsNone(jobs[1]["related_task_id"])

    def test_prepare_filters_algo_kind(self) -> None:
        slime_uuid = next(iter(SLIME_ALGORITHM_UUIDS))
        rows = [
            _FakeJob(
                primary_id=1,
                job_id="j-slime",
                cluster="wuxi",
                name="s",
                status="RUNNING",
                project_uuid="",
                algorithm_uuid=slime_uuid,
                output_dir="/tmp",
                mounted_storages_json="",
                create_time=None,
                update_time=None,
            ),
            _FakeJob(
                primary_id=2,
                job_id="j-eval",
                cluster="wuxi",
                name="e",
                status="RUNNING",
                project_uuid="",
                algorithm_uuid=next(iter(EVAL_ALGORITHM_UUIDS)),
                output_dir="/tmp",
                mounted_storages_json="",
                create_time=None,
                update_time=None,
            ),
        ]
        only_slime = prepare_nanhu_job_list(rows, algo_kind="slime")
        self.assertEqual(len(only_slime), 1)
        self.assertEqual(only_slime[0]["job_id"], "j-slime")
        self.assertEqual(only_slime[0]["algo_kind"], "slime")

        all_jobs = prepare_nanhu_job_list(rows, algo_kind="all")
        self.assertEqual(len(all_jobs), 2)


if __name__ == "__main__":
    unittest.main()
