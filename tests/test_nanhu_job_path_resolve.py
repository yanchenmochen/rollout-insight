"""``utils.nanhu_job_path_resolve`` 单元测试（不联调南湖）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.nanhu_job_path_resolve import (  # noqa: E402
    env_map_from_job,
    extract_job_payload,
    output_dir_from_job,
    resolve_paths_from_job_detail,
)


def _sample_job() -> dict:
    """对齐 jb-aitrain-164420415939700032 的关键字段。"""
    run = (
        "/mnt/seed-program-nas/001688/songquanheng/output/slime/runs/"
        "20261104_142408_c021-eval-f1base"
    )
    return {
        "uuid": "jb-aitrain-164420415939700032",
        "name": "c021-eval-f1base-sqh-20261104-142408",
        "outputDir": {
            "path": "/songquanheng/output/slime/runs/20261104_142408_c021-eval-f1base",
            "outputPath": run + "/",
        },
        "envVars": [
            {"key": "HF_CHECKPOINT", "value": "/mnt/models/021-32B/f1"},
            {"key": "OUTPUT_DIR", "value": run},
            {"key": "RUN_DIR", "value": run},
            {"key": "EXP_NAME", "value": "c021-eval-f1base"},
            {
                "key": "ROLLOUT_DUMP_TEMPLATE",
                "value": run + "/generated/rollout_dumps/rollout_{rollout_id}.pt",
            },
        ],
    }


class TestNanhuJobPathResolve(unittest.TestCase):
    def test_env_map_and_output_dir(self) -> None:
        job = _sample_job()
        env = env_map_from_job(job)
        self.assertEqual(env["HF_CHECKPOINT"], "/mnt/models/021-32B/f1")
        self.assertTrue(output_dir_from_job(job, env).endswith("c021-eval-f1base"))

    def test_resolve_prefers_dump_template(self) -> None:
        resolved = resolve_paths_from_job_detail(_sample_job(), prefer_existing_dirs=False)
        self.assertEqual(resolved["job_id"], "jb-aitrain-164420415939700032")
        self.assertEqual(resolved["task_name"], "c021-eval-f1base-sqh-20261104-142408")
        self.assertTrue(resolved["rollout_root"].endswith("generated/rollout_dumps"))
        self.assertEqual(resolved["tokenizer_dir"], "/mnt/models/021-32B/f1")
        self.assertEqual(resolved["warnings"], [])

    def test_resolve_fallback_without_template(self) -> None:
        job = _sample_job()
        job["envVars"] = [
            e for e in job["envVars"] if e["key"] != "ROLLOUT_DUMP_TEMPLATE"
        ]
        resolved = resolve_paths_from_job_detail(job, prefer_existing_dirs=False)
        self.assertTrue(resolved["rollout_root"].endswith("generated/rollout_dumps"))
        self.assertTrue(any("ROLLOUT_DUMP_TEMPLATE" in w for w in resolved["warnings"]))

    def test_extract_job_payload(self) -> None:
        detail = {"code": "200", "ok": True, "data": _sample_job()}
        payload = extract_job_payload(detail)
        assert payload is not None
        self.assertEqual(payload["uuid"], "jb-aitrain-164420415939700032")
        self.assertIsNone(extract_job_payload({"ok": False, "error": "作业找不到"}))
        self.assertIsNone(extract_job_payload({"error": "作业找不到", "type": "ApiError"}))


if __name__ == "__main__":
    unittest.main()
