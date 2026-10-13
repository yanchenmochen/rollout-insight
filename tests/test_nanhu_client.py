"""``service.nanhu_client`` 的单元测试。

策略：完全 mock 掉 ``subprocess.run`` / ``shutil.which`` / 凭据检查，
不联调真实集群，也不依赖 nanhu-skill 源码存在。

覆盖：
- ``list_train_jobs`` 解析正常 stdout JSON
- ``list_train_jobs`` 凭据缺失时返回 ``[]`` 且写 audit
- ``list_train_jobs`` 不可用（无 nanhu 二进制）时返回 ``[]``
- ``get_job_detail`` 直接透传 CLI JSON
- ``get_job_detail`` nanhu 不可用时返回 ``hint`` 响应
- ``create_train_job`` dry-run（submit=False）不发请求
- ``create_train_job`` yaml 不存在时返回 hint
- ``create_train_job`` submit=True 走 CLI 并解析
- ``create_train_job`` 凭据缺失时返回友好提示
- ``restart_job`` dry-run 不发请求
- ``restart_job`` 真实调用走 CLI
- ``module_diagnostics`` 字段完整
- audit log 文件被写入
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock


# 确保 service package 可被 import
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# 用临时目录隔离 audit 日志
_TMP_LOG_DIR = Path(tempfile.mkdtemp(prefix="nanhu_client_test_"))
from utils.config_loader import config  # noqa: E402

config._data["log_dir"] = str(_TMP_LOG_DIR)  # type: ignore[attr-defined]
config._data["nanhu_cluster"] = "wuxi"  # type: ignore[attr-defined]
config._data["nanhu_user_id"] = ""  # type: ignore[attr-defined]


from service import nanhu_client  # noqa: E402


def _ok_cli_payload(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """构造一个 nanhu CLI 风格的正常响应。"""
    return {
        "code": "200",
        "message": "SUCCESS",
        "data": {
            "page": {"current": 1, "pageSize": 20, "total": len(items)},
            "result": items,
        },
    }


def _ok_detail_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "code": "200",
        "message": "SUCCESS",
        "data": item,
    }


class _FakeCompletedProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class TestListTrainJobs(unittest.TestCase):
    """``list_train_jobs`` 的多种返回结构。"""

    def setUp(self) -> None:
        # 清空 audit log 文件，便于每个 case 独立断言
        audit = Path(config.log_dir) / nanhu_client.AUDIT_LOG_NAME
        if audit.exists():
            audit.unlink()

    def test_returns_parsed_list(self) -> None:
        items = [
            {"uuid": "jb-1", "name": "train-job-1", "status": "running"},
            {"uuid": "jb-2", "name": "train-job-2", "status": "failed"},
        ]
        payload = _ok_cli_payload(items)
        with mock.patch.object(
            nanhu_client,
            "_credentials_configured",
            return_value=True,
        ), mock.patch.object(
            nanhu_client.shutil,
            "which",
            return_value="/usr/local/bin/nanhu",
        ), mock.patch.object(
            nanhu_client,
            "_try_import_nanhu_skill",
            return_value=None,
        ), mock.patch.object(
            nanhu_client.subprocess,
            "run",
            return_value=_FakeCompletedProcess(stdout=json.dumps(payload), returncode=0),
        ) as run_mock:
            result = nanhu_client.list_train_jobs(cluster="wuxi", status="running")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["uuid"], "jb-1")
        # 确认传了 --cluster/--format/json/jobs/list
        called_cmd = run_mock.call_args.args[0]
        self.assertIn("nanhu", called_cmd[0])
        self.assertIn("--cluster", called_cmd)
        self.assertIn("wuxi", called_cmd)
        self.assertIn("--format", called_cmd)
        self.assertIn("json", called_cmd)
        self.assertIn("jobs", called_cmd)
        self.assertIn("list", called_cmd)
        self.assertIn("--status", called_cmd)
        self.assertIn("running", called_cmd)

    def test_missing_credentials_returns_empty(self) -> None:
        with mock.patch.object(
            nanhu_client, "_credentials_configured", return_value=False
        ), mock.patch.object(
            nanhu_client.shutil, "which", return_value="/usr/local/bin/nanhu"
        ), mock.patch.object(
            nanhu_client, "_try_import_nanhu_skill", return_value=None
        ), mock.patch.object(
            nanhu_client.subprocess, "run"
        ) as run_mock:
            result = nanhu_client.list_train_jobs(cluster="wuxi")

        self.assertEqual(result, [])
        run_mock.assert_not_called()
        # audit 必须写一条
        audit = Path(config.log_dir) / nanhu_client.AUDIT_LOG_NAME
        self.assertTrue(audit.exists())
        lines = audit.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertEqual(entry["result"], "no_creds")
        self.assertEqual(entry["action"], "jobs list")

    def test_unavailable_returns_empty(self) -> None:
        with mock.patch.object(nanhu_client, "_resolve_nanhu_binary", return_value=None), \
             mock.patch.object(nanhu_client, "_try_import_nanhu_skill", return_value=None), \
             mock.patch.object(nanhu_client.subprocess, "run") as run_mock:
            result = nanhu_client.list_train_jobs(cluster="wuxi")

        self.assertEqual(result, [])
        run_mock.assert_not_called()

    def test_status_and_user_id_passed_through(self) -> None:
        payload = _ok_cli_payload([])
        with mock.patch.object(
            nanhu_client, "_credentials_configured", return_value=True
        ), mock.patch.object(
            nanhu_client.shutil, "which", return_value="/usr/local/bin/nanhu"
        ), mock.patch.object(
            nanhu_client, "_try_import_nanhu_skill", return_value=None
        ), mock.patch.object(
            nanhu_client.subprocess,
            "run",
            return_value=_FakeCompletedProcess(stdout=json.dumps(payload), returncode=0),
        ) as run_mock:
            nanhu_client.list_train_jobs(cluster="x10000", status="failed", user_id="1783329701595")

        cmd = run_mock.call_args.args[0]
        self.assertIn("x10000", cmd)
        self.assertIn("--status", cmd)
        self.assertIn("failed", cmd)

    def test_cli_failure_returns_empty(self) -> None:
        # nanhu CLI 失败 -> 返回 {}
        with mock.patch.object(
            nanhu_client, "_credentials_configured", return_value=True
        ), mock.patch.object(
            nanhu_client.shutil, "which", return_value="/usr/local/bin/nanhu"
        ), mock.patch.object(
            nanhu_client, "_try_import_nanhu_skill", return_value=None
        ), mock.patch.object(
            nanhu_client.subprocess,
            "run",
            return_value=_FakeCompletedProcess(stdout="", stderr="boom", returncode=1),
        ):
            result = nanhu_client.list_train_jobs(cluster="wuxi")

        self.assertEqual(result, [])

    def test_uses_config_defaults(self) -> None:
        """cluster 和 user_id 为空时，应该从 config 读取。"""
        config._data["nanhu_cluster"] = "x10000"  # type: ignore[attr-defined]
        config._data["nanhu_user_id"] = ""  # type: ignore[attr-defined]
        payload = _ok_cli_payload([])
        with mock.patch.object(
            nanhu_client, "_credentials_configured", return_value=True
        ), mock.patch.object(
            nanhu_client.shutil, "which", return_value="/usr/local/bin/nanhu"
        ), mock.patch.object(
            nanhu_client, "_try_import_nanhu_skill", return_value=None
        ), mock.patch.object(
            nanhu_client.subprocess,
            "run",
            return_value=_FakeCompletedProcess(stdout=json.dumps(payload), returncode=0),
        ) as run_mock:
            nanhu_client.list_train_jobs()  # 全默认

        cmd = run_mock.call_args.args[0]
        self.assertIn("x10000", cmd)


class TestGetJobDetail(unittest.TestCase):
    def setUp(self) -> None:
        audit = Path(config.log_dir) / nanhu_client.AUDIT_LOG_NAME
        if audit.exists():
            audit.unlink()

    def test_returns_detail(self) -> None:
        detail = {"uuid": "jb-1", "name": "train-job-1", "status": "running"}
        payload = _ok_detail_payload(detail)
        with mock.patch.object(
            nanhu_client, "_credentials_configured", return_value=True
        ), mock.patch.object(
            nanhu_client.shutil, "which", return_value="/usr/local/bin/nanhu"
        ), mock.patch.object(
            nanhu_client, "_try_import_nanhu_skill", return_value=None
        ), mock.patch.object(
            nanhu_client.subprocess,
            "run",
            return_value=_FakeCompletedProcess(stdout=json.dumps(payload), returncode=0),
        ) as run_mock:
            result = nanhu_client.get_job_detail(cluster="wuxi", job_id="jb-1")

        # 透传 CLI dict
        self.assertEqual(result["data"]["uuid"], "jb-1")
        called_cmd = run_mock.call_args.args[0]
        self.assertIn("detail", called_cmd)
        self.assertIn("--job-id", called_cmd)
        self.assertIn("jb-1", called_cmd)

    def test_unavailable_returns_hint(self) -> None:
        with mock.patch.object(nanhu_client, "_resolve_nanhu_binary", return_value=None), \
             mock.patch.object(nanhu_client, "_try_import_nanhu_skill", return_value=None):
            result = nanhu_client.get_job_detail(cluster="wuxi", job_id="jb-1")

        self.assertEqual(result.get("ok"), False)
        self.assertEqual(result.get("hint"), "nanhu skill not available")

    def test_missing_creds_returns_hint(self) -> None:
        with mock.patch.object(
            nanhu_client, "_credentials_configured", return_value=False
        ), mock.patch.object(
            nanhu_client.shutil, "which", return_value="/usr/local/bin/nanhu"
        ), mock.patch.object(
            nanhu_client, "_try_import_nanhu_skill", return_value=None
        ):
            result = nanhu_client.get_job_detail(cluster="wuxi", job_id="jb-1")

        self.assertEqual(result.get("ok"), False)
        self.assertEqual(result.get("hint"), "nanhu credentials not configured")


class TestCreateTrainJob(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_yaml = Path(tempfile.mkdtemp(prefix="nanhu_yaml_")) / "job.yaml"
        self.tmp_yaml.write_text("name: test\n", encoding="utf-8")
        audit = Path(config.log_dir) / nanhu_client.AUDIT_LOG_NAME
        if audit.exists():
            audit.unlink()

    def tearDown(self) -> None:
        try:
            self.tmp_yaml.parent.rmdir()
        except OSError:
            pass

    def test_dry_run_default(self) -> None:
        """submit 默认 False 时不应该真正调用 CLI。"""
        with mock.patch.object(
            nanhu_client, "_credentials_configured", return_value=True
        ), mock.patch.object(
            nanhu_client.shutil, "which", return_value="/usr/local/bin/nanhu"
        ), mock.patch.object(
            nanhu_client, "_try_import_nanhu_skill", return_value=None
        ), mock.patch.object(
            nanhu_client.subprocess, "run"
        ) as run_mock:
            result = nanhu_client.create_train_job(
                cluster="wuxi", yaml_path=str(self.tmp_yaml)
            )

        self.assertTrue(result.get("dry_run"))
        self.assertEqual(result.get("cluster"), "wuxi")
        self.assertFalse(result.get("ok") is False)  # ok 应为 True
        run_mock.assert_not_called()

    def test_yaml_missing_returns_hint(self) -> None:
        result = nanhu_client.create_train_job(
            cluster="wuxi", yaml_path="/nonexistent/job.yaml"
        )
        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("hint"), "yaml file not found")
        self.assertIn("/nonexistent/job.yaml", result.get("message", ""))

    def test_submit_true_invokes_cli(self) -> None:
        payload = _ok_detail_payload({"jobId": "new-job-1"})
        with mock.patch.object(
            nanhu_client, "_credentials_configured", return_value=True
        ), mock.patch.object(
            nanhu_client.shutil, "which", return_value="/usr/local/bin/nanhu"
        ), mock.patch.object(
            nanhu_client, "_try_import_nanhu_skill", return_value=None
        ), mock.patch.object(
            nanhu_client.subprocess,
            "run",
            return_value=_FakeCompletedProcess(stdout=json.dumps(payload), returncode=0),
        ) as run_mock:
            result = nanhu_client.create_train_job(
                cluster="wuxi", yaml_path=str(self.tmp_yaml), submit=True
            )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result["data"]["jobId"], "new-job-1")
        called_cmd = run_mock.call_args.args[0]
        self.assertIn("create-train", called_cmd)
        self.assertIn("--config", called_cmd)
        self.assertIn(str(self.tmp_yaml), called_cmd)

    def test_submit_true_no_creds_returns_hint(self) -> None:
        with mock.patch.object(
            nanhu_client, "_credentials_configured", return_value=False
        ), mock.patch.object(
            nanhu_client.shutil, "which", return_value="/usr/local/bin/nanhu"
        ), mock.patch.object(
            nanhu_client, "_try_import_nanhu_skill", return_value=None
        ):
            result = nanhu_client.create_train_job(
                cluster="wuxi", yaml_path=str(self.tmp_yaml), submit=True
            )

        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("hint"), "nanhu credentials not configured")

    def test_submit_true_unavailable_returns_hint(self) -> None:
        with mock.patch.object(
            nanhu_client, "_resolve_nanhu_binary", return_value=None
        ), mock.patch.object(
            nanhu_client, "_try_import_nanhu_skill", return_value=None
        ):
            result = nanhu_client.create_train_job(
                cluster="wuxi", yaml_path=str(self.tmp_yaml), submit=True
            )

        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("hint"), "nanhu skill not available")


class TestRestartJob(unittest.TestCase):
    def setUp(self) -> None:
        audit = Path(config.log_dir) / nanhu_client.AUDIT_LOG_NAME
        if audit.exists():
            audit.unlink()

    def test_dry_run_default(self) -> None:
        with mock.patch.object(
            nanhu_client, "_credentials_configured", return_value=True
        ), mock.patch.object(
            nanhu_client.shutil, "which", return_value="/usr/local/bin/nanhu"
        ), mock.patch.object(
            nanhu_client, "_try_import_nanhu_skill", return_value=None
        ), mock.patch.object(
            nanhu_client.subprocess, "run"
        ) as run_mock:
            result = nanhu_client.restart_job(cluster="wuxi", job_id="jb-1")

        self.assertTrue(result.get("dry_run"))
        self.assertEqual(result.get("job_id"), "jb-1")
        run_mock.assert_not_called()

    def test_submit_true_invokes_cli(self) -> None:
        payload = _ok_detail_payload({"jobId": "jb-1", "restarted": True})
        with mock.patch.object(
            nanhu_client, "_credentials_configured", return_value=True
        ), mock.patch.object(
            nanhu_client.shutil, "which", return_value="/usr/local/bin/nanhu"
        ), mock.patch.object(
            nanhu_client, "_try_import_nanhu_skill", return_value=None
        ), mock.patch.object(
            nanhu_client.subprocess,
            "run",
            return_value=_FakeCompletedProcess(stdout=json.dumps(payload), returncode=0),
        ) as run_mock:
            result = nanhu_client.restart_job(
                cluster="wuxi", job_id="jb-1", submit=True
            )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result["data"]["restarted"], True)
        called_cmd = run_mock.call_args.args[0]
        self.assertIn("restart", called_cmd)
        self.assertIn("--job-id", called_cmd)
        self.assertIn("jb-1", called_cmd)

    def test_no_creds_returns_hint(self) -> None:
        with mock.patch.object(
            nanhu_client, "_credentials_configured", return_value=False
        ), mock.patch.object(
            nanhu_client.shutil, "which", return_value="/usr/local/bin/nanhu"
        ), mock.patch.object(
            nanhu_client, "_try_import_nanhu_skill", return_value=None
        ):
            result = nanhu_client.restart_job(
                cluster="wuxi", job_id="jb-1", submit=True
            )

        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("hint"), "nanhu credentials not configured")


class TestAuditLog(unittest.TestCase):
    def test_audit_log_written_on_real_call(self) -> None:
        audit_path = Path(config.log_dir) / nanhu_client.AUDIT_LOG_NAME
        if audit_path.exists():
            audit_path.unlink()

        items = [{"uuid": "jb-1"}]
        payload = _ok_cli_payload(items)
        with mock.patch.object(
            nanhu_client, "_credentials_configured", return_value=True
        ), mock.patch.object(
            nanhu_client.shutil, "which", return_value="/usr/local/bin/nanhu"
        ), mock.patch.object(
            nanhu_client, "_try_import_nanhu_skill", return_value=None
        ), mock.patch.object(
            nanhu_client.subprocess,
            "run",
            return_value=_FakeCompletedProcess(stdout=json.dumps(payload), returncode=0),
        ):
            nanhu_client.list_train_jobs(cluster="wuxi", status="running")

        self.assertTrue(audit_path.exists())
        text = audit_path.read_text(encoding="utf-8").strip()
        self.assertTrue(text)
        entry = json.loads(text.splitlines()[-1])
        self.assertEqual(entry["action"], "jobs list")
        self.assertEqual(entry["cluster"], "wuxi")
        self.assertEqual(entry["status"], "running")
        self.assertIn("ts", entry)


class TestModuleDiagnostics(unittest.TestCase):
    def test_diagnostics_has_all_keys(self) -> None:
        diag = nanhu_client.module_diagnostics()
        for key in (
            "nanhu_skill_path",
            "nanhu_skill_path_exists",
            "nanhu_skill_import_ok",
            "nanhu_binary",
            "credentials_configured",
            "audit_log_path",
            "config_cluster",
            "config_user_id",
        ):
            self.assertIn(key, diag)


if __name__ == "__main__":
    unittest.main()