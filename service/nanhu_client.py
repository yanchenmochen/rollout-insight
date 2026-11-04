"""NanhuClient 封装：通过 ``nanhu`` / ``nanhu-cli`` CLI 调用南湖 OpenAPI。

设计目标（rollout-insight v0.5.x）：

- **不重写 OpenAPI 客户端**：鉴权 / token 缓存等细节由 nanhu CLI 负责。
- **凭据来自数据库**：``ri_user_cluster_config_table``（用户在
  ``/api/me/cluster-configs`` 录入）；调用 CLI 时注入 ``NANHU_APP_KEY`` /
  ``NANHU_APP_SECRET`` / ``NANHU_USER_ID`` / ``NANHU_API_BASE`` 环境变量。
  **不再**依赖硬编码的 nanhu-skill 源码路径或 ``credentials.env`` 文件。
- **CLI 二进制**：从 ``PATH`` 查找 ``nanhu`` / ``nanhu-cli``；也可
  ``import nanhu``（若已 pip install）。
- **dry-run 默认**：``create_train_job`` / ``restart_job`` 默认 ``submit=False``。
- **缺凭据返回** ``{"hint": "...", "ok": False}``，由 controller 友好展示。

对外暴露：
    - ``list_train_jobs(cluster, status, user_id, local_user_id)``
    - ``get_job_detail(cluster, job_id, local_user_id)``
    - ``create_train_job(cluster, yaml_path, submit, local_user_id)``
    - ``restart_job(cluster, job_id, submit, local_user_id)``
    - ``module_diagnostics()``
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from utils.config_loader import config

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 审计日志文件名（位于 config.log_dir 下）
AUDIT_LOG_NAME = "nanhu_audit.log"

# 写操作的安全子命令集合（默认 dry-run；必须显式 submit=True 才真正下发）
_WRITE_ACTIONS = {
    "create-train",
    "create-deploy",
    "restart",
    "restart-deploy",
    "restart-pod",
}


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _audit_log_path() -> Path:
    """返回审计日志文件路径，并保证目录存在。"""
    base = Path(config.log_dir or str(Path(__file__).resolve().parents[1] / "log"))
    base.mkdir(parents=True, exist_ok=True)
    return base / AUDIT_LOG_NAME


def _write_audit(entry: Dict[str, Any]) -> None:
    """追加一条审计记录（JSON Lines 格式）。"""
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()), **entry}
    try:
        with _audit_log_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str))
            fh.write("\n")
    except OSError as exc:  # pragma: no cover - 写日志失败不影响主流程
        logger.warning("nanhu_audit.log 写入失败: {}", exc)


def _try_import_nanhu_skill() -> Optional[Any]:
    """尝试 import 已安装的 ``nanhu`` 包（不依赖任何硬编码路径）。"""
    try:
        return importlib.import_module("nanhu")
    except Exception:  # pragma: no cover
        return None


def _resolve_nanhu_binary() -> Optional[str]:
    """查找 ``nanhu`` / ``nanhu-cli`` 二进制路径；找不到返回 ``None``。"""
    for name in ("nanhu", "nanhu-cli"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _nanhu_available() -> bool:
    """nanhu 是否可用（已安装包或 CLI 二进制至少一种）。"""
    if _try_import_nanhu_skill() is not None:
        return True
    return _resolve_nanhu_binary() is not None


def _credential_env_from_db(
    cluster: str = "",
    local_user_id: Optional[int] = None,
) -> Optional[Dict[str, str]]:
    """从 ``ri_user_cluster_config_table`` 解析凭据，返回可注入 CLI 的 env 片段。

    成功时至少包含 ``NANHU_APP_KEY`` / ``NANHU_APP_SECRET``；有则附带
    ``NANHU_USER_ID`` / ``NANHU_API_BASE``。
    """
    name = (cluster or "").strip() or (config.nanhu_cluster or "").strip()
    if not name:
        return None
    try:
        from db.db_user_cluster_config_service import DBUserClusterConfigService

        row = DBUserClusterConfigService.resolve_for_cluster(
            name, user_id=local_user_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("解析南湖 DB 凭据失败 cluster={}: {}", name, exc)
        return None
    if row is None:
        return None

    app_key = (getattr(row, "app_key", None) or "").strip()
    app_secret = (getattr(row, "app_secret", None) or "").strip()
    if not app_key or not app_secret:
        return None

    env: Dict[str, str] = {
        "NANHU_APP_KEY": app_key,
        "NANHU_APP_SECRET": app_secret,
    }
    user_id_field = (getattr(row, "user_id_field", None) or "").strip()
    if user_id_field:
        env["NANHU_USER_ID"] = user_id_field
    api_base = (getattr(row, "api_base_url", None) or "").strip().rstrip("/")
    if api_base:
        env["NANHU_API_BASE"] = api_base
    return env


def _env_has_nanhu_creds() -> bool:
    """进程环境里是否已有南湖 app 凭据（运维临时覆盖用）。"""
    return bool(
        (os.environ.get("NANHU_APP_KEY") or "").strip()
        and (os.environ.get("NANHU_APP_SECRET") or "").strip()
    )


def _credentials_configured(
    cluster: str = "",
    local_user_id: Optional[int] = None,
) -> bool:
    """凭据是否可用：优先 DB 集群配置，其次进程环境变量。"""
    if _credential_env_from_db(cluster, local_user_id):
        return True
    return _env_has_nanhu_creds()


def _missing_credentials_response(action: str) -> Dict[str, Any]:
    """凭据未配置时的友好提示响应。"""
    return {
        "ok": False,
        "action": action,
        "hint": "nanhu credentials not configured",
        "message": (
            "未配置南湖凭据：请登录后在 /api/me/cluster-configs 录入集群网关、"
            "app_key / app_secret / userId，保存后再试。"
        ),
    }


def _unavailable_response(action: str) -> Dict[str, Any]:
    """nanhu CLI 不可用时的友好提示响应。"""
    return {
        "ok": False,
        "action": action,
        "hint": "nanhu skill not available",
        "message": (
            "未找到 nanhu / nanhu-cli 可执行文件：请先 "
            "`pip install -e <nanhu-skill 仓库路径>` 并保证 PATH 可找到。"
        ),
    }


# ---------------------------------------------------------------------------
# CLI 调用
# ---------------------------------------------------------------------------


def _run_cli(
    args: List[str],
    timeout: int = 60,
    *,
    cluster: str = "",
    local_user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """通过 ``nanhu`` 子进程调用命令并解析 stdout JSON。

    :param args: 不包含 ``nanhu`` 二进制名的参数列表。
    :param timeout: 超时秒数。
    :param cluster: 用于从 DB 解析凭据并注入 ``NANHU_*`` 环境变量。
    :param local_user_id: 本地登录用户主键，优先匹配其集群配置。
    :returns: 解析后的 dict；解析失败时返回 ``{"ok": False, ...}``。
    """
    binary = _resolve_nanhu_binary()
    if not binary:
        return {
            "ok": False,
            "error": "nanhu binary not found",
            "hint": "ensure nanhu CLI is installed (pip install -e <nanhu-skill>)",
        }

    env = os.environ.copy()
    cred_env = _credential_env_from_db(cluster, local_user_id)
    if cred_env:
        env.update(cred_env)

    cmd = [binary, *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "nanhu CLI timeout", "cmd": cmd}
    except OSError as exc:
        return {"ok": False, "error": f"nanhu CLI OSError: {exc}", "cmd": cmd}

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if not stdout:
        return {
            "ok": proc.returncode == 0,
            "error": stderr or f"nanhu CLI exit={proc.returncode}, empty stdout",
            "exit_code": proc.returncode,
        }

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "nanhu CLI stdout not JSON",
            "raw": stdout,
            "stderr": stderr,
            "exit_code": proc.returncode,
        }

    # nanhu CLI 退出码 != 0 时把错误透传
    if proc.returncode != 0 and isinstance(parsed, dict) and "ok" not in parsed:
        parsed["ok"] = False
        parsed["exit_code"] = proc.returncode
        if stderr:
            parsed["stderr"] = stderr
    elif isinstance(parsed, dict) and "ok" not in parsed:
        parsed["ok"] = proc.returncode == 0

    return parsed


# ---------------------------------------------------------------------------
# 公共入口：四个对外函数
# ---------------------------------------------------------------------------


def list_train_jobs(
    cluster: str = "",
    status: str = "",
    user_id: str = "",
    local_user_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """查询训练作业列表。

    :param cluster: 集群名（``wuxi`` / ``x10000``）；为空时读 ``config.nanhu_cluster``。
    :param status: 状态过滤（如 ``running`` / ``failed``），空字符串表示不过滤。
    :param user_id: 南湖 userId（保留参数，当前 list CLI 未透传）。
    :param local_user_id: 本地登录用户主键，用于匹配 DB 集群凭据。
    :returns: 作业列表；缺凭据或 nanhu 不可用时返回 ``[]`` 并 ``_write_audit``。
    """
    cluster = cluster or config.nanhu_cluster
    user_id = user_id or config.nanhu_user_id

    action = "jobs list"
    audit_extra: Dict[str, Any] = {
        "cluster": cluster,
        "status": status,
        "user_id": user_id,
        "local_user_id": local_user_id,
    }

    if not _nanhu_available():
        _write_audit({"action": action, "result": "unavailable", **audit_extra})
        return []

    if not _credentials_configured(cluster, local_user_id):
        _write_audit({"action": action, "result": "no_creds", **audit_extra})
        logger.info("[nanhu_client] list_train_jobs: 凭据未配置，返回空列表")
        return []

    args = ["--cluster", cluster, "--format", "json", "jobs", "list", "--job-type", "AI_TRAIN"]
    if status:
        args.extend(["--status", status])

    result = _run_cli(args, cluster=cluster, local_user_id=local_user_id)
    _write_audit({"action": action, "result": "ok", "ok": result.get("ok"), **audit_extra})

    if not result.get("ok", True):
        logger.warning("[nanhu_client] list_train_jobs 返回失败: {}", result.get("error"))
        return []

    payload = result.get("data") if isinstance(result, dict) else None
    if isinstance(payload, dict):
        items = payload.get("result")
        if isinstance(items, list):
            return items
    if isinstance(result, dict) and isinstance(result.get("result"), list):
        return result["result"]
    return []


def get_job_detail(
    cluster: str,
    job_id: str,
    local_user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """查询作业详情。

    :param cluster: 集群名（必填）。
    :param job_id: 作业 ID（必填）。
    :param local_user_id: 本地登录用户主键，用于匹配 DB 集群凭据。
    :returns: dict（含 ``ok`` 字段）；缺凭据时返回友好提示。
    """
    action = "jobs detail"
    audit_extra = {
        "cluster": cluster,
        "job_id": job_id,
        "local_user_id": local_user_id,
    }

    if not _nanhu_available():
        _write_audit({"action": action, "result": "unavailable", **audit_extra})
        return _unavailable_response(action)

    if not _credentials_configured(cluster, local_user_id):
        _write_audit({"action": action, "result": "no_creds", **audit_extra})
        return _missing_credentials_response(action)

    args = [
        "--cluster", cluster,
        "--format", "json",
        "jobs", "detail",
        "--job-id", job_id,
    ]
    result = _run_cli(args, cluster=cluster, local_user_id=local_user_id)
    _write_audit({"action": action, "result": "ok", "ok": result.get("ok"), **audit_extra})
    return result


def create_train_job(
    cluster: str,
    yaml_path: str,
    submit: bool = False,
    local_user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """读取训练作业 YAML 并创建南湖训练作业（**默认 dry-run**）。"""
    action = "jobs create-train"
    audit_extra = {
        "cluster": cluster,
        "yaml_path": yaml_path,
        "submit": submit,
        "local_user_id": local_user_id,
    }

    yaml_file = Path(yaml_path)
    if not yaml_file.is_file():
        _write_audit({"action": action, "result": "yaml_missing", **audit_extra})
        return {
            "ok": False,
            "action": action,
            "hint": "yaml file not found",
            "message": f"yaml 配置文件不存在: {yaml_path}",
            "yaml_path": yaml_path,
        }

    if not _nanhu_available():
        _write_audit({"action": action, "result": "unavailable", **audit_extra})
        return _unavailable_response(action)

    if not submit:
        _write_audit({"action": action, "result": "dry_run", **audit_extra})
        return {
            "ok": True,
            "action": action,
            "dry_run": True,
            "cluster": cluster,
            "yaml_path": yaml_path,
            "message": "dry-run 未提交；如需真实提交请传 submit=True",
        }

    if not _credentials_configured(cluster, local_user_id):
        _write_audit({"action": action, "result": "no_creds", **audit_extra})
        return _missing_credentials_response(action)

    args = [
        "--cluster", cluster,
        "--format", "json",
        "jobs", "create-train",
        "--config", str(yaml_file),
    ]
    result = _run_cli(
        args, timeout=120, cluster=cluster, local_user_id=local_user_id
    )
    _write_audit({"action": action, "result": "ok", "ok": result.get("ok"), **audit_extra})
    return result


def restart_job(
    cluster: str,
    job_id: str,
    submit: bool = False,
    local_user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """重启南湖作业（**默认 dry-run**）。"""
    action = "jobs restart"
    audit_extra = {
        "cluster": cluster,
        "job_id": job_id,
        "submit": submit,
        "local_user_id": local_user_id,
    }

    if not _nanhu_available():
        _write_audit({"action": action, "result": "unavailable", **audit_extra})
        return _unavailable_response(action)

    if not submit:
        _write_audit({"action": action, "result": "dry_run", **audit_extra})
        return {
            "ok": True,
            "action": action,
            "dry_run": True,
            "cluster": cluster,
            "job_id": job_id,
            "message": "dry-run 未提交；如需真实提交请传 submit=True",
        }

    if not _credentials_configured(cluster, local_user_id):
        _write_audit({"action": action, "result": "no_creds", **audit_extra})
        return _missing_credentials_response(action)

    args = [
        "--cluster", cluster,
        "--format", "json",
        "jobs", "restart",
        "--job-id", job_id,
    ]
    result = _run_cli(
        args, timeout=120, cluster=cluster, local_user_id=local_user_id
    )
    _write_audit({"action": action, "result": "ok", "ok": result.get("ok"), **audit_extra})
    return result


# ---------------------------------------------------------------------------
# 模块级别诊断（供 debug / healthz 使用）
# ---------------------------------------------------------------------------


def module_diagnostics() -> Dict[str, Any]:
    """返回当前模块的诊断信息，便于 debug 或 healthz 展示。"""
    nanhu_pkg = _try_import_nanhu_skill()
    cluster = config.nanhu_cluster
    return {
        "nanhu_skill_path": "",
        "nanhu_skill_path_exists": False,
        "nanhu_skill_import_ok": nanhu_pkg is not None,
        "nanhu_binary": _resolve_nanhu_binary(),
        "credentials_configured": _credentials_configured(cluster),
        "credentials_source": (
            "db"
            if _credential_env_from_db(cluster)
            else ("env" if _env_has_nanhu_creds() else "none")
        ),
        "audit_log_path": str(_audit_log_path()),
        "config_cluster": config.nanhu_cluster,
        "config_user_id": config.nanhu_user_id,
    }


__all__ = [
    "list_train_jobs",
    "get_job_detail",
    "create_train_job",
    "restart_job",
    "module_diagnostics",
]


# 让 ``python -m service.nanhu_client`` 也能打印诊断
if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(module_diagnostics(), ensure_ascii=False, indent=2))
