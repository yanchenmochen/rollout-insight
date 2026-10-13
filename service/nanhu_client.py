"""NanhuClient 封装：复用本地 nanhu-skill 提供的 ``nanhu`` CLI。

设计目标（rollout-insight v0.3.0 阶段 3A）：

- **不重写 OpenAPI 客户端**：所有南湖调用都通过 ``nanhu`` / ``nanhu-cli`` CLI
  转发，避免重复实现鉴权、token 缓存、限流等细节。
- **优先 importlib 动态 import** ``/mnt/zj-data/data/songquanheng/nanhu-skill/src/nanhu``，
  若该路径不可用则降级到 ``python -m nanhu.cli`` / ``nanhu`` 子进程；
  若 ``nanhu`` 二进制也找不到或凭据未配置，**返回友好提示**，不抛异常。
- **dry-run 默认**：``create_train_job`` / ``restart_job`` 默认 ``submit=False``，
  CLI 在不传 ``--submit`` 时也是 dry-run 行为；所有调用都会写入
  ``log/nanhu_audit.log`` 便于审计。
- **stdout JSON 透传**：nanhu CLI 默认输出 compact JSON；本模块把解析后的
  dict 直接返回给调用方，由 controller 层负责格式化。
- **缺凭据返回** ``{"hint": "...", "ok": False}`` 风格的 dict，让 controller
  能优雅地 fallback 而非 500。

对外暴露 4 个函数：
    - ``list_train_jobs(cluster, status, user_id)``
    - ``get_job_detail(cluster, job_id)``
    - ``create_train_job(cluster, yaml_path, submit=False)``
    - ``restart_job(cluster, job_id, submit=False)``
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from utils.config_loader import config

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------

# nanhu-skill 本地源码路径（v0.3.0 复用约定）
NANHU_SKILL_SRC = Path(
    "/mnt/zj-data/data/songquanheng/nanhu-skill/src/nanhu"
)

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
    """尝试通过 importlib 加载 nanhu-skill 本地源码。

    返回加载到的顶层 package 对象（``nanhu``），失败返回 ``None``。
    """
    if not NANHU_SKILL_SRC.is_dir():
        return None
    src_parent = NANHU_SKILL_SRC.parent
    src_parent_str = str(src_parent)
    if src_parent_str not in sys.path:
        sys.path.insert(0, src_parent_str)
    try:
        # 先把可能存在的旧缓存踢掉，便于热重载
        for mod_name in [m for m in list(sys.modules) if m == "nanhu" or m.startswith("nanhu.")]:
            sys.modules.pop(mod_name, None)
        return importlib.import_module("nanhu")
    except Exception as exc:  # pragma: no cover - 仅在 import 失败时
        logger.warning("importlib 加载 nanhu-skill 失败: {}", exc)
        return None


def _resolve_nanhu_binary() -> Optional[str]:
    """查找 ``nanhu`` / ``nanhu-cli`` 二进制路径；找不到返回 ``None``。"""
    for name in ("nanhu", "nanhu-cli"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _nanhu_available() -> bool:
    """nanhu 是否可用（importlib 或 CLI 二进制至少一种）。"""
    if _try_import_nanhu_skill() is not None:
        return True
    return _resolve_nanhu_binary() is not None


def _credentials_configured() -> bool:
    """检查凭据是否已配置（环境变量或 nanhu-skill 默认 env 文件）。"""
    if os.environ.get("NANHU_APP_KEY") and os.environ.get("NANHU_APP_SECRET"):
        return True
    # nanhu-skill 默认 env 文件（位于 <repo>/config/credentials.env）
    # NANHU_SKILL_SRC 指向 src/nanhu，repo 根是其 parents[2]
    repo_root = NANHU_SKILL_SRC.parent.parent
    default_env = repo_root / "config" / "credentials.env"
    if default_env.is_file():
        try:
            content = default_env.read_text(encoding="utf-8")
            has_key = any(
                line.strip().startswith("NANHU_APP_KEY=") and "=" in line
                and line.split("=", 1)[1].strip()
                for line in content.splitlines()
            )
            has_secret = any(
                line.strip().startswith("NANHU_APP_SECRET=") and "=" in line
                and line.split("=", 1)[1].strip()
                for line in content.splitlines()
            )
            if has_key and has_secret:
                return True
        except OSError:
            pass
    # 也允许自定义路径
    override = os.environ.get("NANHU_CREDENTIALS_PATH")
    if override and Path(override).is_file():
        return True
    return False


def _missing_credentials_response(action: str) -> Dict[str, Any]:
    """凭据未配置时的友好提示响应。"""
    return {
        "ok": False,
        "action": action,
        "hint": "nanhu credentials not configured",
        "message": (
            "未配置南湖凭据：请在环境变量或 "
            "/mnt/zj-data/data/songquanheng/nanhu-skill/config/credentials.env "
            "中填写 NANHU_APP_KEY / NANHU_APP_SECRET / NANHU_USER_ID 后再试。"
        ),
    }


def _unavailable_response(action: str) -> Dict[str, Any]:
    """nanhu-skill 与 CLI 都不可用时的友好提示响应。"""
    return {
        "ok": False,
        "action": action,
        "hint": "nanhu skill not available",
        "message": (
            "未找到 nanhu-skill 源码或 nanhu 二进制：请检查 "
            f"{NANHU_SKILL_SRC} 或执行 pip install -e "
            "/mnt/zj-data/data/songquanheng/nanhu-skill。"
        ),
    }


# ---------------------------------------------------------------------------
# CLI 调用
# ---------------------------------------------------------------------------


def _run_cli(args: List[str], timeout: int = 60) -> Dict[str, Any]:
    """通过 ``nanhu`` 子进程调用命令并解析 stdout JSON。

    :param args: 不包含 ``nanhu`` 二进制名的参数列表，例如
        ``["--cluster", "wuxi", "jobs", "list", "--job-type", "AI_TRAIN"]``。
    :param timeout: 超时秒数。
    :returns: 解析后的 dict；解析失败时返回 ``{"ok": False, "raw": ..., "error": ...}``。
    """
    binary = _resolve_nanhu_binary()
    if not binary:
        return {
            "ok": False,
            "error": "nanhu binary not found",
            "hint": "ensure nanhu CLI is installed (pip install -e /mnt/zj-data/data/songquanheng/nanhu-skill)",
        }

    cmd = [binary, *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
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
) -> List[Dict[str, Any]]:
    """查询训练作业列表。

    :param cluster: 集群名（``wuxi`` / ``x10000``）；为空时读 ``config.nanhu_cluster``。
    :param status: 状态过滤（如 ``running`` / ``failed``），空字符串表示不过滤。
    :param user_id: 用户 ID（用于跨用户查询）；空时读 ``config.nanhu_user_id``。
    :returns: 作业列表；缺凭据或 nanhu 不可用时返回 ``[]`` 并 ``_write_audit``。
    """
    cluster = cluster or config.nanhu_cluster
    user_id = user_id or config.nanhu_user_id

    action = "jobs list"
    audit_extra: Dict[str, Any] = {"cluster": cluster, "status": status, "user_id": user_id}

    if not _nanhu_available():
        _write_audit({"action": action, "result": "unavailable", **audit_extra})
        # 注意：保持返回类型稳定（list），不可用时返回空列表
        return []

    if not _credentials_configured():
        _write_audit({"action": action, "result": "no_creds", **audit_extra})
        logger.info("[nanhu_client] list_train_jobs: 凭据未配置，返回空列表")
        return []

    args = ["--cluster", cluster, "--format", "json", "jobs", "list", "--job-type", "AI_TRAIN"]
    if status:
        args.extend(["--status", status])

    result = _run_cli(args)
    _write_audit({"action": action, "result": "ok", "ok": result.get("ok"), **audit_extra})

    if not result.get("ok", True):
        logger.warning("[nanhu_client] list_train_jobs 返回失败: {}", result.get("error"))
        return []

    # nanhu list 返回结构: {"data": {"result": [...]}, "code": "200", ...}
    payload = result.get("data") if isinstance(result, dict) else None
    if isinstance(payload, dict):
        items = payload.get("result")
        if isinstance(items, list):
            return items
    if isinstance(result, dict) and isinstance(result.get("result"), list):
        return result["result"]
    return []


def get_job_detail(cluster: str, job_id: str) -> Dict[str, Any]:
    """查询作业详情。

    :param cluster: 集群名（必填）。
    :param job_id: 作业 ID（必填）。
    :returns: dict（含 ``ok`` 字段）；缺凭据时返回 ``_missing_credentials_response`` 风格 dict。
    """
    action = "jobs detail"
    audit_extra = {"cluster": cluster, "job_id": job_id}

    if not _nanhu_available():
        _write_audit({"action": action, "result": "unavailable", **audit_extra})
        return _unavailable_response(action)

    if not _credentials_configured():
        _write_audit({"action": action, "result": "no_creds", **audit_extra})
        return _missing_credentials_response(action)

    args = [
        "--cluster", cluster,
        "--format", "json",
        "jobs", "detail",
        "--job-id", job_id,
    ]
    result = _run_cli(args)
    _write_audit({"action": action, "result": "ok", "ok": result.get("ok"), **audit_extra})
    return result


def create_train_job(
    cluster: str,
    yaml_path: str,
    submit: bool = False,
) -> Dict[str, Any]:
    """读取训练作业 YAML 并创建南湖训练作业（**默认 dry-run**）。

    :param cluster: 集群名。
    :param yaml_path: 训练作业 YAML 配置路径。
    :param submit: 是否真正提交；``False``（默认）走 dry-run，仅打印请求体。
    :returns: dict；缺凭据 / nanhu 不可用 / 文件不存在时返回友好提示。
    """
    action = "jobs create-train"
    audit_extra = {
        "cluster": cluster,
        "yaml_path": yaml_path,
        "submit": submit,
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
        # dry-run：记录意图但不下发
        _write_audit({"action": action, "result": "dry_run", **audit_extra})
        return {
            "ok": True,
            "action": action,
            "dry_run": True,
            "cluster": cluster,
            "yaml_path": yaml_path,
            "message": "dry-run 未提交；如需真实提交请传 submit=True",
        }

    if not _credentials_configured():
        _write_audit({"action": action, "result": "no_creds", **audit_extra})
        return _missing_credentials_response(action)

    args = [
        "--cluster", cluster,
        "--format", "json",
        "jobs", "create-train",
        "--config", str(yaml_file),
    ]
    result = _run_cli(args, timeout=120)
    _write_audit({"action": action, "result": "ok", "ok": result.get("ok"), **audit_extra})
    return result


def restart_job(
    cluster: str,
    job_id: str,
    submit: bool = False,
) -> Dict[str, Any]:
    """重启南湖作业（**默认 dry-run**）。

    :param cluster: 集群名。
    :param job_id: 作业 ID。
    :param submit: 是否真正提交。
    """
    action = "jobs restart"
    audit_extra = {
        "cluster": cluster,
        "job_id": job_id,
        "submit": submit,
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

    if not _credentials_configured():
        _write_audit({"action": action, "result": "no_creds", **audit_extra})
        return _missing_credentials_response(action)

    args = [
        "--cluster", cluster,
        "--format", "json",
        "jobs", "restart",
        "--job-id", job_id,
    ]
    result = _run_cli(args, timeout=120)
    _write_audit({"action": action, "result": "ok", "ok": result.get("ok"), **audit_extra})
    return result


# ---------------------------------------------------------------------------
# 模块级别诊断（供 debug / healthz 使用）
# ---------------------------------------------------------------------------


def module_diagnostics() -> Dict[str, Any]:
    """返回当前模块的诊断信息，便于 debug 或 healthz 展示。"""
    nanhu_pkg = _try_import_nanhu_skill()
    return {
        "nanhu_skill_path": str(NANHU_SKILL_SRC),
        "nanhu_skill_path_exists": NANHU_SKILL_SRC.is_dir(),
        "nanhu_skill_import_ok": nanhu_pkg is not None,
        "nanhu_binary": _resolve_nanhu_binary(),
        "credentials_configured": _credentials_configured(),
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
    "NANHU_SKILL_SRC",
]


# 让 ``python -m service.nanhu_client`` 也能打印诊断
if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(module_diagnostics(), ensure_ascii=False, indent=2))