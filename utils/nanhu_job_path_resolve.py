"""从南湖作业详情解析创建 Rollout 任务所需路径字段。

典型来源（slime / 021-32B 作业）：
- ``envVars.ROLLOUT_DUMP_TEMPLATE`` → dumps 目录（最优）
- ``OUTPUT_DIR`` / ``RUN_DIR`` / ``outputDir.outputPath`` → 运行根，再探测
  ``generated/rollout_dumps`` 或 ``rollout_dumps``
- ``HF_CHECKPOINT`` / ``LOAD_MODEL_PATH`` / ``SFT_MODEL`` → tokenizer 目录
- ``name`` / ``EXP_NAME`` → 任务名
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

__all__ = [
    "env_map_from_job",
    "output_dir_from_job",
    "resolve_paths_from_job_detail",
]


_PLACEHOLDER_RE = re.compile(r"\{[^}]+\}")


def env_map_from_job(job: Mapping[str, Any]) -> Dict[str, str]:
    """把 ``envVars`` / ``env_vars`` 列表转成 ``{KEY: value}``。"""
    raw = job.get("envVars") if "envVars" in job else job.get("env_vars")
    if not isinstance(raw, list):
        return {}
    out: Dict[str, str] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        key = item.get("key") or item.get("name")
        if not key:
            continue
        val = item.get("value")
        out[str(key)] = "" if val is None else str(val)
    return out


def output_dir_from_job(job: Mapping[str, Any], env: Optional[Mapping[str, str]] = None) -> str:
    """解析作业输出根目录（不含 ``rollout_dumps`` 子路径）。"""
    env = env if env is not None else env_map_from_job(job)
    for key in ("OUTPUT_DIR", "RUN_DIR"):
        val = (env.get(key) or "").strip()
        if val:
            return val.rstrip("/")

    od = job.get("outputDir") if "outputDir" in job else job.get("output_dir")
    if isinstance(od, Mapping):
        path = (od.get("outputPath") or od.get("path") or "").strip()
        return path.rstrip("/") if path else ""
    if isinstance(od, str):
        return od.strip().rstrip("/")
    return ""


def _dirname_from_dump_template(template: str) -> str:
    """``.../rollout_{rollout_id}.pt`` → 所在目录。"""
    text = (template or "").strip()
    if not text:
        return ""
    cleaned = _PLACEHOLDER_RE.sub("X", text)
    try:
        parent = Path(cleaned).parent
    except (TypeError, ValueError):
        return ""
    s = str(parent)
    return "" if s in (".", "") else s.rstrip("/")


def _prefer_existing_dumps_under(run_root: str) -> str:
    """在运行根下优先选用已有 dumps 目录。"""
    root = Path(run_root)
    candidates = [
        root / "generated" / "rollout_dumps",
        root / "rollout_dumps",
    ]
    for cand in candidates:
        try:
            if cand.is_dir():
                return str(cand)
        except OSError:
            continue
    # 目录尚不存在时仍给出 slime 惯例路径，便于作业跑起来后扫描命中
    return str(candidates[0])


def resolve_paths_from_job_detail(
    job: Mapping[str, Any],
    *,
    prefer_existing_dirs: bool = True,
) -> Dict[str, Any]:
    """从作业详情 dict 解析创建任务字段。

    Returns:
        ``task_name`` / ``rollout_root`` / ``tokenizer_dir`` / ``job_id`` /
        ``job_name`` / ``output_dir`` / ``warnings``（list[str]）
    """
    warnings: List[str] = []
    env = env_map_from_job(job)

    job_id = str(
        job.get("uuid")
        or job.get("job_id")
        or job.get("jobId")
        or job.get("id")
        or ""
    ).strip()
    job_name = str(job.get("name") or "").strip()
    exp_name = (env.get("EXP_NAME") or "").strip()
    task_name = job_name or exp_name
    if not task_name:
        warnings.append("作业缺少 name / EXP_NAME，无法推断任务名")

    output_dir = output_dir_from_job(job, env)
    rollout_root = ""

    template = (env.get("ROLLOUT_DUMP_TEMPLATE") or "").strip()
    if template:
        rollout_root = _dirname_from_dump_template(template)
        if not rollout_root:
            warnings.append("ROLLOUT_DUMP_TEMPLATE 无法解析为目录")

    if not rollout_root and output_dir:
        if prefer_existing_dirs:
            rollout_root = _prefer_existing_dumps_under(output_dir)
        else:
            rollout_root = f"{output_dir.rstrip('/')}/generated/rollout_dumps"
        warnings.append(
            "未找到 ROLLOUT_DUMP_TEMPLATE，已按 OUTPUT_DIR 推断 rollout_root"
        )

    if not rollout_root:
        warnings.append("无法解析 Rollout 根路径")

    tokenizer_dir = ""
    for key in ("HF_CHECKPOINT", "LOAD_MODEL_PATH", "SFT_MODEL"):
        val = (env.get(key) or "").strip()
        if val:
            tokenizer_dir = val.rstrip("/")
            break
    if not tokenizer_dir:
        warnings.append("未找到 HF_CHECKPOINT / LOAD_MODEL_PATH / SFT_MODEL")

    return {
        "job_id": job_id,
        "job_name": job_name,
        "task_name": task_name,
        "rollout_root": rollout_root,
        "tokenizer_dir": tokenizer_dir,
        "output_dir": output_dir,
        "warnings": warnings,
    }


def extract_job_payload(detail: Mapping[str, Any]) -> Optional[MutableMapping[str, Any]]:
    """从 ``get_job_detail`` 返回体取出作业 dict；失败返回 ``None``。"""
    if not isinstance(detail, Mapping):
        return None
    if detail.get("ok") is False and "data" not in detail:
        return None
    # CLI 错误：{"error":"作业找不到",...}
    if detail.get("error") and "data" not in detail:
        return None
    code = detail.get("code")
    if code is not None and str(code) not in ("200", "0") and "data" not in detail:
        return None
    data = detail.get("data")
    if isinstance(data, Mapping) and (
        data.get("uuid")
        or data.get("job_id")
        or data.get("name")
        or data.get("envVars")
        or data.get("env_vars")
    ):
        return dict(data)
    # 少数透传场景：detail 本身就是作业体
    if detail.get("uuid") or detail.get("envVars") or detail.get("env_vars"):
        return dict(detail)
    return None
