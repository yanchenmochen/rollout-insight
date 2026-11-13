"""按南湖 / slime 作业环境变量区分：真实训练 / rollout-only / 评测。

判定优先级对齐 nanhu-slime 三类作业矩阵：

1. ``TRAINING_SCRIPT`` / launcher 指向 ``eval/run_*_eval.sh`` → 评测
2. ``NUM_ROLLOUT=0`` 且未 ``DISABLE_EVAL=1`` → 评测
3. ``DEBUG_ROLLOUT_ONLY=1`` → rollout-only（断点续推）
4. ``DEBUG_ROLLOUT_ONLY`` 为 ``0`` 或不设，且 ``TRAINING_SCRIPT`` 为非评测训练脚本
   → 真实训练（不以 ``ROLLOUT_SHUFFLE`` 作为主判定）
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

__all__ = [
    "JOB_TYPE_LABELS",
    "classify_slime_job_type",
    "attach_slime_job_type",
    "load_env_from_run_dir",
    "normalize_env_map",
]

JOB_TYPE_LABELS = {
    "train": "真实训练",
    "rollout_only": "rollout-only / 断点续推",
    "eval": "评测",
    "unknown": "未知",
}

_EVAL_SCRIPT_RE = re.compile(
    r"(?:^|/)(?:musa_run/)?eval/run_[^/\s]*eval\.sh$",
    re.IGNORECASE,
)

# 真实训练 / rollout-only 共用训练 launcher（非 eval/）
_TRAIN_SCRIPT_RE = re.compile(
    r"(?:^|/)(?:musa_run/)?run_[^/\s]+\.sh$",
    re.IGNORECASE,
)


def normalize_env_map(raw: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """把任意 mapping 规整为 ``{KEY: str}``（跳过空键）。"""
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[str, str] = {}
    for key, val in raw.items():
        k = str(key or "").strip()
        if not k:
            continue
        out[k] = "" if val is None else str(val)
    return out


def _env_get(env: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        if key in env and str(env[key]).strip() != "":
            return str(env[key]).strip()
        # 兼容大小写不一致
        lower = key.lower()
        for ek, ev in env.items():
            if ek.lower() == lower and str(ev).strip() != "":
                return str(ev).strip()
    return ""


def _is_truthy_one(value: str) -> bool:
    return str(value or "").strip() == "1"


def _is_zero(value: str) -> bool:
    return str(value or "").strip() == "0"


def _script_is_eval(script: str) -> bool:
    text = (script or "").strip().replace("\\", "/")
    if not text:
        return False
    if _EVAL_SCRIPT_RE.search(text):
        return True
    name = Path(text).name.lower()
    return name.startswith("run_") and "eval" in name and name.endswith(".sh")


def _script_is_train(script: str) -> bool:
    """非评测的训练脚本（如 ``musa_run/run_021_32b_scale_swe.sh``）。"""
    text = (script or "").strip().replace("\\", "/")
    if not text or _script_is_eval(text):
        return False
    if _TRAIN_SCRIPT_RE.search(text):
        return True
    name = Path(text).name.lower()
    return name.startswith("run_") and name.endswith(".sh") and "eval" not in name


def _debug_rollout_only_off(value: str) -> bool:
    """真实训练：``DEBUG_ROLLOUT_ONLY`` 为 ``0`` 或不设。"""
    text = str(value or "").strip()
    return text == "" or text == "0"


def classify_slime_job_type(
    env: Optional[Mapping[str, Any]] = None,
    *,
    launcher: str = "",
    training_script: str = "",
) -> Dict[str, Any]:
    """按 env / launcher 判定作业类型。

    返回 ``job_type`` / ``job_type_label`` / ``signals``（判定依据列表）。
    """
    env_map = normalize_env_map(env)
    script = (
        (training_script or "").strip()
        or _env_get(env_map, "TRAINING_SCRIPT", "TRAIN_SCRIPT")
        or (launcher or "").strip()
    )
    debug_ro = _env_get(env_map, "DEBUG_ROLLOUT_ONLY")
    disable_eval = _env_get(env_map, "DISABLE_EVAL")
    num_rollout = _env_get(env_map, "NUM_ROLLOUT")
    colocate = _env_get(env_map, "COLOCATE")
    slime_eval_cfg = _env_get(env_map, "SLIME_EVAL_CONFIG")

    signals: list[str] = []

    if _script_is_eval(script):
        signals.append(f"TRAINING_SCRIPT/launcher={script}")
        kind = "eval"
    elif num_rollout and _is_zero(num_rollout) and not _is_truthy_one(disable_eval):
        signals.append("NUM_ROLLOUT=0 且未 DISABLE_EVAL=1")
        kind = "eval"
    elif (
        slime_eval_cfg
        and _is_truthy_one(debug_ro)
        and (not colocate or _is_zero(colocate))
    ):
        signals.append("SLIME_EVAL_CONFIG + DEBUG_ROLLOUT_ONLY=1 + COLOCATE≠1")
        kind = "eval"
    elif _is_truthy_one(debug_ro):
        signals.append("DEBUG_ROLLOUT_ONLY=1")
        if script:
            signals.append(f"TRAINING_SCRIPT={script}")
        kind = "rollout_only"
    elif _debug_rollout_only_off(debug_ro) and _script_is_train(script):
        debug_label = "DEBUG_ROLLOUT_ONLY=0" if _is_zero(debug_ro) else "DEBUG_ROLLOUT_ONLY未设"
        signals.append(debug_label)
        signals.append(f"TRAINING_SCRIPT={script}")
        kind = "train"
    else:
        kind = "unknown"
        signals.append("环境变量不足以判定（需 DEBUG_ROLLOUT_ONLY=0/未设 + 训练 TRAINING_SCRIPT）")

    return {
        "job_type": kind,
        "job_type_label": JOB_TYPE_LABELS.get(kind, JOB_TYPE_LABELS["unknown"]),
        "signals": signals,
        "training_script": script,
    }


def attach_slime_job_type(
    target: MutableMapping[str, Any],
    env: Optional[Mapping[str, Any]] = None,
    *,
    launcher: str = "",
) -> MutableMapping[str, Any]:
    """就地写入 ``job_type`` / ``job_type_label`` 等字段。"""
    target.update(classify_slime_job_type(env, launcher=launcher))
    return target


def load_env_from_run_dir(run_dir: Optional[Path | str]) -> Tuple[Dict[str, str], str, str]:
    """从 run 目录读取 env 与 launcher。

    优先 ``resolved_config.yaml`` 的 ``env`` + ``launcher``；
    其次 ``manifest.yaml`` 的 ``launch.launcher``。

    Returns:
        ``(env_map, launcher, source)``；无文件时为空。
    """
    if not run_dir:
        return {}, "", ""
    try:
        root = Path(run_dir)
    except (TypeError, ValueError):
        return {}, "", ""
    if not root.is_dir():
        return {}, "", ""

    try:
        import yaml
    except Exception:
        return {}, "", ""

    env_map: Dict[str, str] = {}
    launcher = ""
    source = ""

    resolved = root / "resolved_config.yaml"
    if resolved.is_file():
        try:
            data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
        if isinstance(data, Mapping):
            env_map = normalize_env_map(data.get("env") if isinstance(data.get("env"), Mapping) else {})
            launcher = str(data.get("launcher") or "").strip()
            source = "resolved_config.yaml"

    if not launcher:
        manifest = root / "manifest.yaml"
        if manifest.is_file():
            try:
                data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
            except Exception:
                data = {}
            if isinstance(data, Mapping):
                launch = data.get("launch") if isinstance(data.get("launch"), Mapping) else {}
                launcher = str(launch.get("launcher") or "").strip()
                if launcher and not source:
                    source = "manifest.yaml"
                elif launcher:
                    source = f"{source}+manifest" if source else "manifest.yaml"

    return env_map, launcher, source
