"""CLI 子命令：tokenizer。

- ``tokenizer set``  → 改 config.json 的 ``tokenizer_dir`` 字段
- ``tokenizer test`` → 验证指定路径确实是 HuggingFace tokenizer 目录
  （含 tokenizer.json / tokenizer_config.json / vocab.json 之一）

路径解析：
1. ``--config PATH`` 显式指定 config.json
2. ``ROLLOUT_INSIGHT_CONFIG_PATH`` 环境变量
3. ``./config.json``（CLI cwd）

注意：
- 写操作默认 dry-run；加 ``--submit`` 才会真的覆盖 config.json
- 覆盖前若文件已含 ``tokenizer_dir``，会在 diff 里展示新旧值
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

REQUIRED_TOKENIZER_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
    "merges.txt",
)


def _resolve_config_path(explicit: Optional[str] = None) -> Path:
    """解析 config.json 路径（与 ``utils.config_loader`` 策略一致）。"""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("ROLLOUT_INSIGHT_CONFIG_PATH")
    if env:
        return Path(env).expanduser().resolve()
    # 默认 cwd 下的 config.json
    return Path("./config.json").resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    """读 config.json；不存在抛 FileNotFoundError。"""
    if not config_path.exists():
        raise FileNotFoundError(f"config.json 不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config_path: Path, data: dict[str, Any]) -> None:
    """写 config.json（pretty，UTF-8）。"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def test_tokenizer_path(path: str) -> dict[str, Any]:
    """验证 ``path`` 是否是 HF tokenizer 目录。

    Returns:
        dict 包含
            - valid: bool
            - missing: [文件名, ...]  缺失的关键文件
            - present: [文件名, ...]  命中的关键文件
            - path: 解析后的绝对路径
            - message: 给人看的总结
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {
            "valid": False,
            "missing": list(REQUIRED_TOKENIZER_FILES),
            "present": [],
            "path": str(p),
            "message": f"路径不存在: {p}",
        }
    if not p.is_dir():
        return {
            "valid": False,
            "missing": list(REQUIRED_TOKENIZER_FILES),
            "present": [],
            "path": str(p),
            "message": f"不是目录: {p}",
        }

    present: list[str] = []
    missing: list[str] = []
    for name in REQUIRED_TOKENIZER_FILES:
        if (p / name).is_file():
            present.append(name)
        else:
            missing.append(name)

    # 至少命中 1 个文件才算通过
    valid = len(present) >= 1
    msg = "OK" if valid else f"未找到任何 tokenizer 关键文件（需要 {REQUIRED_TOKENIZER_FILES} 任一）"
    return {
        "valid": valid,
        "missing": missing,
        "present": present,
        "path": str(p),
        "message": msg,
    }


def set_tokenizer_dir(
    *,
    config_path: Path,
    new_dir: str,
    submit: bool = False,
) -> dict[str, Any]:
    """覆盖 config.json 的 ``tokenizer_dir``。

    Args:
        config_path: config.json 路径
        new_dir: 新 tokenizer 目录
        submit: True → 写文件；False → dry-run

    Returns:
        dict 含 diff（old/new）、applied(bool)、path
    """
    data = load_config(config_path)
    old_value = data.get("tokenizer_dir", "")

    plan: dict[str, Any] = {
        "action": "set tokenizer_dir",
        "config_path": str(config_path),
        "old": old_value,
        "new": new_dir,
        "applied": False,
    }

    if not submit:
        plan["dry_run"] = True
        plan["note"] = "dry-run，未写文件；加 --submit 才执行"
        return plan

    data["tokenizer_dir"] = new_dir
    save_config(config_path, data)
    plan["applied"] = True
    plan["dry_run"] = False
    return plan