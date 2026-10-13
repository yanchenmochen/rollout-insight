"""
rollout 落盘 .pt 文件分析工具，封装 inspect-rollout-trajectory skill。
"""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_VIEW_TRAJ_PATH = Path.home() / ".cursor/skills/inspect-rollout-trajectory/scripts/view_trajectory.py"
_TOKENIZERS_PARALLELISM = "false"


def _load_view_module():
    """动态加载 view_trajectory 脚本为模块。"""
    if not _VIEW_TRAJ_PATH.is_file():
        raise FileNotFoundError(f"view_trajectory 脚本不存在: {_VIEW_TRAJ_PATH}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", _TOKENIZERS_PARALLELISM)
    spec = importlib.util.spec_from_file_location("view_trajectory", str(_VIEW_TRAJ_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _vt():
    return _load_view_module()


def load_dump_meta(path: str | Path) -> Dict[str, Any]:
    """读取 .pt 顶部元数据：rollout_id、n_samples、n_attempts、mean_reward。"""
    mod = _vt()
    dump = mod.load_dump(str(path))
    samples = dump.get("samples", [])
    rollout_id = dump.get("rollout_id")
    n_samples = len(samples)
    # n_attempts：去重 (group_index, instance_id, rollout_id) 计数
    attempt_keys = set()
    rewards = []
    for s in samples:
        m = s.get("metadata") or {}
        key = (s.get("group_index"), m.get("instance_id"), s.get("rollout_id"))
        attempt_keys.add(key)
        r = s.get("reward")
        if isinstance(r, (int, float)):
            rewards.append(r)
    n_attempts = len(attempt_keys)
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return {
        "rollout_id": rollout_id,
        "n_samples": n_samples,
        "n_attempts": n_attempts,
        "mean_reward": mean_reward,
        "pt_path": str(path),
    }


def summarize_dump(path: str | Path) -> List[Dict[str, Any]]:
    """返回每个 sample 的摘要行（与 view_trajectory 表格一致）。"""
    mod = _vt()
    dump = mod.load_dump(str(path))
    samples = dump["samples"]
    rows: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        m = s.get("metadata") or {}
        loss_mask = s.get("loss_mask") or []
        rows.append({
            "idx": i,
            "group_index": s.get("group_index"),
            "rollout_id": s.get("rollout_id"),
            "segment_index": m.get("segment_index"),
            "segment_count": m.get("segment_count"),
            "instance_id": m.get("instance_id"),
            "reward": s.get("reward"),
            "status": str(s.get("status", "")).split(".")[-1].lower(),
            "agent_turns": m.get("agent_turns"),
            "tool_calls": m.get("tool_calls"),
            "diff_line_count": m.get("diff_line_count"),
            "response_length": s.get("response_length"),
            "mask_sum": sum(loss_mask) if loss_mask else 0,
            "sandbox_id": m.get("sandbox_id"),
        })
    return rows


def render_transcript(
    path: str | Path,
    sample_idx: int,
    tokenizer_dir: Optional[str] = None,
    include_prompt: bool = True,
) -> str:
    """渲染一个 sample 的完整 transcript（Markdown）。"""
    mod = _vt()
    dump = mod.load_dump(str(path))
    samples = dump["samples"]
    if not (0 <= sample_idx < len(samples)):
        raise IndexError(f"sample_idx {sample_idx} 越界（0..{len(samples)-1}）")
    tok = None
    if tokenizer_dir:
        try:
            tok = mod._load_tokenizer(tokenizer_dir)
        except Exception as exc:
            raise RuntimeError(f"加载 tokenizer 失败: {exc}") from exc
    return mod.render_transcript(samples[sample_idx], include_prompt=include_prompt, tokenizer=tok)


def split_turns(text: str) -> List[tuple[str, str]]:
    """复用 view_trajectory 的 turn 切分。"""
    return _vt().split_turns(text)
