"""Agent-transcript 分布分析（v0.4.0 T6）。

对 cursor 多轮 agent 对话日志（jsonl，每行一个 event）做：
- 基础统计：total_events / role_distribution / assistant turn / user turn / tool_call_count
- Token 分布：每个 assistant message 字符数 → 折线 / 直方图
- 隐式 reward：按 user feedback 文本匹配关键词，给 0 / 0.5 / 1

设计原则：
- 纯 stdlib + loguru，不引入 numpy / pandas / matplotlib
- 错误容忍：单行 json 解析失败时计数 + skip，不抛
- 依赖：typing / loguru
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# 常量：隐式 reward 关键词
# ---------------------------------------------------------------------------

POSITIVE_KEYWORDS: Tuple[str, ...] = (
    "完成",
    "好的",
    "谢谢",
    "可以",
    "通过",
    "approved",
    "done",
    "ok",
    "yes",
    "great",
    "perfect",
)
NEGATIVE_KEYWORDS: Tuple[str, ...] = (
    "不对",
    "重做",
    "失败",
    "再试",
    "错误",
    "no",
    "wrong",
    "bad",
    "fail",
)

# token 估算系数：1 token ≈ 3.5 字符（英文/中文混合，保守估计）
TOKEN_ESTIMATE_DIVISOR: float = 3.5

# 直方图 bin 数量（assistant turn 字符数分布）
HISTOGRAM_BINS: int = 10

# top N 最长 turn
TOP_LONGEST_N: int = 5

# preview 字符数
PREVIEW_MAX_CHARS: int = 100


# ---------------------------------------------------------------------------
# 异常 / 结果类
# ---------------------------------------------------------------------------


class AgentTranscriptError(ValueError):
    """agent-transcripts 分析相关错误。"""


@dataclass
class AnalysisResult:
    """单文件分析结果（也可直接序列化为 dict）。"""

    path: str
    total_events: int
    role_distribution: Dict[str, int]
    assistant_turn_count: int
    user_turn_count: int
    tool_call_count: int
    parse_error_count: int
    char_distribution: Dict[str, Any]
    implicit_rewards: Dict[str, Any]
    top_longest_turns: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "total_events": self.total_events,
            "role_distribution": dict(self.role_distribution),
            "assistant_turn_count": self.assistant_turn_count,
            "user_turn_count": self.user_turn_count,
            "tool_call_count": self.tool_call_count,
            "parse_error_count": self.parse_error_count,
            "char_distribution": self.char_distribution,
            "implicit_rewards": self.implicit_rewards,
            "top_longest_turns": self.top_longest_turns,
        }


# ---------------------------------------------------------------------------
# 基础工具函数
# ---------------------------------------------------------------------------


def load_transcript(path: str) -> List[Dict[str, Any]]:
    """流式读 jsonl，容错单行解析失败。

    - 文件不存在 → 抛 AgentTranscriptError
    - 单行 json.loads 失败 → 跳过并在日志记录
    - 返回 list[dict]（保留原始结构）
    """
    p = Path(path)
    if not p.is_file():
        raise AgentTranscriptError(f"transcript 文件不存在: {path}")

    events: List[Dict[str, Any]] = []
    skipped = 0
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                skipped += 1
                logger.warning(
                    "transcript 单行解析失败 path={} line={} err={}",
                    path,
                    lineno,
                    exc,
                )
    if skipped:
        logger.warning("transcript path={} 共 {} 行解析失败（已跳过）", path, skipped)
    return events


def extract_text_from_message(msg: Any) -> str:
    """从 ``message.content`` 数组里抽所有 ``type=text`` 项的 text 字段。

    兼容以下结构：
    - ``msg`` 是 dict 且含 ``content`` 数组
    - ``content`` 元素是 ``{"type": "text", "text": "..."}`` 或直接字符串
    - 若 ``msg`` 已经是字符串，则原样返回
    """
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg
    if not isinstance(msg, dict):
        return ""

    content = msg.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text", "")
            if isinstance(text, str):
                parts.append(text)
        elif "text" in item and isinstance(item.get("text"), str):
            # 兜底：没有 type 字段但有 text
            parts.append(item["text"])
    return "\n".join(parts)


def count_tool_calls(msg: Any) -> int:
    """从 ``message.content`` 数组数 ``type=tool_use`` 项。"""
    if not isinstance(msg, dict):
        return 0
    content = msg.get("content")
    if not isinstance(content, list):
        return 0
    n = 0
    for item in content:
        if isinstance(item, dict) and item.get("type") in ("tool_use", "tool_call"):
            n += 1
    return n


# ---------------------------------------------------------------------------
# 隐式 reward 分类
# ---------------------------------------------------------------------------


def classify_user_feedback(text: str) -> float:
    """根据关键词规则把 user 文本分类为 0 / 0.5 / 1。

    - 1（正向）：命中 POSITIVE_KEYWORDS 任一
    - 0（负向）：命中 NEGATIVE_KEYWORDS 任一（优先级高于 positive）
    - 0.5（中性）：其它 / 空文本
    """
    if not text or not text.strip():
        return 0.5
    lowered = text.lower()
    # 先检查负向
    for kw in NEGATIVE_KEYWORDS:
        if kw in lowered:
            return 0.0
    # 再检查正向
    for kw in POSITIVE_KEYWORDS:
        if kw in lowered:
            return 1.0
    return 0.5


# ---------------------------------------------------------------------------
# 统计辅助
# ---------------------------------------------------------------------------


def _compute_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0, "min": 0, "sum": 0, "count": 0}
    return {
        "mean": sum(values) / len(values),
        "max": max(values),
        "min": min(values),
        "sum": sum(values),
        "count": len(values),
    }


def _histogram(values: List[int], bins: int = HISTOGRAM_BINS) -> Dict[str, List[float]]:
    """固定 bin 数量的等宽直方图。

    返回 ``{edges: [float,...], counts: [float,...], centers: [float,...]}``，长度均为 bins+1 / bins / bins。
    所有值用 float 以便 JSON 序列化。
    """
    if not values:
        zeros = [0.0] * bins
        edges = [0.0] * (bins + 1)
        return {"edges": edges, "counts": zeros, "centers": zeros}

    v_min = float(min(values))
    v_max = float(max(values))
    if v_max == v_min:
        # 所有值相等 → 全部塞到第一个 bin
        edges = [v_min] + [v_min + 1.0] * bins
        counts: List[float] = [0.0] * bins
        counts[0] = float(len(values))
        centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(bins)]
        return {"edges": edges, "counts": counts, "centers": centers}

    width = (v_max - v_min) / bins
    edges = [v_min + i * width for i in range(bins + 1)]
    counts = [0.0] * bins
    for v in values:
        idx = int((float(v) - v_min) / width)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        counts[idx] += 1.0
    centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(bins)]
    return {"edges": edges, "counts": counts, "centers": centers}


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


# ---------------------------------------------------------------------------
# 主分析函数
# ---------------------------------------------------------------------------


def analyze_transcript(path: str) -> Dict[str, Any]:
    """分析单个 jsonl 路径，返回 dict（可直接 ``json.dumps``）。

    返回结构见 plan/任务说明：

    - path / total_events / role_distribution
    - assistant_turn_count / user_turn_count / tool_call_count
    - char_distribution: {turn_indices, char_counts, token_estimates, mean_chars, max_chars}
    - implicit_rewards: {user_turn_indices, rewards, mean_reward, n_feedback}
    - top_longest_turns: [{turn_index, char_count, preview}, ...]
    """
    events = load_transcript(path)

    role_counter: Counter = Counter()
    assistant_turns: List[Dict[str, Any]] = []  # 顺序保留
    user_turns: List[Dict[str, Any]] = []  # 顺序保留
    tool_call_count = 0
    parse_error_count = 0

    for ev in events:
        if not isinstance(ev, dict):
            parse_error_count += 1
            continue
        role = ev.get("role")
        # 空 role 或 None 视作 '?'
        if not isinstance(role, str) or not role.strip():
            role = "?"
        role_counter[role] += 1

        msg = ev.get("message")
        if not isinstance(msg, dict):
            # 极少数 event 没有 message 字段（系统行等）
            if role in ("assistant", "user"):
                # 仍记一条空 turn 以保留顺序
                if role == "assistant":
                    assistant_turns.append({
                        "turn_index": len(assistant_turns),
                        "text": "",
                        "char_count": 0,
                    })
                else:
                    user_turns.append({
                        "user_index": len(user_turns),
                        "text": "",
                        "char_count": 0,
                    })
            continue

        text = extract_text_from_message(msg)
        n_tools = count_tool_calls(msg)
        tool_call_count += n_tools

        if role == "assistant":
            assistant_turns.append({
                "turn_index": len(assistant_turns),
                "text": text,
                "char_count": len(text),
                "tool_calls": n_tools,
            })
        elif role == "user":
            user_turns.append({
                "user_index": len(user_turns),
                "text": text,
                "char_count": len(text),
            })
        # '?' / 其它 role 只计数，不入 turn 列表

    # ---- char distribution ----
    turn_indices: List[int] = []
    char_counts: List[int] = []
    for t in assistant_turns:
        turn_indices.append(int(t["turn_index"]))
        char_counts.append(int(t["char_count"]))
    token_estimates: List[int] = [
        int(round(c / TOKEN_ESTIMATE_DIVISOR)) for c in char_counts
    ]
    stats = _compute_stats(char_counts)
    char_distribution = {
        "turn_indices": turn_indices,
        "char_counts": char_counts,
        "token_estimates": token_estimates,
        "mean_chars": float(stats["mean"]),
        "max_chars": int(stats["max"]),
        "min_chars": int(stats["min"]),
        "sum_chars": int(stats["sum"]),
        "count": int(stats["count"]),
        "histogram": _histogram(char_counts, HISTOGRAM_BINS),
    }

    # ---- implicit rewards ----
    user_indices: List[int] = []
    rewards: List[float] = []
    for ut in user_turns:
        r = classify_user_feedback(ut["text"])
        user_indices.append(int(ut["user_index"]))
        rewards.append(r)
    if rewards:
        mean_reward = sum(rewards) / len(rewards)
    else:
        mean_reward = 0.0
    implicit_rewards = {
        "user_turn_indices": user_indices,
        "rewards": rewards,
        "mean_reward": float(mean_reward),
        "n_feedback": len(rewards),
        "n_positive": int(sum(1 for r in rewards if r == 1.0)),
        "n_negative": int(sum(1 for r in rewards if r == 0.0)),
        "n_neutral": int(sum(1 for r in rewards if r == 0.5)),
    }

    # ---- top longest ----
    sorted_turns = sorted(
        assistant_turns,
        key=lambda t: int(t["char_count"]),
        reverse=True,
    )[:TOP_LONGEST_N]
    top_longest_turns: List[Dict[str, Any]] = []
    for t in sorted_turns:
        top_longest_turns.append({
            "turn_index": int(t["turn_index"]),
            "char_count": int(t["char_count"]),
            "preview": _truncate(t["text"], PREVIEW_MAX_CHARS),
        })

    result = AnalysisResult(
        path=str(path),
        total_events=len(events),
        role_distribution={k: int(v) for k, v in role_counter.items()},
        assistant_turn_count=len(assistant_turns),
        user_turn_count=len(user_turns),
        tool_call_count=int(tool_call_count),
        parse_error_count=parse_error_count,
        char_distribution=char_distribution,
        implicit_rewards=implicit_rewards,
        top_longest_turns=top_longest_turns,
    )
    return result.to_dict()


def batch_analyze(paths: Iterable[str]) -> List[Dict[str, Any]]:
    """批量调用 analyze_transcript；单个文件失败 → 记 error 字段并继续。"""
    results: List[Dict[str, Any]] = []
    for p in paths:
        try:
            results.append(analyze_transcript(p))
        except Exception as exc:  # noqa: BLE001
            logger.exception("批量分析失败 path={}: {}", p, exc)
            results.append({
                "path": str(p),
                "error": str(exc),
            })
    return results


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def _default_transcript_files() -> List[str]:
    """返回默认扫描的 2 个真实数据文件绝对路径。"""
    base = (
        "/root/.cursor/projects/"
        "mnt-zj-data-data-songquanheng-slime/agent-transcripts"
    )
    ef2a = (
        f"{base}/ef2a8b4d-129d-472b-ba10-9cae83f4d080/"
        "ef2a8b4d-129d-472b-ba10-9cae83f4d080.jsonl"
    )
    f40b = (
        f"{base}/40b0d2f3-8432-433d-981f-61b46216f249/"
        "40b0d2f3-8432-433d-981f-61b46216f249.jsonl"
    )
    out: List[str] = []
    for p in (ef2a, f40b):
        if Path(p).is_file():
            out.append(p)
    return out


__all__ = [
    "AgentTranscriptError",
    "POSITIVE_KEYWORDS",
    "NEGATIVE_KEYWORDS",
    "TOKEN_ESTIMATE_DIVISOR",
    "HISTOGRAM_BINS",
    "TOP_LONGEST_N",
    "load_transcript",
    "extract_text_from_message",
    "count_tool_calls",
    "classify_user_feedback",
    "analyze_transcript",
    "batch_analyze",
]


if __name__ == "__main__":
    # CLI 调试入口
    for p in _default_transcript_files():
        r = analyze_transcript(p)
        print(json.dumps(r, ensure_ascii=False, indent=2)[:2000])
        print("---")
