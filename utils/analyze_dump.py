"""rollout 落盘 .pt 文件分析工具。

历史上曾动态加载 Cursor skill ``inspect-rollout-trajectory`` 的
``view_trajectory.py``；该路径在生产环境常不存在，会导致扫描全部失败、
任务被标为 FAILED。

本模块改为：
1. 优先用 zip + pickle 直接读 torch 2.x 的 ``.pt``（无需 libmusa / 完整 torch）；
2. 失败时再惰性 ``torch.load``；
3. transcript 渲染逻辑内置，与 view_trajectory 输出格式兼容。
"""
from __future__ import annotations

import io
import os
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_TOKENIZERS_PARALLELISM = "false"

# 可选：仍兼容历史 skill / syctool 脚本路径（仅作兜底，不作为主路径）
_VIEW_TRAJ_CANDIDATES: Tuple[Path, ...] = (
    Path.home() / ".cursor/skills/inspect-rollout-trajectory/scripts/view_trajectory.py",
    Path(
        "/home/nanhu-agentic-rl-stack/projects/nanhu-slime/musa_run/tools/syctool/view_trajectory.py"
    ),
)


def _ensure_tokenizer_env() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", _TOKENIZERS_PARALLELISM)


# ---------------------------------------------------------------------------
# .pt 加载（无硬依赖 torch / 外部 skill）
# ---------------------------------------------------------------------------


class _TorchStub:
    """pickle 反序列化时替掉 torch.* 类，尽量还原 Python 原生结构。"""

    def __init__(self, qualname: str = "torch.Tensor") -> None:
        self._qualname = qualname

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Storage / Tensor 构造：多数扫描场景不需要真实数值
        if args and isinstance(args[0], (list, tuple)):
            return list(args[0])
        return []

    def __getattr__(self, item: str) -> Any:
        return _TorchStub(f"{self._qualname}.{item}")


class _PtUnpickler(pickle.Unpickler):
    def persistent_load(self, pid: Any) -> Any:  # noqa: ANN401
        # torch storage 外部引用；扫描元数据路径不需要真实 storage
        return []

    def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401
        if module.startswith("torch"):
            return _TorchStub(f"{module}.{name}")
        return super().find_class(module, name)


def _load_dump_via_zip_pickle(path: str) -> Dict[str, Any]:
    """读取 torch 2.x zip 格式 .pt（``{name}/data.pkl``）。"""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        data_members = [n for n in names if n.endswith("/data.pkl") or n == "data.pkl"]
        if not data_members:
            raise ValueError(f".pt zip 中未找到 data.pkl: {path}")
        # 通常只有一个 archive 根
        member = sorted(data_members)[0]
        with zf.open(member) as raw:
            obj = _PtUnpickler(raw).load()
    if not isinstance(obj, dict):
        raise ValueError(f".pt dump 不是 dict: {type(obj).__name__}")
    return obj


def _load_dump_via_torch(path: str) -> Dict[str, Any]:
    """惰性 torch.load 兜底（旧格式 / pickle 协议失败时）。"""
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import torch  # noqa: WPS433

        dump = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(dump, dict):
        raise ValueError(f".pt dump 不是 dict: {type(dump).__name__}")
    return dump


def _load_dump_via_view_trajectory(path: str) -> Dict[str, Any]:
    """兼容环境中若仍装有 view_trajectory 脚本。"""
    import importlib.util

    for candidate in _VIEW_TRAJ_CANDIDATES:
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location("view_trajectory", str(candidate))
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        dump = mod.load_dump(path)
        if not isinstance(dump, dict):
            raise ValueError(f"view_trajectory.load_dump 返回非 dict: {type(dump).__name__}")
        return dump
    raise FileNotFoundError("view_trajectory 脚本不可用")


def _load_dump(path: str | Path) -> Dict[str, Any]:
    """加载 .pt dump：zip/pickle → torch.load → view_trajectory。"""
    _ensure_tokenizer_env()
    p = str(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f".pt dump 文件不存在: {p}")

    errors: List[str] = []
    if zipfile.is_zipfile(p):
        try:
            return _load_dump_via_zip_pickle(p)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"zip/pickle: {exc}")

    try:
        return _load_dump_via_torch(p)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"torch.load: {exc}")

    try:
        return _load_dump_via_view_trajectory(p)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"view_trajectory: {exc}")

    raise RuntimeError(f"无法加载 .pt dump {p}; 尝试失败: {' | '.join(errors)}")


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def load_dump(path: str | Path) -> Dict[str, Any]:
    """加载完整 .pt dump dict（优先 zip/pickle，不依赖可 import 的 torch）。

    供 ``rollout_pt_parser.load_pt``、transcript 等复用；失败时抛出
    ``RuntimeError``（内部已聚合 zip / torch / view_trajectory 尝试错误）。
    """
    return _load_dump(path)


def load_dump_meta(path: str | Path) -> Dict[str, Any]:
    """读取 .pt 顶部元数据：rollout_id、n_samples、n_attempts、mean_reward。"""
    dump = load_dump(path)
    samples = dump.get("samples", []) or []
    rollout_id = dump.get("rollout_id")
    n_samples = len(samples)
    attempt_keys = set()
    rewards: List[float] = []
    for s in samples:
        m = s.get("metadata") or {}
        key = (s.get("group_index"), m.get("instance_id"), s.get("rollout_id"))
        attempt_keys.add(key)
        r = s.get("reward")
        if isinstance(r, (int, float)):
            rewards.append(float(r))
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
    dump = load_dump(path)
    samples = dump.get("samples") or []
    rows: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        m = s.get("metadata") or {}
        loss_mask = s.get("loss_mask") or []
        try:
            mask_sum = int(sum(loss_mask)) if loss_mask else 0
        except TypeError:
            mask_sum = 0
        seg_index = m.get("segment_index")
        if seg_index is None:
            seg_index = i
        seg_count = m.get("segment_count")
        if seg_count is None:
            seg_count = 1
        rows.append(
            {
                "idx": i,
                "group_index": s.get("group_index"),
                "rollout_id": s.get("rollout_id"),
                "segment_index": seg_index,
                "segment_count": seg_count,
                "instance_id": m.get("instance_id"),
                "reward": s.get("reward"),
                "status": str(s.get("status", "")).split(".")[-1].lower(),
                "agent_turns": m.get("agent_turns"),
                "tool_calls": m.get("tool_calls"),
                "diff_line_count": m.get("diff_line_count"),
                "response_length": s.get("response_length"),
                "mask_sum": mask_sum,
                "sandbox_id": m.get("sandbox_id"),
            }
        )
    return rows


def split_turns(text: str) -> List[tuple[str, str]]:
    """按 ``<|im_start|>role\\n...<|im_end|>`` 切分为 ``(role, content)``。"""
    turns: List[tuple[str, str]] = []
    parts = (text or "").split("<|im_start|>")
    if parts and parts[0].strip():
        turns.append(("assistant", parts[0]))
    for part in parts[1:]:
        newline = part.find("\n")
        role = (part[:newline] if newline != -1 else part).strip() or "?"
        content = (part[newline + 1 :] if newline != -1 else "").split("<|im_end|>")[0]
        turns.append((role, content))
    return turns


def _split_token_turns(
    tokens: Sequence[int], tokenizer: Any
) -> List[Tuple[str, str, int, int]]:
    """在 token 序列上按 ``<|im_start|>`` 切分，返回 (role, content, start, end)。"""
    turn_start_id = tokenizer.token_to_id("<|im_start|>")
    if turn_start_id is None:
        return [
            (
                "sequence",
                tokenizer.decode(list(tokens), skip_special_tokens=False),
                0,
                len(tokens),
            )
        ]

    starts = [i for i, token_id in enumerate(tokens) if token_id == turn_start_id]
    if not starts:
        return [
            (
                "sequence",
                tokenizer.decode(list(tokens), skip_special_tokens=False),
                0,
                len(tokens),
            )
        ]

    bounds = ([0] if starts[0] else []) + starts + [len(tokens)]
    turns: List[Tuple[str, str, int, int]] = []
    for start, end in zip(bounds, bounds[1:]):
        text = tokenizer.decode(list(tokens[start:end]), skip_special_tokens=False)
        if tokens[start] != turn_start_id:
            turns.append(("sequence", text, start, end))
            continue
        marker = "<|im_start|>"
        body = text[len(marker) :] if text.startswith(marker) else text
        role, separator, content = body.partition("\n")
        if not separator:
            turns.append((role.strip() or "?", "", start, end))
            continue
        content = re.sub(r"<\|im_end\|>\s*$", "", content)
        turns.append((role.strip() or "?", content, start, end))
    return turns


def _mask_spans(
    full_mask: Sequence[int], prompt_length: int, start: int, end: int
) -> List[Tuple[int, int, str, str]]:
    spans: List[Tuple[int, int, str, str]] = []
    span_start = start
    while span_start < end:
        location = "PROMPT" if span_start < prompt_length else "RESPONSE"
        trainable = bool(full_mask[span_start])
        span_end = span_start + 1
        while span_end < end:
            next_location = "PROMPT" if span_end < prompt_length else "RESPONSE"
            if next_location != location or bool(full_mask[span_end]) != trainable:
                break
            span_end += 1
        spans.append(
            (span_start, span_end, location, "TRAINABLE" if trainable else "MASKED")
        )
        span_start = span_end
    return spans


def _format_spans(spans: Sequence[Tuple[int, int, str, str]]) -> str:
    return " | ".join(
        f"{location}/{loss} [{start}:{end}) {end - start} tok"
        for start, end, location, loss in spans
    )


def _legacy_render_transcript(sample: Dict[str, Any], include_prompt: bool = True) -> str:
    """无 token ids / tokenizer 时的兼容渲染。"""
    output: List[str] = []
    metadata = sample.get("metadata") or {}
    output.append(
        f"reward={sample.get('reward')}  status={sample.get('status')}  "
        f"turns={metadata.get('agent_turns')}  tool_calls={metadata.get('tool_calls')}  "
        f"diff_lines={metadata.get('diff_line_count')}  "
        f"resp_len={sample.get('response_length')}  sandbox={metadata.get('sandbox_id')}"
    )
    if include_prompt and sample.get("prompt"):
        prompt = sample.get("prompt")
        output.extend(
            (
                "\n----- DATASET PROMPT (full token sequence unavailable) -----",
                str(prompt).strip(),
            )
        )
    output.append("\n----- LEGACY TRANSCRIPT (response only) -----")
    for i, (role, content) in enumerate(split_turns(sample.get("response", "") or ""), 1):
        content = content.strip()
        if content:
            output.extend((f"\n### turn {i} · {role}", content))
    return "\n".join(output)


def _as_int_list(values: Any) -> List[int]:
    if values is None:
        return []
    if isinstance(values, list):
        out: List[int] = []
        for v in values:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                return []
        return out
    try:
        return [int(v) for v in list(values)]
    except Exception:  # noqa: BLE001
        return []


def _render_sample_transcript(
    sample: Dict[str, Any],
    *,
    include_prompt: bool = True,
    tokenizer: Any = None,
) -> str:
    """渲染单个 sample 的完整 transcript（Markdown）。"""
    tokens = _as_int_list(sample.get("tokens"))
    if not tokens or tokenizer is None:
        return _legacy_render_transcript(sample, include_prompt=include_prompt)

    response_length = int(sample.get("response_length") or 0)
    if response_length < 0 or response_length > len(tokens):
        raise ValueError(
            f"invalid response_length={response_length} for {len(tokens)} stored tokens"
        )
    prompt_length = len(tokens) - response_length
    response_mask = _as_int_list(sample.get("loss_mask"))
    if not response_mask:
        response_mask = [1] * response_length
    if len(response_mask) != response_length:
        raise ValueError(
            f"loss_mask length {len(response_mask)} != response_length {response_length}"
        )
    full_mask = [0] * prompt_length + response_mask

    metadata = sample.get("metadata") or {}
    segment_index = metadata.get("segment_index")
    segment_count = metadata.get("segment_count")
    segment = (
        f"{segment_index + 1}/{segment_count}"
        if isinstance(segment_index, int) and isinstance(segment_count, int)
        else "?"
    )
    trainable_tokens = sum(response_mask)
    output = [
        (
            f"segment={segment}  attempt={sample.get('rollout_id')}  reward={sample.get('reward')}  "
            f"status={sample.get('status')}  sandbox={metadata.get('sandbox_id')}"
        ),
        (
            f"full_sequence_length={len(tokens)}  leading_prompt_length={prompt_length}  "
            f"response_length={response_length}  trainable_tokens={trainable_tokens}  "
            f"masked_tokens={len(tokens) - trainable_tokens}"
        ),
        "\n----- COMPLETE STORED TRAINING SEQUENCE -----",
        "loss labels describe this segment; prompt turns may have been trainable in an earlier segment.",
    ]

    visible_turn = 0
    for role, content, start, end in _split_token_turns(tokens, tokenizer):
        if not include_prompt and end <= prompt_length:
            continue
        visible_turn += 1
        spans = _mask_spans(full_mask, prompt_length, start, end)
        output.extend(
            (
                f"\n### turn {visible_turn} · {role} · tokens [{start}:{end})",
                f"mask: {_format_spans(spans)}",
                content.strip(),
            )
        )
    return "\n".join(output)


def render_transcript(
    path: str | Path,
    sample_idx: int,
    tokenizer_dir: Optional[str] = None,
    include_prompt: bool = True,
) -> str:
    """渲染一个 sample 的完整 transcript（Markdown）。"""
    dump = load_dump(path)
    samples = dump.get("samples") or []
    if not (0 <= sample_idx < len(samples)):
        raise IndexError(f"sample_idx {sample_idx} 越界（0..{len(samples) - 1}）")
    tok = None
    if tokenizer_dir:
        try:
            # 惰性导入，避免 utils 包导入时拉起 torch
            from utils.rollout_pt_parser import _load_tokenizer

            tok = _load_tokenizer(tokenizer_dir)
        except Exception as exc:
            raise RuntimeError(f"加载 tokenizer 失败: {exc}") from exc
    return _render_sample_transcript(
        samples[sample_idx],
        include_prompt=include_prompt,
        tokenizer=tok,
    )
