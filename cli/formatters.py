"""CLI 输出格式化辅助：table / json（与 nanhu-skill 风格一致）。

设计：
- ``emit(payload, fmt)`` 是统一入口；fmt ∈ {"json", "table"}
- table 模式：把 dict-of-list / list-of-dict 渲染成 markdown 表格；
  无法识别时打「(unprintable)」 跳过
- json 模式：紧凑 JSON（compact 模式：separators=(",", ":")），UTF-8
- 顶层 ``Response`` 风格（code/msg/data）自动展开 ``data`` 字段
"""

from __future__ import annotations

import json
import sys
from typing import Any, Iterable, Mapping, Sequence


def _unwrap_response(payload: Any) -> Any:
    """如果 payload 是 ``{code, msg, data}``，返回 ``data``；否则原样返回。

    主要给本地 Web API 的 ``Response.success`` / ``Response.error`` 用，
    CLI 默认展开 data，便于 list 渲染。
    """
    if (
        isinstance(payload, Mapping)
        and "code" in payload
        and "msg" in payload
        and "data" in payload
    ):
        return payload.get("data")
    return payload


def _project_to_rows(payload: Any) -> list[dict]:
    """把各种 list-like payload 规整成 ``[dict, ...]``。

    支持：
        - List of dict
        - Dict with list value（取首个 list value）
        - 单 dict → [dict]
    """
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, Mapping)]
    if isinstance(payload, Mapping):
        # 找到首个 list 类型的 value
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], Mapping):
                return list(v)
        return [dict(payload)]
    return []


def _stringify(value: Any) -> str:
    """把任意值转字符串；None → '-'。"""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # 保留 4 位小数
        return f"{value:.4f}" if abs(value) >= 1e-4 else f"{value:.4e}"
    if isinstance(value, (list, tuple)):
        return ",".join(_stringify(v) for v in value)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def format_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """把 list-of-dict 渲染成 markdown 表格。

    列顺序：取所有行 key 的并集，前两次出现的顺序稳定。
    列宽：截断到 32 字符避免溢出（长字段在右侧加 ...）。
    """
    if not rows:
        return "(empty)"

    # 列顺序：所有 key 的并集，按第一次出现排序
    seen: set[str] = set()
    columns: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                columns.append(k)

    # 渲染表格
    lines: list[str] = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")

    max_col_width = 48
    for r in rows:
        cells: list[str] = []
        for col in columns:
            cell = _stringify(r.get(col))
            if len(cell) > max_col_width:
                cell = cell[: max_col_width - 3] + "..."
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def format_payload(payload: Any, fmt: str = "json", *, pretty: bool = False) -> str:
    """顶层格式化入口。

    Args:
        payload: 任意数据
        fmt: "json" | "table"
        pretty: json 模式是否 indent

    Returns:
        可直接打印的字符串
    """
    payload = _unwrap_response(payload)

    if fmt == "table":
        rows = _project_to_rows(payload)
        return format_table(rows)

    # 默认 JSON
    if pretty:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def emit(payload: Any, fmt: str = "json", *, pretty: bool = False) -> None:
    """打印 payload 到 stdout，自动处理 format。

    同 ``format_payload`` 但直接 write 到 sys.stdout。
    """
    text = format_payload(payload, fmt=fmt, pretty=pretty)
    sys.stdout.write(text)
    sys.stdout.write("\n")


def emit_error(msg: str, code: int = -1, fmt: str = "json") -> None:
    """打错误信息（stderr），保证 agent 能感知。"""
    payload = {"code": code, "msg": msg, "data": None}
    text = format_payload(payload, fmt=fmt)
    sys.stderr.write(text)
    sys.stderr.write("\n")