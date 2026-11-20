"""解析训练 run 目录下的 ``evaluations.md``（外部评测记录）。

该文件由外部评测流程持续追加，用来记录「训练每到某个 step 就拉起一次外部
评测」的台账。表格列固定为::

    | Training step | Evaluation | Round | Nanhu job name | Nanhu job ID | Run | Job status | Result |

``Result`` 单元格里嵌着分数，常见两种形态：

- ``ungraded 1 ['xx-123']; weighted score on 429 tasks: 0.660;
  95% selection interval +-0.039; holdout score: 0.400 over 85 of 87 tasks``
- ``first round 288/500; merged 298/500 = 0.596; 2 reruns; still bad {...}``

**一个评测集可能同时给出多个评测集的分数**：上例里 ``weighted score on 429
tasks`` 与 ``holdout score ... over 85 of 87 tasks`` 是**两个不同题池**的口径，
必须分开画线，不能只取其一。

本模块只做纯解析（无 DB / 路由依赖），供训练曲线页渲染「外部评测分数」。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "EVALUATION_FILE_NAME",
    "EVALUATION_COLUMNS",
    "SCORE_KIND_LABELS",
    "SCORE_KIND_ORDER",
    "extract_scores",
    "iter_score_kinds",
    "parse_evaluations_md",
    "read_evaluations_md",
    "read_evaluations_for_rollout_root",
]

# run 根目录下的台账文件名
EVALUATION_FILE_NAME = "evaluations.md"

# 标准列顺序（与文件里的表头一一对应）
EVALUATION_COLUMNS = (
    "step",
    "evaluation",
    "round",
    "job_name",
    "job_id",
    "run",
    "job_status",
    "result",
)

# 表头别名 → 标准列名
_HEADER_ALIASES = {
    "training step": "step",
    "step": "step",
    "evaluation": "evaluation",
    "eval": "evaluation",
    "round": "round",
    "nanhu job name": "job_name",
    "nanhu job id": "job_id",
    "run": "run",
    "job status": "job_status",
    "status": "job_status",
    "result": "result",
}

# 开头的说明行：Training job: xxx (jb-xxx). Updated 2026-11-19 15:23:25.
_JOB_LINE_RE = re.compile(
    r"Training job:\s*(?P<name>[^\n(]+?)\s*"
    r"(?:\((?P<job_id>[^)\n]+)\))?"
    r"[.\s]*Updated\s+(?P<updated>[0-9][0-9\-: ]*)",
    re.IGNORECASE,
)

# 各形态的分数
_WEIGHTED_RE = re.compile(r"weighted score on\s+(\d+)\s+tasks:\s*([0-9]*\.?[0-9]+)")
_HOLDOUT_RE = re.compile(
    r"holdout score:\s*([0-9]*\.?[0-9]+)\s+over\s+(\d+)\s+of\s+(\d+)\s+tasks"
)
_INTERVAL_RE = re.compile(r"selection interval\s*\+-([0-9]*\.?[0-9]+)")
_MERGED_RE = re.compile(r"merged\s+(\d+)\s*/\s*(\d+)\s*=\s*([0-9]*\.?[0-9]+)")
_FIRST_ROUND_RE = re.compile(r"first round\s+(\d+)\s*/\s*(\d+)")
_UNGRADED_RE = re.compile(r"ungraded\s+(\d+)")
_REGRADE_RE = re.compile(r"regrade[ds]?\s+(\d+)")

# 作业状态：failed / error 等一律视为「未完成」
_STATUS_DONE = "finished"
_STATUS_RUNNING = "running"
_STATUS_SCHEDULED = "scheduling"
_STATUS_ERROR = "error"

# ---------------------------------------------------------------------------
# 分数口径（score kind）
#
# 一次外部评测可能同时给出**多个评测集**的分数，例如 panel 同时给出
# 「429 题的加权分」与「87 题的 holdout 分」——两者题池不同，必须分开画线。
# 下面定义各口径的显示名与固定顺序；口径名同时就是 ``extract_scores``
# 结果里的取值键（weighted / merged / holdout）。
# ---------------------------------------------------------------------------

# 口径 → 中文显示名
SCORE_KIND_LABELS: Dict[str, str] = {
    "weighted": "加权分",
    "merged": "merged 分",
    "holdout": "holdout 分",
}

# 口径的展示顺序（加权分优先，其后 merged / holdout）
SCORE_KIND_ORDER: tuple = ("weighted", "merged", "holdout")


def iter_score_kinds(scores: Optional[Dict[str, Any]]) -> List[str]:
    """列出该次评测里**实际有分数**的口径，按固定顺序返回。

    用于把一次评测拆成多条曲线（例如 panel 的加权分 + holdout 分）。
    """
    if not scores:
        return []
    out: List[str] = []
    for kind in SCORE_KIND_ORDER:
        value = scores.get(kind)
        if value is None:
            continue
        out.append(kind)
    return out


def _to_float(text: Any) -> Optional[float]:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def _to_int(text: Any) -> Optional[int]:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def _split_cells(line: str) -> List[str]:
    """拆一行 markdown 表格单元格（去掉首尾的 ``|``）。"""
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [cell.strip() for cell in text.split("|")]


def _is_separator(cells: List[str]) -> bool:
    """是否为 ``|---|---|`` 这样的分隔行。"""
    if not cells:
        return False
    return all(cell and set(cell) <= set("-: ") for cell in cells)


def extract_scores(result_text: Any) -> Dict[str, Any]:
    """从 ``Result`` 单元格里抽出各个分数。

    返回字典的键固定，缺失时值为 ``None``：

    - ``weighted`` / ``weighted_tasks``：加权分与参与题数
    - ``holdout`` / ``holdout_tasks`` / ``holdout_tasks_total``：留出分与题数
    - ``interval``：95% 选择区间
    - ``merged`` / ``merged_pass`` / ``merged_total``：merged 形态的分数
    - ``first_round_pass`` / ``first_round_total``：first round 形态的分数
    - ``ungraded``：未评分题数
    """
    text = str(result_text or "")
    out: Dict[str, Any] = {
        "weighted": None,
        "weighted_tasks": None,
        "holdout": None,
        "holdout_tasks": None,
        "holdout_tasks_total": None,
        "interval": None,
        "merged": None,
        "merged_pass": None,
        "merged_total": None,
        "first_round_pass": None,
        "first_round_total": None,
        "ungraded": None,
    }
    if not text.strip():
        return out

    match = _WEIGHTED_RE.search(text)
    if match:
        out["weighted_tasks"] = _to_int(match.group(1))
        out["weighted"] = _to_float(match.group(2))

    match = _HOLDOUT_RE.search(text)
    if match:
        out["holdout"] = _to_float(match.group(1))
        out["holdout_tasks"] = _to_int(match.group(2))
        out["holdout_tasks_total"] = _to_int(match.group(3))

    match = _INTERVAL_RE.search(text)
    if match:
        out["interval"] = _to_float(match.group(1))

    match = _MERGED_RE.search(text)
    if match:
        out["merged_pass"] = _to_int(match.group(1))
        out["merged_total"] = _to_int(match.group(2))
        out["merged"] = _to_float(match.group(3))

    match = _FIRST_ROUND_RE.search(text)
    if match:
        out["first_round_pass"] = _to_int(match.group(1))
        out["first_round_total"] = _to_int(match.group(2))

    match = _UNGRADED_RE.search(text)
    if match:
        out["ungraded"] = _to_int(match.group(1))

    return out


def _normalize_status(text: Any) -> str:
    """把作业状态归一成四档：finished / running / scheduling / error。"""
    raw = str(text or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    if _STATUS_DONE in low:
        return _STATUS_DONE
    if _STATUS_ERROR in low or "attention" in low:
        return _STATUS_ERROR
    if _STATUS_RUNNING in low:
        return _STATUS_RUNNING
    if _STATUS_SCHEDULED in low:
        return _STATUS_SCHEDULED
    return raw


def _build_header(cells: List[str]) -> List[str]:
    """把表头单元格映射成标准列名；无法识别的列丢弃。"""
    out: List[str] = []
    for cell in cells:
        key = cell.strip().lower()
        name = _HEADER_ALIASES.get(key)
        out.append(name or "")
    return out


def parse_evaluations_md(text: Optional[str]) -> Dict[str, Any]:
    """解析 ``evaluations.md`` 全文。

    Returns:
        ``{"exists": bool, "job_name": str, "job_id": str, "updated_at": str,
        "runs": [...]}``；``runs`` 每项含标准列与解析出的 ``scores``。
    """
    empty: Dict[str, Any] = {
        "exists": False,
        "job_name": "",
        "job_id": "",
        "updated_at": "",
        "runs": [],
    }
    if not text:
        return empty

    lines = str(text).splitlines()
    job_name = ""
    job_id = ""
    updated_at = ""
    for line in lines:
        match = _JOB_LINE_RE.search(line)
        if not match:
            continue
        job_name = (match.group("name") or "").strip()
        job_id = (match.group("job_id") or "").strip()
        updated_at = (match.group("updated") or "").strip()
        break

    header_index = -1
    columns: List[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = _split_cells(stripped)
        if _is_separator(cells):
            continue
        mapped = _build_header(cells)
        if "step" in mapped and "result" in mapped:
            header_index = index
            columns = mapped
            break

    if header_index < 0:
        # 没有识别到表头：仍视为「文件存在但无评测记录」
        return {
            "exists": True,
            "job_name": job_name,
            "job_id": job_id,
            "updated_at": updated_at,
            "runs": [],
        }

    runs: List[Dict[str, Any]] = []
    for line in lines[header_index + 1:]:
        stripped = line.strip()
        if not stripped:
            break
        if not stripped.startswith("|"):
            break
        cells = _split_cells(stripped)
        if _is_separator(cells):
            continue
        if len(cells) < len(columns):
            cells = cells + [""] * (len(columns) - len(cells))
        elif len(cells) > len(columns):
            # 多出来的列并入最后一列（Result 里可能出现未转义的竖线）
            extra = cells[len(columns) - 1:]
            cells = cells[: len(columns) - 1] + [" | ".join(extra)]

        item: Dict[str, Any] = {}
        for name, cell in zip(columns, cells):
            if not name:
                continue
            item[name] = cell
        item["step"] = _to_int(item.get("step")) or 0
        item["job_status"] = _normalize_status(item.get("job_status"))
        item["scores"] = extract_scores(item.get("result"))
        runs.append(item)

    runs.sort(key=lambda r: (int(r.get("step") or 0), str(r.get("evaluation") or "")))

    return {
        "exists": True,
        "job_name": job_name,
        "job_id": job_id,
        "updated_at": updated_at,
        "runs": runs,
    }


def read_evaluations_md(path: Optional[Path | str]) -> Dict[str, Any]:
    """读取并解析指定的 ``evaluations.md``；文件不存在返回 ``exists=False``。"""
    if not path:
        return parse_evaluations_md("")
    try:
        target = Path(path)
        if not target.is_file():
            return parse_evaluations_md("")
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return parse_evaluations_md("")
    return parse_evaluations_md(text)


def read_evaluations_for_rollout_root(rollout_root: Optional[str]) -> Dict[str, Any]:
    """从 ``task.rollout_root`` 解析 run 目录并读取其 ``evaluations.md``。

    Returns:
        ``{"exists": bool, "path": str, "job_name": str, "job_id": str,
        "updated_at": str, "runs": [...]}``
    """
    from utils.tb_run_paths import resolve_run_dir

    run_dir = resolve_run_dir(rollout_root)
    if run_dir is None:
        return {
            "exists": False,
            "path": "",
            "job_name": "",
            "job_id": "",
            "updated_at": "",
            "runs": [],
        }
    target = Path(run_dir) / EVALUATION_FILE_NAME
    data = read_evaluations_md(target)
    data["path"] = str(target)
    return data
