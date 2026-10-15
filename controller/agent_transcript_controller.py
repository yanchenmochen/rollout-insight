"""Agent-transcripts 分析控制器（v0.4.0 T6）。

端点：
- GET /api/agent-transcripts/page        HTML 报告页（默认扫描 2 个真实数据文件）
- GET /api/agent-transcripts/report      JSON 单文件报告（?path=...）
- GET /api/agent-transcripts/batch       JSON 批量（?dir=...）
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from loguru import logger
from starlette.templating import Jinja2Templates

from data.response import Response
from data.rollout_insight_exception import RolloutInsightException
from utils.auth_session import require_login
from utils.config_loader import config

# v0.5.1：业务路由统一要求登录
router = APIRouter(dependencies=[Depends(require_login)])
templates = Jinja2Templates(directory=str(config.templates_dir))


# ---------------------------------------------------------------------------
# 默认路径：写死
# ---------------------------------------------------------------------------

DEFAULT_TRANSCRIPTS_DIR = (
    "/root/.cursor/projects/"
    "mnt-zj-data-data-songquanheng-slime/agent-transcripts"
)

DEFAULT_FILES: List[str] = [
    f"{DEFAULT_TRANSCRIPTS_DIR}/ef2a8b4d-129d-472b-ba10-9cae83f4d080/ef2a8b4d-129d-472b-ba10-9cae83f4d080.jsonl",
    f"{DEFAULT_TRANSCRIPTS_DIR}/40b0d2f3-8432-433d-981f-61b46216f249/40b0d2f3-8432-433d-981f-61b46216f249.jsonl",
]


def _existing_default_files() -> List[str]:
    return [p for p in DEFAULT_FILES if Path(p).is_file()]


# ---------------------------------------------------------------------------
# Service 延迟导入
# ---------------------------------------------------------------------------


def _analyzer_module():
    import importlib
    return importlib.import_module("service.agent_transcript_analyzer")


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


@router.get("/api/agent-transcripts/page", include_in_schema=False)
async def page_agent_transcripts(request: Request) -> HTMLResponse:
    """渲染 agent_transcript_report.html。"""
    analyzer = _analyzer_module()
    files = _existing_default_files()

    reports: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for p in files:
        try:
            reports.append(analyzer.analyze_transcript(p))
        except Exception as exc:  # noqa: BLE001
            logger.exception("分析失败 path={}: {}", p, exc)
            warnings.append(f"{Path(p).name}: {exc}")

    # 概览摘要（两文件对比）
    summary_rows: List[Dict[str, Any]] = []
    for r in reports:
        summary_rows.append({
            "name": Path(r.get("path", "")).name,
            "path": r.get("path"),
            "total_events": r.get("total_events", 0),
            "user_turn_count": r.get("user_turn_count", 0),
            "assistant_turn_count": r.get("assistant_turn_count", 0),
            "tool_call_count": r.get("tool_call_count", 0),
            "mean_chars": r.get("char_distribution", {}).get("mean_chars", 0.0),
            "max_chars": r.get("char_distribution", {}).get("max_chars", 0),
            "mean_reward": r.get("implicit_rewards", {}).get("mean_reward", 0.0),
            "n_feedback": r.get("implicit_rewards", {}).get("n_feedback", 0),
        })

    return templates.TemplateResponse(
        request,
        "agent_transcript_report.html",
        {
            "page_title": "Agent-Transcripts 报告",
            "reports": reports,
            "summary_rows": summary_rows,
            "warnings": warnings,
            "default_files": DEFAULT_FILES,
        },
    )


@router.get("/api/agent-transcripts/report")
async def api_agent_transcript_report(
    path: str = Query(..., description="单个 jsonl 绝对路径"),
) -> Response:
    """返回单文件分析 JSON。"""
    if not path:
        raise RolloutInsightException(
            RolloutInsightException.INVALID_ARGUMENT,
            "path 参数不能为空",
        )
    if not Path(path).is_file():
        raise RolloutInsightException(
            RolloutInsightException.PARSE_DUMP_NOT_FOUND,
            f"文件不存在: {path}",
        )
    analyzer = _analyzer_module()
    try:
        result = analyzer.analyze_transcript(path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("analyze_transcript 失败: {}", exc)
        raise RolloutInsightException(
            RolloutInsightException.PARSE_ERROR,
            f"分析失败: {exc}",
        ) from exc
    return Response.success(msg="ok", data=result)


@router.get("/api/agent-transcripts/batch")
async def api_agent_transcript_batch(
    dir: Optional[str] = Query(
        default=None,
        description="批量目录；缺省时使用默认 2 个文件",
    ),
) -> Response:
    """批量分析；返回 list[dict]。"""
    analyzer = _analyzer_module()
    if dir:
        d = Path(dir)
        if not d.is_dir():
            raise RolloutInsightException(
                RolloutInsightException.PARSE_DUMP_NOT_FOUND,
                f"目录不存在: {dir}",
            )
        paths = sorted(str(p) for p in d.glob("**/*.jsonl"))
    else:
        paths = list(DEFAULT_FILES)

    if not paths:
        return Response.success(
            msg="no files to analyze",
            data={"reports": [], "count": 0},
        )

    try:
        results = analyzer.batch_analyze(paths)
    except Exception as exc:  # noqa: BLE001
        logger.exception("batch_analyze 失败: {}", exc)
        raise RolloutInsightException(
            RolloutInsightException.PARSE_ERROR,
            f"批量分析失败: {exc}",
        ) from exc
    return Response.success(
        msg="ok",
        data={"reports": results, "count": len(results)},
    )
