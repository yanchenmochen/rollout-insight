"""南湖（nanhu）作业调度相关路由（v0.3.0 阶段 3B）。

提供：
- GET  /api/nanhu/jobs/page                渲染 nanhu_jobs.html（列表 + 过滤）
- GET  /api/nanhu/jobs                     JSON 列表（带 cluster / status 过滤）
- POST /api/nanhu/jobs/{job_id}/link-task   把 task_id 关联到南湖 job；
                                            可选 restart_index 同步写一条重启记录
- POST /api/nanhu/jobs/{job_id}/unlink-task 解绑（清 ri_task.nanhu_job_id）
- POST /api/nanhu/jobs/restart              重启南湖作业（dry-run 默认）

设计：
- 业务错误用 RolloutInsightException 抛出
- 参数错误用 Response.error(code=50001, msg=...) 直接返回
- 所有对 db/service 的导入在函数内**延迟**进行，避免 3A worker 还没合并时被 import 链拉断
- 所有写操作默认 dry-run，前端传 submit=true 时真正执行（向 nanhu_client 发请求）
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from starlette.templating import Jinja2Templates

from data.response import Response
from data.rollout_insight_exception import RolloutInsightException
from utils.config_loader import config

router = APIRouter()

templates = Jinja2Templates(directory=str(config.templates_dir))


# ---------------------------------------------------------------------------
# 延迟导入辅助
# ---------------------------------------------------------------------------

def _safe_import_attr(module_name: str, attr: str):
    """延迟导入 module_name.attr；导入失败返回 None，不抛错。"""
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return getattr(mod, attr, None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("延迟导入 {}.{} 失败: {}", module_name, attr, exc)
        return None


def _session_factory():
    return _safe_import_attr("db.db_element", "Session")


def _task_model():
    return _safe_import_attr("db.db_task", "DBTask")


def _nanhu_job_service():
    return _safe_import_attr("db.db_nanhu_job_service", "DBNanhuJobService")


def _nanhu_restart_service():
    return _safe_import_attr("db.db_nanhu_restart_service", "DBNanhuRestartService")


def _nanhu_client_module():
    """返回 service.nanhu_client 模块本身；模块不存在时返回 None。

    service.nanhu_client 是一个**模块**（不是类），里面挂着 list_train_jobs /
    restart_job 等函数。因此延迟导入策略是：先 import 出 module，再按需
    getattr(func_name)。
    """
    try:
        import importlib
        mod = importlib.import_module("service.nanhu_client")
        return mod
    except Exception as exc:  # noqa: BLE001
        logger.warning("延迟导入 {}.{} 失败: {}", "service.nanhu_client", "*", exc)
        return None


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------

def _job_to_dict(job: Any) -> Dict[str, Any]:
    """ORM DBNanhuJob → 模板/JSON 友好字典。"""
    if job is None:
        return {}
    return {
        "primary_id": getattr(job, "primary_id", None),
        "job_id": getattr(job, "job_id", "") or "",
        "cluster": getattr(job, "cluster", "") or "",
        "name": getattr(job, "name", "") or "",
        "status": getattr(job, "status", "") or "",
        "project_uuid": getattr(job, "project_uuid", "") or "",
        "algorithm_uuid": getattr(job, "algorithm_uuid", "") or "",
        "output_dir": getattr(job, "output_dir", "") or "",
        "mounted_storages_json": getattr(job, "mounted_storages_json", "") or "",
        "create_time": getattr(job, "create_time", None),
        "update_time": getattr(job, "update_time", None),
    }


def _task_to_dict(task: Any) -> Dict[str, Any]:
    """ORM DBTask → 模板友好字典（仅 nanhu 块用到的字段）。"""
    if task is None:
        return {}
    return {
        "primary_id": getattr(task, "primary_id", None),
        "task_name": getattr(task, "task_name", "") or "",
        "status": getattr(task, "status", "") or "",
        "nanhu_job_id": getattr(task, "nanhu_job_id", "") or "",
    }


def _normalize_remote_job(payload: Dict[str, Any], cluster: str) -> Dict[str, Any]:
    """把 nanhu_client.list_train_jobs 返回的 dict 规整成 ORM upsert 字段。

    兼容多种命名风格（南湖 CLI / 旧版 REST / 测试 stub）— 优先匹配 camelCase + snake_case。
    """
    if not isinstance(payload, dict):
        return {}
    job_id = (
        payload.get("job_id")
        or payload.get("jobId")
        or payload.get("id")
        or payload.get("uuid")  # 南湖 CLI 默认字段
        or payload.get("jobInstanceId")
        or ""
    )
    name = (
        payload.get("name")
        or payload.get("display_name")
        or payload.get("displayName")
        or ""
    )
    status = (
        payload.get("status")
        or payload.get("state")
        or ""
    )
    project_uuid = (
        payload.get("project_uuid")
        or payload.get("projectUuid")
        or payload.get("project")
        or ""
    )
    algorithm_uuid = (
        payload.get("algorithm_uuid")
        or payload.get("algorithmUuid")
        or payload.get("algorithm")
        or ""
    )
    output_dir = (
        payload.get("output_dir")
        or payload.get("outputDir")
        or payload.get("output")
        or ""
    )
    mounted = (
        payload.get("mounted_storages")
        or payload.get("mountedStorages")
        or payload.get("storage")
        or []
    )
    if not isinstance(mounted, str):
        mounted_json = json.dumps(mounted, ensure_ascii=False)
    else:
        mounted_json = mounted or ""
    return {
        "job_id": str(job_id).strip(),
        "cluster": (payload.get("cluster") or cluster or "").strip(),
        "name": str(name or ""),
        "status": str(status or ""),
        "project_uuid": str(project_uuid or ""),
        "algorithm_uuid": str(algorithm_uuid or ""),
        "output_dir": str(output_dir or ""),
        "mounted_storages_json": mounted_json,
    }


def _call_list_train_jobs(
    client_module: Any,
    cluster: Optional[str],
    status: Optional[str],
) -> List[Dict[str, Any]]:
    """调用 nanhu_client.list_train_jobs；统一异常收口为 RolloutInsightException。

    返回：原始 payload 字典列表；若上游异常则抛 NANHU_RESOURCE_QUERY_FAILED。
    """
    if client_module is None:
        return []
    # 兼容两种形态：模块含 list_train_jobs 函数 / 类含 list_train_jobs 方法
    func = getattr(client_module, "list_train_jobs", None)
    if func is None:
        return []
    try:
        result = func(cluster=cluster, status=status)
    except RolloutInsightException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("nanhu_client.list_train_jobs 调用失败: {}", exc)
        raise RolloutInsightException(
            RolloutInsightException.NANHU_RESOURCE_QUERY_FAILED,
            f"调用 list_train_jobs 失败: {exc}",
        ) from exc
    if result is None:
        return []
    if not isinstance(result, list):
        # 兼容 dict 包了 jobs 字段的形式
        if isinstance(result, dict):
            inner = result.get("jobs") or result.get("data") or []
            return [r for r in inner if isinstance(r, dict)]
        return []
    return [r for r in result if isinstance(r, dict)]


def _call_restart_job(
    client_module: Any,
    job_id: str,
    cluster: str,
    submit: bool,
) -> Dict[str, Any]:
    """调用 nanhu_client.restart_job；submit=False 时返回 dry-run 结果。

    返回：原始 payload 字典（含 new_job_id 等）。
    """
    if client_module is None:
        raise RolloutInsightException(
            RolloutInsightException.NANHU_ERROR,
            "nanhu_client 模块尚未就绪（3A 阶段未完成）",
        )
    func = getattr(client_module, "restart_job", None)
    if func is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "nanhu_client.restart_job 不可用",
        )
    try:
        result = func(job_id=job_id, cluster=cluster, submit=submit)
    except RolloutInsightException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("nanhu_client.restart_job 调用失败: {}", exc)
        raise RolloutInsightException(
            RolloutInsightException.NANHU_JOB_RESTART_FAILED,
            f"调用 restart_job 失败: {exc}",
        ) from exc
    if isinstance(result, dict):
        return result
    return {"ok": True, "raw": str(result)}


# ---------------------------------------------------------------------------
# 路由：HTML
# ---------------------------------------------------------------------------

@router.get("/api/nanhu/jobs/page", include_in_schema=False)
async def page_nanhu_jobs(
    request: Request,
    cluster: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
) -> HTMLResponse:
    """渲染 nanhu_jobs.html（页面 + 表 + 过滤）。"""
    jobs: List[Dict[str, Any]] = []
    warning: Optional[str] = None
    Session = _session_factory()
    Svc = _nanhu_job_service()
    if Session is None or Svc is None:
        warning = "数据库尚未就绪（3A 阶段未完成）"
    else:
        try:
            with Session() as session:
                svc = Svc()
                rows = svc.list_jobs(cluster=cluster, status=status, limit=200)
                # 同步尝试从 nanhu_client 拉新数据并 upsert
                client_module = _nanhu_client_module()
                if client_module is not None:
                    try:
                        remote_jobs = _call_list_train_jobs(client_module, cluster, status)
                        for rj in remote_jobs:
                            norm = _normalize_remote_job(rj, cluster=cluster or "")
                            if norm.get("job_id"):
                                svc.upsert(**norm)
                        # 二次查询（基于最新 upsert）
                        rows = svc.list_jobs(cluster=cluster, status=status, limit=200)
                    except RolloutInsightException as exc:
                        warning = f"南湖 list 失败：{exc.message}"
                jobs = [_job_to_dict(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.exception("加载南湖作业列表失败: {}", exc)
            warning = f"加载失败：{exc}"

    return templates.TemplateResponse(
        request,
        "nanhu_jobs.html",
        {
            "page_title": "南湖作业",
            "jobs": jobs,
            "filter_cluster": cluster or "",
            "filter_status": status or "",
            "warning": warning,
        },
    )


# ---------------------------------------------------------------------------
# 路由：JSON
# ---------------------------------------------------------------------------

@router.get("/api/nanhu/jobs", response_model=None)
async def api_list_nanhu_jobs(
    request: Request,
    cluster: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    refresh: bool = Query(default=False, description="是否先从南湖侧拉新再返回"),
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> JSONResponse:
    """JSON 列表；refresh=1 会先调 nanhu_client.list_train_jobs 同步落库。"""
    Session = _session_factory()
    Svc = _nanhu_job_service()
    if Session is None or Svc is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "数据库尚未就绪（3A 阶段未完成）",
        )

    synced = 0
    sync_error: Optional[str] = None
    client_module = _nanhu_client_module()

    try:
        with Session() as session:
            svc = Svc()
            if refresh:
                if client_module is None:
                    sync_error = "nanhu_client 模块尚未就绪（3A 阶段未完成）"
                else:
                    try:
                        remote_jobs = _call_list_train_jobs(client_module, cluster, status)
                        for rj in remote_jobs:
                            norm = _normalize_remote_job(rj, cluster=cluster or "")
                            if norm.get("job_id"):
                                svc.upsert(**norm)
                                synced += 1
                    except RolloutInsightException as exc:
                        sync_error = exc.message

            rows = svc.list_jobs(
                cluster=cluster, status=status, limit=limit, offset=offset
            )
            jobs = [_job_to_dict(r) for r in rows]
        return Response.success(
            msg="ok",
            data={
                "jobs": jobs,
                "total": len(jobs),
                "synced": synced,
                "sync_error": sync_error,
                "cluster": cluster or "",
                "status": status or "",
            },
        )
    except RolloutInsightException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("查询 nanhu job 列表失败: {}", exc)
        raise RolloutInsightException(
            RolloutInsightException.DB_ERROR, f"查询 nanhu job 列表失败: {exc}"
        ) from exc


@router.post("/api/nanhu/jobs/{job_id}/link-task")
async def api_link_task(
    job_id: str,
    request: Request,
    submit: bool = Query(default=False, description="是否真正发起动作；默认 dry-run"),
) -> JSONResponse:
    """关联 task 到南湖 job。

    - 默认 dry-run：仅返回将要发生的操作描述，不写库
    - submit=true：写 ri_task.nanhu_job_id 与 ri_nanhu_restart_table
    """
    job_id = (job_id or "").strip()
    if not job_id:
        return Response.error(code=50001, msg="job_id 不能为空")

    # body 解析（兼容 JSON / form-encoded）
    body_task_id: Optional[int] = None
    body_restart_index: Optional[int] = None
    body_cluster: Optional[str] = None
    raw_ct = request.headers.get("content-type", "")
    try:
        if "application/json" in raw_ct:
            raw = await request.json()
            if isinstance(raw, dict):
                body_task_id = raw.get("task_id") or raw.get("taskId")
                body_restart_index = raw.get("restart_index") or raw.get("restartIndex")
                body_cluster = raw.get("cluster")
        else:
            form = await request.form()
            if "task_id" in form:
                body_task_id = form.get("task_id")
            if "restart_index" in form:
                body_restart_index = form.get("restart_index")
            if "cluster" in form:
                body_cluster = form.get("cluster")
    except Exception as exc:  # noqa: BLE001
        logger.warning("解析 link-task body 失败: {}", exc)

    try:
        body_task_id_int = int(body_task_id) if body_task_id not in (None, "") else None
    except (TypeError, ValueError):
        return Response.error(code=50001, msg=f"task_id 必须为整数：{body_task_id}")
    try:
        body_restart_index_int = (
            int(body_restart_index) if body_restart_index not in (None, "") else None
        )
    except (TypeError, ValueError):
        return Response.error(
            code=50001, msg=f"restart_index 必须为整数：{body_restart_index}"
        )
    if body_task_id_int is None:
        return Response.error(code=50001, msg="task_id 必填")

    if not submit:
        # dry-run：仅返回动作描述
        return Response.success(
            msg="dry-run：尚未真正落库",
            data={
                "dry_run": True,
                "action": "link-task",
                "job_id": job_id,
                "task_id": body_task_id_int,
                "restart_index": body_restart_index_int,
                "cluster": body_cluster or "",
            },
        )

    # 真正写库
    Session = _session_factory()
    Task = _task_model()
    JobSvc = _nanhu_job_service()
    RestartSvc = _nanhu_restart_service()
    if Session is None or Task is None or JobSvc is None or RestartSvc is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "数据库或 Service 尚未就绪（3A 阶段未完成）",
        )
    try:
        with Session() as session:
            task = (
                session.query(Task)
                .filter(Task.primary_id == body_task_id_int)
                .first()
            )
            if task is None:
                raise RolloutInsightException(
                    RolloutInsightException.TASK_NOT_FOUND,
                    f"task #{body_task_id_int} 不存在",
                )
            # 校验 nanhu job 行存在（list 时会先 upsert，所以这里必须已经同步过）
            job_svc = JobSvc()
            job_row = job_svc.get_by_job_id(job_id)
            if job_row is None:
                raise RolloutInsightException(
                    RolloutInsightException.NANHU_VALIDATION,
                    f"nanhu job_id={job_id} 未入库，请先在列表页同步（refresh=1）",
                )

            task.nanhu_job_id = job_id
            task.update_time = task.update_time  # 不显式改时间，让 onupdate 触发
            session.commit()

            restart_row: Optional[Dict[str, Any]] = None
            if body_restart_index_int is not None:
                restart_row = RestartSvc.create_record(
                    nanhu_job_id=job_id,
                    restart_status="REQUESTED",
                    related_task_id=body_task_id_int,
                    restart_index=body_restart_index_int,
                ).to_dict()

        return Response.success(
            msg="已关联",
            data={
                "dry_run": False,
                "job_id": job_id,
                "task_id": body_task_id_int,
                "restart": restart_row,
            },
        )
    except RolloutInsightException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("link-task 失败 job={} task={}", job_id, body_task_id_int)
        raise RolloutInsightException(
            RolloutInsightException.DB_ERROR, f"link-task 失败: {exc}"
        ) from exc


@router.post("/api/nanhu/jobs/{job_id}/unlink-task")
async def api_unlink_task(
    job_id: str,
    request: Request,
    submit: bool = Query(default=False, description="是否真正解绑；默认 dry-run"),
) -> JSONResponse:
    """解绑 task 与南湖 job。

    - 默认 dry-run：仅返回动作描述
    - submit=true：清空 ri_task.nanhu_job_id（仅当其值匹配当前 job_id 时）
    """
    job_id = (job_id or "").strip()
    if not job_id:
        return Response.error(code=50001, msg="job_id 不能为空")

    if not submit:
        return Response.success(
            msg="dry-run：尚未真正解绑",
            data={"dry_run": True, "action": "unlink-task", "job_id": job_id},
        )

    Session = _session_factory()
    Task = _task_model()
    if Session is None or Task is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "任务模型尚未就绪（3A 阶段未完成）",
        )
    try:
        affected = 0
        with Session() as session:
            q = (
                session.query(Task)
                .filter(Task.nanhu_job_id == job_id)
            )
            for t in q.all():
                t.nanhu_job_id = ""
                affected += 1
            session.commit()
        return Response.success(
            msg=f"已解绑 {affected} 个 task",
            data={"dry_run": False, "job_id": job_id, "affected_tasks": affected},
        )
    except RolloutInsightException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("unlink-task 失败 job={}", job_id)
        raise RolloutInsightException(
            RolloutInsightException.DB_ERROR, f"unlink-task 失败: {exc}"
        ) from exc


@router.post("/api/nanhu/jobs/restart")
async def api_restart_job(
    request: Request,
    submit: bool = Query(default=False, description="是否真正调用 restart_job"),
) -> JSONResponse:
    """重启南湖 job；默认 dry-run。

    body（form 或 JSON）：
        job_id: 必填
        cluster: 可选
        task_id: 可选（关联到重启记录 related_task_id）
    """
    # 解析 body
    raw_ct = request.headers.get("content-type", "")
    job_id: Optional[str] = None
    cluster: Optional[str] = None
    task_id: Optional[int] = None
    try:
        if "application/json" in raw_ct:
            raw = await request.json()
            if isinstance(raw, dict):
                job_id = (raw.get("job_id") or raw.get("jobId") or "").strip()
                cluster = (raw.get("cluster") or "").strip() or None
                t_raw = raw.get("task_id") or raw.get("taskId")
                if t_raw not in (None, ""):
                    task_id = int(t_raw)
        else:
            form = await request.form()
            job_id = (form.get("job_id") or "").strip()
            cluster = (form.get("cluster") or "").strip() or None
            t_raw = form.get("task_id")
            if t_raw not in (None, ""):
                task_id = int(t_raw)
    except (TypeError, ValueError) as exc:
        return Response.error(code=50001, msg=f"参数解析失败：{exc}")
    except Exception as exc:  # noqa: BLE001
        return Response.error(code=50001, msg=f"请求体解析失败：{exc}")

    if not job_id:
        return Response.error(code=50001, msg="job_id 必填")

    client_module = _nanhu_client_module()
    if client_module is None:
        # 没有 client 也要支持 dry-run：以便 UI 在 3A 未合时仍能展示
        if not submit:
            return Response.success(
                msg="dry-run：nanhu_client 尚未就绪，返回占位结果",
                data={
                    "dry_run": True,
                    "client_ready": False,
                    "action": "restart",
                    "job_id": job_id,
                    "cluster": cluster or "",
                },
            )
        raise RolloutInsightException(
            RolloutInsightException.NANHU_ERROR,
            "nanhu_client 尚未就绪（3A 阶段未完成）",
        )

    if not submit:
        return Response.success(
            msg="dry-run：未真正调用 restart_job",
            data={
                "dry_run": True,
                "client_ready": True,
                "action": "restart",
                "job_id": job_id,
                "cluster": cluster or "",
                "task_id": task_id,
            },
        )

    # 真正调用
    try:
        remote_result = _call_restart_job(
            client_module, job_id=job_id, cluster=cluster or "", submit=True
        )
    except RolloutInsightException as exc:
        # 同步落 restart 失败记录，便于审计
        try:
            RestartSvc = _nanhu_restart_service()
            if RestartSvc is not None:
                RestartSvc.create_record(
                    nanhu_job_id=job_id,
                    restart_status="FAILED",
                    related_task_id=task_id,
                )
        except Exception:  # noqa: BLE001
            pass
        raise

    # 成功后写一条 restart 记录
    RestartSvc = _nanhu_restart_service()
    restart_row: Optional[Dict[str, Any]] = None
    if RestartSvc is not None:
        try:
            restart_row = RestartSvc.create_record(
                nanhu_job_id=job_id,
                restart_status="SUBMITTED",
                related_task_id=task_id,
            ).to_dict()
        except Exception as exc:  # noqa: BLE001
            logger.warning("写入 restart 记录失败（不影响重启结果）: {}", exc)

    return Response.success(
        msg="已发起重启",
        data={
            "dry_run": False,
            "action": "restart",
            "job_id": job_id,
            "cluster": cluster or "",
            "task_id": task_id,
            "remote": remote_result,
            "restart": restart_row,
        },
    )