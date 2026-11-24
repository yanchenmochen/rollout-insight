"""南湖（nanhu）作业调度相关路由（v0.3.0 阶段 3B）。

提供：
- GET  /api/nanhu/jobs/page                渲染 nanhu_jobs.html（列表 + 过滤）
- GET  /api/nanhu/jobs                     JSON 列表（带 cluster / status 过滤）
- GET  /api/nanhu/jobs/{job_id}/resolve-for-task
                                           按 job 详情解析 task_name / rollout_root / tokenizer_dir
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
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

from data.response import Response
from data.rollout_insight_exception import RolloutInsightException
from utils.auth_session import current_user, require_login
from utils.config_loader import config
from utils.template_factory import create_templates

# v0.5.1：业务路由统一要求登录
router = APIRouter(dependencies=[Depends(require_login)])

templates = create_templates()


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
        "algorithm_name": getattr(job, "algorithm_name", "") or "",
        "creator": getattr(job, "creator", "") or "",
        "creator_name": getattr(job, "creator_name", "") or "",
        "algo_creator": getattr(job, "algo_creator", "") or "",
        "algo_creator_name": getattr(job, "algo_creator_name", "") or "",
        "is_public": int(getattr(job, "is_public", 0) or 0),
        "shared_users_json": getattr(job, "shared_users_json", "") or "",
        "output_dir": getattr(job, "output_dir", "") or "",
        "mounted_storages_json": getattr(job, "mounted_storages_json", "") or "",
        "create_time": getattr(job, "create_time", None),
        "update_time": getattr(job, "update_time", None),
        "related_task_id": None,
        "related_task_name": "",
        "is_shared": False,
    }


def _task_service():
    return _safe_import_attr("db.db_task_service", "DBTaskService")


def enrich_jobs_with_task_links(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """批量回填 ``related_task_id`` / ``related_task_name``（按 ri_task.nanhu_job_id）。"""
    if not jobs:
        return jobs
    TaskSvc = _task_service()
    if TaskSvc is None:
        return jobs
    job_ids = [str(j.get("job_id") or "").strip() for j in jobs]
    try:
        mapping = TaskSvc.list_by_nanhu_job_ids(job_ids)
    except Exception as exc:  # noqa: BLE001
        logger.warning("enrich related_task 失败: {}", exc)
        return jobs
    for j in jobs:
        key = str(j.get("job_id") or "").strip()
        info = mapping.get(key) if key else None
        if info:
            j["related_task_id"] = info.get("primary_id")
            j["related_task_name"] = info.get("task_name") or ""
        else:
            j["related_task_id"] = None
            j["related_task_name"] = ""
    return jobs


def attach_algo_kinds(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """为每条 job 写入 ``algo_kind`` / ``algo_label``。"""
    from utils.nanhu_job_kind import attach_nanhu_job_kind

    for j in jobs:
        attach_nanhu_job_kind(j)
    return jobs


def filter_jobs_by_algo_kind(
    jobs: List[Dict[str, Any]],
    algo_kind: Optional[str],
) -> List[Dict[str, Any]]:
    """按 ``algo_kind`` 过滤；空 / ``all`` 表示不过滤。"""
    kind = (algo_kind or "").strip().lower()
    if not kind or kind == "all":
        return jobs
    return [j for j in jobs if (j.get("algo_kind") or "") == kind]


def filter_jobs_by_algo_query(
    jobs: List[Dict[str, Any]],
    algo_query: Optional[str],
) -> List[Dict[str, Any]]:
    """按算法库自由文本过滤（命中 label / kind / uuid / 真实算法名 / 名称 / 路径）。"""
    raw = (algo_query or "").strip().lower()
    if not raw:
        return jobs
    variants = {
        raw,
        raw.replace(" ", "-"),
        raw.replace(" ", "_"),
        raw.replace("-", " "),
        raw.replace("_", " "),
    }
    out: List[Dict[str, Any]] = []
    for j in jobs:
        hay = " ".join(
            [
                str(j.get("algo_label") or ""),
                str(j.get("algo_kind") or ""),
                str(j.get("algorithm_uuid") or ""),
                str(j.get("algorithm_name") or ""),
                str(j.get("name") or ""),
                str(j.get("output_dir") or ""),
            ]
        ).lower()
        if any(v and v in hay for v in variants):
            out.append(j)
    return out


def filter_jobs_by_creator(
    jobs: List[Dict[str, Any]],
    creator: Optional[str],
) -> List[Dict[str, Any]]:
    """按创建人 userId / 显示名模糊过滤。"""
    raw = (creator or "").strip().lower()
    if not raw:
        return jobs
    out: List[Dict[str, Any]] = []
    for j in jobs:
        hay = f"{j.get('creator') or ''} {j.get('creator_name') or ''}".lower()
        if raw in hay:
            out.append(j)
    return out


def mark_shared_jobs(
    jobs: List[Dict[str, Any]],
    *,
    local_nanhu_user_id: Optional[str] = None,
    shared_algo_uuids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """标记 ``is_shared``：共享算法 / 非本人创建 / 公开作业。

    「共享给我的」口径：
    - 算法 UUID 落在共享算法列表；或
    - 算法创建人 ≠ 当前南湖 userId；或
    - 作业创建人 ≠ 当前南湖 userId；或
    - ``is_public``。
    """
    me = (local_nanhu_user_id or "").strip()
    shared_set = shared_algo_uuids or set()
    for j in jobs:
        algo_uuid = str(j.get("algorithm_uuid") or "").strip()
        creator = str(j.get("creator") or "").strip()
        algo_creator = str(j.get("algo_creator") or "").strip()
        is_public = bool(int(j.get("is_public") or 0))
        shared = False
        if algo_uuid and algo_uuid in shared_set:
            shared = True
        elif me and algo_creator and algo_creator != me:
            shared = True
        elif me and creator and creator != me:
            shared = True
        elif is_public:
            shared = True
        j["is_shared"] = shared
    return jobs


def filter_jobs_by_share_scope(
    jobs: List[Dict[str, Any]],
    share_scope: Optional[str],
) -> List[Dict[str, Any]]:
    """``mine`` / ``shared`` / ``all``（默认）。"""
    scope = (share_scope or "").strip().lower()
    if not scope or scope == "all":
        return jobs
    if scope == "shared":
        return [j for j in jobs if j.get("is_shared")]
    if scope in ("mine", "owned", "self"):
        return [j for j in jobs if not j.get("is_shared")]
    return jobs


def prepare_nanhu_job_list(
    rows: List[Any],
    *,
    algo_kind: Optional[str] = None,
    algo_query: Optional[str] = None,
    creator: Optional[str] = None,
    share_scope: Optional[str] = None,
    local_nanhu_user_id: Optional[str] = None,
    shared_algo_uuids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """ORM 行 → dict → 关联任务 enrichment → 算法库分类 → 共享标记 → 可选过滤。"""
    jobs = [_job_to_dict(r) for r in rows]
    enrich_jobs_with_task_links(jobs)
    attach_algo_kinds(jobs)
    mark_shared_jobs(
        jobs,
        local_nanhu_user_id=local_nanhu_user_id,
        shared_algo_uuids=shared_algo_uuids,
    )
    jobs = filter_jobs_by_algo_kind(jobs, algo_kind)
    jobs = filter_jobs_by_algo_query(jobs, algo_query)
    jobs = filter_jobs_by_creator(jobs, creator)
    return filter_jobs_by_share_scope(jobs, share_scope)


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


# 南湖作业 UUID（与 CLI ``jobs detail --job-id`` 同源）；列表接口不下发共享作业时靠详情补齐
_JOB_ID_TOKEN_RE = re.compile(
    r"\b(jb-(?:aitrain|aideploy|aieval)-[0-9a-zA-Z_-]+|jb-[0-9a-zA-Z_-]{8,})\b",
    re.IGNORECASE,
)


def extract_job_id_tokens(text: Optional[str]) -> List[str]:
    """从关键字 / 粘贴文本中提取南湖 job UUID（保序去重）。"""
    if not text:
        return []
    seen: set = set()
    out: List[str] = []
    for match in _JOB_ID_TOKEN_RE.finditer(str(text)):
        token = match.group(1).strip()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def merge_job_id_filters(
    keyword: Optional[str] = None,
    *extra_texts: Optional[str],
) -> Tuple[Optional[str], List[str]]:
    """合并 keyword / job_ids 粘贴文本中的 UUID。

    若解析出至少 1 个 UUID：返回 ``(None, ids)``，列表应按 ``job_ids`` 精确过滤，
    避免整串 ``LIKE '%jb-1 jb-2%'`` 匹配失败。
    否则返回 ``(keyword_stripped_or_None, [])``。
    """
    chunks: List[str] = []
    if keyword:
        chunks.append(str(keyword))
    for t in extra_texts:
        if t:
            chunks.append(str(t))
    ids: List[str] = []
    seen: set = set()
    for chunk in chunks:
        for token in extract_job_id_tokens(chunk):
            if token not in seen:
                seen.add(token)
                ids.append(token)
    if ids:
        return None, ids
    kw = (keyword or "").strip() or None
    return kw, []


def _upsert_job_via_detail(
    *,
    svc: Any,
    client_module: Any,
    cluster: str,
    job_id: str,
    local_user_id: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """走与 ``nanhu jobs detail --job-id`` 相同的 getJobDetail，写入本地缓存。

    :returns: ``(row_summary, error_message)``；成功时 error 为 None。
    """
    from utils.nanhu_job_path_resolve import extract_job_payload

    jid = (job_id or "").strip()
    if not jid:
        return None, "job_id 为空"
    if client_module is None or not hasattr(client_module, "get_job_detail"):
        return None, "nanhu_client 不可用"
    try:
        detail = client_module.get_job_detail(
            cluster=cluster or "x10000",
            job_id=jid,
            local_user_id=local_user_id,
        )
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    if not isinstance(detail, dict):
        return None, "详情返回格式异常"
    payload = extract_job_payload(detail)
    if payload is None:
        err = (
            detail.get("error")
            or detail.get("message")
            or detail.get("msg")
            or "作业找不到或无权限"
        )
        return None, str(err)
    if not payload.get("uuid") and not payload.get("job_id"):
        payload["uuid"] = jid
    norm = _normalize_remote_job(dict(payload), cluster=cluster or "x10000")
    if not norm.get("job_id"):
        return None, "无法解析 job_id"
    try:
        row = svc.upsert(**norm)
    except Exception as exc:  # noqa: BLE001
        return None, f"写入失败: {exc}"
    return (
        {
            "job_id": getattr(row, "job_id", norm.get("job_id")),
            "name": getattr(row, "name", "") or norm.get("name") or "",
            "creator_name": getattr(row, "creator_name", "")
            or norm.get("creator_name")
            or "",
            "algorithm_name": getattr(row, "algorithm_name", "")
            or norm.get("algorithm_name")
            or "",
        },
        None,
    )


def _fetch_jobs_by_detail_ids(
    *,
    svc: Any,
    client_module: Any,
    cluster: str,
    job_ids: List[str],
    local_user_id: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """批量按 UUID 走 getJobDetail 落库（共享作业绕过 jobList）。"""
    imported: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []
    for jid in job_ids:
        summary, err = _upsert_job_via_detail(
            svc=svc,
            client_module=client_module,
            cluster=cluster,
            job_id=jid,
            local_user_id=local_user_id,
        )
        if err:
            failed.append({"job_id": jid, "error": err})
        elif summary:
            imported.append(summary)
    return imported, failed


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
    algo_obj = payload.get("algorithm")
    if not isinstance(algo_obj, dict):
        algo_obj = {}
    algorithm_uuid = (
        payload.get("algorithm_uuid")
        or payload.get("algorithmUuid")
        or algo_obj.get("uuid")
        or (payload.get("algorithm") if isinstance(payload.get("algorithm"), str) else "")
        or ""
    )
    algorithm_name = (
        payload.get("algorithm_name")
        or payload.get("algorithmName")
        or algo_obj.get("name")
        or ""
    )
    creator = payload.get("creator") or payload.get("createUser") or ""
    creator_name = (
        payload.get("creator_name")
        or payload.get("creatorName")
        or payload.get("createUserName")
        or ""
    )
    algo_creator = algo_obj.get("creator") or payload.get("algo_creator") or ""
    algo_creator_name = (
        algo_obj.get("creatorName")
        or algo_obj.get("creator_name")
        or payload.get("algo_creator_name")
        or ""
    )
    raw_public = payload.get("is_public")
    if raw_public is None:
        raw_public = payload.get("isPublic")
    if raw_public is None:
        raw_public = algo_obj.get("isPublic")
    is_public = 1 if raw_public in (True, 1, "1", "true", "True") else 0
    shared_users = (
        payload.get("shared_users")
        or payload.get("sharedUsers")
        or payload.get("shareUsers")
        or []
    )
    if not isinstance(shared_users, str):
        shared_users_json = json.dumps(shared_users or [], ensure_ascii=False)
    else:
        shared_users_json = shared_users or ""
    raw_output = (
        payload.get("output_dir")
        or payload.get("outputDir")
        or payload.get("output")
        or ""
    )
    if isinstance(raw_output, dict):
        output_dir = (
            raw_output.get("outputPath")
            or raw_output.get("path")
            or ""
        )
    else:
        output_dir = raw_output
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
        "algorithm_uuid": str(algorithm_uuid or "").strip(),
        "algorithm_name": str(algorithm_name or "").strip(),
        "creator": str(creator or "").strip(),
        "creator_name": str(creator_name or "").strip(),
        "algo_creator": str(algo_creator or "").strip(),
        "algo_creator_name": str(algo_creator_name or "").strip(),
        "is_public": is_public,
        "shared_users_json": shared_users_json,
        "output_dir": str(output_dir or ""),
        "mounted_storages_json": mounted_json,
    }


def _call_list_train_jobs(
    client_module: Any,
    cluster: Optional[str],
    status: Optional[str],
    local_user_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """调用 nanhu_client.list_train_jobs；统一异常收口为 RolloutInsightException。

    返回：原始 payload 字典列表；若上游异常则抛 NANHU_RESOURCE_QUERY_FAILED。
    必须传入 ``local_user_id``，才能用当前登录用户自己的 app_key / app_secret。
    """
    if client_module is None:
        return []
    # 兼容两种形态：模块含 list_train_jobs 函数 / 类含 list_train_jobs 方法
    func = getattr(client_module, "list_train_jobs", None)
    if func is None:
        return []
    try:
        result = func(
            cluster=cluster,
            status=status,
            local_user_id=local_user_id,
        )
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


def _request_local_user_id(request: Request) -> Optional[int]:
    """从登录态取本地用户主键，供南湖凭据解析使用。

    单测无 SessionMiddleware 时安全返回 None，不抛 AssertionError。
    """
    try:
        if "session" not in getattr(request, "scope", {}):
            # 兼容 TestClient 未挂 SessionMiddleware 的场景
            state_user = getattr(getattr(request, "state", None), "current_user", None)
            if state_user is not None:
                return getattr(state_user, "primary_id", None)
            return None
        user = current_user(request)
    except Exception:  # noqa: BLE001
        return None
    if user is None:
        return None
    return getattr(user, "primary_id", None)


def _resolve_nanhu_user_and_shared_algos(
    client_module: Any,
    cluster: Optional[str],
    local_user_id: Optional[int],
) -> tuple:
    """返回 ``(nanhu_user_id, shared_algo_uuid_set)``。"""
    nanhu_uid = ""
    shared: set = set()
    if client_module is None:
        return nanhu_uid, shared
    resolve_uid = getattr(client_module, "resolve_nanhu_user_id", None)
    if callable(resolve_uid):
        try:
            nanhu_uid = (resolve_uid(cluster=cluster or "", local_user_id=local_user_id) or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("resolve_nanhu_user_id 失败: {}", exc)
    list_shared = getattr(client_module, "list_shared_algorithms", None)
    if callable(list_shared):
        try:
            rows = list_shared(cluster=cluster or "", local_user_id=local_user_id) or []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                uid = str(row.get("uuid") or row.get("algorithm_uuid") or "").strip()
                if uid:
                    shared.add(uid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_shared_algorithms 失败: {}", exc)
    return nanhu_uid, shared


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
    keyword: Optional[str] = Query(
        default=None,
        description="任务名称关键字；也可粘贴一组 jb-… UUID（空格/换行/逗号分隔）",
    ),
    job_ids: Optional[str] = Query(
        default=None,
        description="批量 job UUID（逗号/空白分隔）；与 keyword 中的 UUID 合并",
    ),
    creator: Optional[str] = Query(
        default=None,
        description="创建人过滤（userId 或显示名）",
    ),
    algo_kind: Optional[str] = Query(
        default="slime",
        description="算法库类型：slime/eval/convert/other/all；默认 slime",
    ),
    algo_query: Optional[str] = Query(
        default=None,
        description="算法库名称检索（如 agentic rl / nanhu-slime）",
    ),
    share_scope: Optional[str] = Query(
        default="all",
        description="范围：all / mine / shared（共享算法或非本人创建）",
    ),
) -> HTMLResponse:
    """渲染 nanhu_jobs.html（页面 + 表 + 过滤）。"""
    jobs: List[Dict[str, Any]] = []
    warning: Optional[str] = None
    filter_algo = (algo_kind if algo_kind is not None else "slime") or "slime"
    filter_kw, detail_ids = merge_job_id_filters(keyword, job_ids)
    filter_algo_q = (algo_query or "").strip()
    filter_creator = (creator or "").strip()
    filter_scope = (share_scope or "all").strip() or "all"
    local_uid = _request_local_user_id(request)
    Session = _session_factory()
    Svc = _nanhu_job_service()
    if Session is None or Svc is None:
        warning = "数据库尚未就绪（3A 阶段未完成）"
    else:
        try:
            with Session() as session:
                svc = Svc()
                rows = svc.list_jobs(
                    cluster=cluster,
                    status=status,
                    keyword=filter_kw,
                    creator=filter_creator or None,
                    job_ids=detail_ids or None,
                    limit=500,
                )
                client_module = _nanhu_client_module()
                nanhu_uid = ""
                shared_algos: set = set()
                if client_module is not None:
                    try:
                        remote_jobs = _call_list_train_jobs(
                            client_module,
                            cluster,
                            status,
                            local_user_id=local_uid,
                        )
                        for rj in remote_jobs:
                            norm = _normalize_remote_job(rj, cluster=cluster or "")
                            if norm.get("job_id"):
                                svc.upsert(**norm)
                        # 一组 jb-UUID 时走 getJobDetail（与 CLI jobs detail 同源）
                        if detail_ids:
                            _fetch_jobs_by_detail_ids(
                                svc=svc,
                                client_module=client_module,
                                cluster=(cluster or "").strip() or "x10000",
                                job_ids=detail_ids,
                                local_user_id=local_uid,
                            )
                        rows = svc.list_jobs(
                            cluster=cluster,
                            status=status,
                            keyword=filter_kw,
                            creator=filter_creator or None,
                            job_ids=detail_ids or None,
                            limit=500,
                        )
                        nanhu_uid, shared_algos = _resolve_nanhu_user_and_shared_algos(
                            client_module, cluster, local_uid
                        )
                    except RolloutInsightException as exc:
                        warning = f"南湖 list 失败：{exc.message}"
                jobs = prepare_nanhu_job_list(
                    rows,
                    algo_kind=filter_algo,
                    algo_query=filter_algo_q or None,
                    creator=filter_creator or None,
                    share_scope=filter_scope,
                    local_nanhu_user_id=nanhu_uid or None,
                    shared_algo_uuids=shared_algos or None,
                )
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
            "filter_keyword": (keyword or "").strip(),
            "filter_job_ids": "\n".join(detail_ids) if detail_ids else "",
            "filter_creator": filter_creator,
            "filter_algo_kind": filter_algo,
            "filter_algo_query": filter_algo_q,
            "filter_share_scope": filter_scope,
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
    keyword: Optional[str] = Query(
        default=None,
        description="任务名称关键字；或粘贴一组 jb-… UUID（空格/换行/逗号分隔）",
    ),
    job_ids: Optional[str] = Query(
        default=None,
        description="批量 job UUID（逗号/空白/换行分隔）；与 keyword 中的 UUID 合并后精确过滤",
    ),
    creator: Optional[str] = Query(
        default=None,
        description="创建人过滤（userId 或显示名）",
    ),
    algo_kind: Optional[str] = Query(
        default=None,
        description="算法库类型：slime/eval/convert/other/all；缺省不过滤",
    ),
    algo_query: Optional[str] = Query(
        default=None,
        description="算法库名称检索（如 agentic rl / nanhu-slime）",
    ),
    share_scope: Optional[str] = Query(
        default=None,
        description="范围：all / mine / shared",
    ),
    refresh: bool = Query(default=False, description="是否先从南湖侧拉新再返回"),
    limit: int = Query(default=500, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> JSONResponse:
    """JSON 列表；refresh=1 会先调 nanhu_client.list_train_jobs 同步落库。

    同步时使用当前登录用户的 app_key / app_secret（``local_user_id``），并翻页拉全量。
    若 ``keyword`` / ``job_ids`` 含一组 UUID，则按 ``getJobDetail`` 批量补齐后再精确过滤，
    便于同事共享作业批量关联。
    """
    Session = _session_factory()
    Svc = _nanhu_job_service()
    if Session is None or Svc is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "数据库尚未就绪（3A 阶段未完成）",
        )

    synced = 0
    detail_imported = 0
    sync_error: Optional[str] = None
    client_module = _nanhu_client_module()
    filter_algo = (algo_kind or "").strip() or None
    filter_kw, detail_ids = merge_job_id_filters(keyword, job_ids)
    filter_algo_q = (algo_query or "").strip() or None
    filter_creator = (creator or "").strip() or None
    filter_scope = (share_scope or "").strip() or "all"
    local_uid = _request_local_user_id(request)
    nanhu_uid = ""
    shared_algos: set = set()

    try:
        with Session() as session:
            svc = Svc()
            if refresh:
                if client_module is None:
                    sync_error = "nanhu_client 模块尚未就绪（3A 阶段未完成）"
                else:
                    try:
                        remote_jobs = _call_list_train_jobs(
                            client_module,
                            cluster,
                            status,
                            local_user_id=local_uid,
                        )
                        for rj in remote_jobs:
                            norm = _normalize_remote_job(rj, cluster=cluster or "")
                            if norm.get("job_id"):
                                svc.upsert(**norm)
                                synced += 1
                    except RolloutInsightException as exc:
                        sync_error = exc.message

            if detail_ids and client_module is not None:
                imported_rows, failed_rows = _fetch_jobs_by_detail_ids(
                    svc=svc,
                    client_module=client_module,
                    cluster=(cluster or "").strip() or "x10000",
                    job_ids=detail_ids,
                    local_user_id=local_uid,
                )
                detail_imported = len(imported_rows)
                if failed_rows and not sync_error:
                    sync_error = (
                        "按 UUID 拉取详情失败："
                        + "; ".join(
                            f"{x.get('job_id')}: {x.get('error')}" for x in failed_rows[:3]
                        )
                    )

            if client_module is not None:
                nanhu_uid, shared_algos = _resolve_nanhu_user_and_shared_algos(
                    client_module, cluster, local_uid
                )

            rows = svc.list_jobs(
                cluster=cluster,
                status=status,
                keyword=filter_kw,
                creator=filter_creator,
                job_ids=detail_ids or None,
                limit=limit,
                offset=offset,
            )
            jobs = prepare_nanhu_job_list(
                rows,
                algo_kind=filter_algo,
                algo_query=filter_algo_q,
                creator=filter_creator,
                share_scope=filter_scope,
                local_nanhu_user_id=nanhu_uid or None,
                shared_algo_uuids=shared_algos or None,
            )
        return Response.success(
            msg="ok",
            data={
                "jobs": jobs,
                "total": len(jobs),
                "synced": synced,
                "detail_imported": detail_imported,
                "job_ids": detail_ids,
                "sync_error": sync_error,
                "cluster": cluster or "",
                "status": status or "",
                "keyword": (keyword or "").strip(),
                "creator": filter_creator or "",
                "algo_kind": filter_algo or "all",
                "algo_query": filter_algo_q or "",
                "share_scope": filter_scope,
                "shared_algo_count": len(shared_algos),
                "nanhu_user_id": nanhu_uid,
            },
        )
    except RolloutInsightException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("查询 nanhu job 列表失败: {}", exc)
        raise RolloutInsightException(
            RolloutInsightException.DB_ERROR, f"查询 nanhu job 列表失败: {exc}"
        ) from exc


@router.post("/api/nanhu/jobs/import")
async def api_import_nanhu_jobs(
    request: Request,
    job_ids: Optional[str] = Form(
        default=None,
        description="逗号/空白/换行分隔的南湖 job_id 列表",
    ),
    cluster: str = Form(default="x10000"),
) -> Response:
    """按 job_id 导入作业（绕过 jobList 只返回本人作业的限制）。

    南湖网页控制台能看到的「共享作业」往往不会出现在 OpenAPI ``jobList`` 里；
    但若已知 UUID，``getJobDetail`` 通常仍可拉取详情并写入本地缓存。
    支持 JSON body：``{"job_ids":["jb-..."], "cluster":"x10000"}``。
    """
    cluster_name = (cluster or "").strip() or "x10000"
    ids: List[str] = []

    # form 优先；否则尝试 JSON body
    raw = (job_ids or "").strip()
    if not raw:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = None
        if isinstance(body, dict):
            cluster_name = (body.get("cluster") or cluster_name or "").strip() or "x10000"
            payload_ids = body.get("job_ids") or body.get("ids") or body.get("job_id")
            if isinstance(payload_ids, str):
                raw = payload_ids
            elif isinstance(payload_ids, list):
                ids = [str(x).strip() for x in payload_ids if str(x).strip()]

    if raw and not ids:
        # 优先按 jb-… 模式抽取；否则按空白/逗号切分
        parsed = extract_job_id_tokens(raw)
        if parsed:
            ids = parsed
        else:
            for part in raw.replace(",", " ").replace(";", " ").replace("\n", " ").split():
                token = part.strip()
                if token:
                    ids.append(token)

    # 去重保序
    seen: set = set()
    uniq: List[str] = []
    for jid in ids:
        if jid not in seen:
            seen.add(jid)
            uniq.append(jid)
    ids = uniq

    if not ids:
        return Response.error(code=50001, msg="请提供至少一个南湖 job_id")
    if len(ids) > 100:
        return Response.error(code=50001, msg="单次最多导入 100 个 job_id")

    local_uid = _request_local_user_id(request)
    Session = _session_factory()
    Svc = _nanhu_job_service()
    client_module = _nanhu_client_module()
    if Session is None or Svc is None:
        raise RolloutInsightException(
            RolloutInsightException.NOT_IMPLEMENTED,
            "数据库尚未就绪",
        )
    if client_module is None or not hasattr(client_module, "get_job_detail"):
        raise RolloutInsightException(
            RolloutInsightException.NANHU_RESOURCE_QUERY_FAILED,
            "nanhu_client 不可用，无法导入作业",
        )

    with Session() as session:
        svc = Svc()
        imported, failed = _fetch_jobs_by_detail_ids(
            svc=svc,
            client_module=client_module,
            cluster=cluster_name,
            job_ids=ids,
            local_user_id=local_uid,
        )

    return Response.success(
        msg="ok",
        data={
            "imported": imported,
            "failed": failed,
            "imported_count": len(imported),
            "failed_count": len(failed),
            "cluster": cluster_name,
            "hint": (
                "与 ``nanhu jobs detail --job-id`` 同源（getJobDetail）；"
                "jobList 同步不到的同事共享作业，粘贴 UUID 即可入库检索。"
            ),
        },
    )


@router.get("/api/nanhu/jobs/{job_id}/resolve-for-task")
async def api_resolve_job_for_task(
    request: Request,
    job_id: str,
    cluster: Optional[str] = Query(
        default=None,
        description="南湖集群；默认优先 x10000，其次配置 nanhu_cluster",
    ),
) -> Response:
    """根据南湖 job 详情解析创建 Rollout 任务所需字段。

    返回 data：task_name / rollout_root / tokenizer_dir / job_id / warnings 等。
    """
    job_id = (job_id or "").strip()
    if not job_id:
        return Response.error(code=50001, msg="job_id 不能为空")

    cluster_name = (cluster or "").strip() or "x10000"
    user = current_user(request)
    local_user_id = getattr(user, "primary_id", None) if user is not None else None
    client_module = _nanhu_client_module()
    if client_module is None or not hasattr(client_module, "get_job_detail"):
        raise RolloutInsightException(
            RolloutInsightException.NANHU_RESOURCE_QUERY_FAILED,
            "nanhu_client 不可用，无法查询作业详情",
        )

    try:
        detail = client_module.get_job_detail(
            cluster=cluster_name,
            job_id=job_id,
            local_user_id=local_user_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_job_detail 失败 job_id={} cluster={}", job_id, cluster_name)
        raise RolloutInsightException(
            RolloutInsightException.NANHU_RESOURCE_QUERY_FAILED,
            f"查询南湖作业失败: {exc}",
        ) from exc

    if not isinstance(detail, dict):
        raise RolloutInsightException(
            RolloutInsightException.NANHU_RESOURCE_QUERY_FAILED,
            "南湖作业详情返回格式异常",
        )
    if detail.get("ok") is False and "data" not in detail:
        msg = (
            detail.get("message")
            or detail.get("error")
            or detail.get("hint")
            or "查询南湖作业失败"
        )
        raise RolloutInsightException(
            RolloutInsightException.NANHU_RESOURCE_QUERY_FAILED,
            str(msg),
        )

    from utils.nanhu_job_path_resolve import (
        extract_job_payload,
        resolve_paths_from_job_detail,
    )

    payload = extract_job_payload(detail)
    if payload is None:
        err = detail.get("error") or detail.get("message") or "作业找不到"
        raise RolloutInsightException(
            RolloutInsightException.NANHU_JOB_NOT_FOUND,
            f"南湖作业未找到: {job_id} ({err})",
        )

    resolved = resolve_paths_from_job_detail(payload)
    resolved["cluster"] = cluster_name
    if not resolved.get("job_id"):
        resolved["job_id"] = job_id
    return Response.success(msg="ok", data=resolved)


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