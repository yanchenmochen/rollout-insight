"""``rollout-insight`` CLI 入口（v0.3.0 阶段 3C）。

命令树::

    rollout-insight [--api-base URL] [--format json|table] [--cluster wuxi|x10000]
        tasks {list,create,rescan,delete}
        segments {list,show,search}
        nanhu {jobs,restart,link}
        tokenizer {set,test}

写操作默认 dry-run；加 ``--submit`` 才执行。
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

import cli.api_client as api
import cli.formatters as fmt
import cli.nanhu_bridge as nanhu_bridge
import cli.tokenizer_cmd as tokenizer_cmd


# ---------------------------------------------------------------------------
# 顶层 arg parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """顶层 argparse parser。"""
    parser = argparse.ArgumentParser(
        prog="rollout-insight",
        description=(
            "rollout-insight CLI（v0.3.0）：本地 Web API + 南湖调度 + tokenizer 操作。\n"
            "写操作默认 dry-run，加 --submit 才执行。"
        ),
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="本地 Web API base URL（默认 http://localhost:39998，亦可走 $ROLLOUT_INSIGHT_API_BASE）",
    )
    parser.add_argument(
        "--format",
        choices=("json", "table"),
        default="json",
        help="输出格式：json（默认）/ table（markdown 表格）",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="json 模式是否 indent（仅 --format json 有效）",
    )
    parser.add_argument(
        "--cluster",
        choices=("wuxi", "x10000"),
        default="wuxi",
        help="南湖集群（默认 wuxi）",
    )

    sub = parser.add_subparsers(dest="command", required=True, title="commands")

    # ----- tasks -----
    tasks_p = sub.add_parser("tasks", help="任务（task）的 list / create / rescan / delete")
    tasks_sub = tasks_p.add_subparsers(dest="tasks_cmd", required=True)

    # tasks list
    t_list = tasks_sub.add_parser("list", help="列任务")
    t_list.add_argument("--task-name", default=None, help="模糊匹配任务名")
    t_list.add_argument("--status", default=None, help="按状态过滤")
    t_list.add_argument("--page", type=int, default=None, help="页码")
    t_list.add_argument("--page-size", type=int, default=None, help="每页条数")
    t_list.add_argument("--limit", type=int, default=None, help="最大条数")
    t_list.add_argument("--offset", type=int, default=None, help="偏移")

    # tasks create
    t_create = tasks_sub.add_parser("create", help="新建任务（dry-run）")
    t_create.add_argument("--task-name", required=True)
    t_create.add_argument("--rollout-root", required=True)
    t_create.add_argument("--tokenizer-dir", default="")
    t_create.add_argument("--nanhu-job-id", default="")
    t_create.add_argument("--description", default="")
    t_create.add_argument(
        "--source-type",
        default="local_dump",
        help="local_dump | nanhu （默认 local_dump）",
    )
    t_create.add_argument("--recursive", action="store_true", help="递归扫 rollout 目录")
    t_create.add_argument("--submit", action="store_true", help="实际执行（默认 dry-run）")

    # tasks rescan
    t_rescan = tasks_sub.add_parser("rescan", help="重扫任务（dry-run）")
    t_rescan.add_argument("--task-id", type=int, required=True)
    t_rescan.add_argument("--submit", action="store_true")

    # tasks delete
    t_delete = tasks_sub.add_parser("delete", help="删除任务（dry-run）")
    t_delete.add_argument("--task-id", type=int, required=True)
    t_delete.add_argument("--submit", action="store_true")

    # ----- segments -----
    seg_p = sub.add_parser("segments", help="Segment 的 list / show / search")
    seg_sub = seg_p.add_subparsers(dest="segments_cmd", required=True)

    # segments list (alias of search)
    s_list = seg_sub.add_parser("list", help="列 segment（按 task 过滤）")
    s_list.add_argument("--task-id", type=int, required=True)
    s_list.add_argument("--min-reward", type=float, default=None)
    s_list.add_argument("--keyword", default=None)
    s_list.add_argument("--page", type=int, default=1)
    s_list.add_argument("--page-size", type=int, default=50)

    # segments show
    s_show = seg_sub.add_parser("show", help="展示 segment transcript（JSON）")
    s_show.add_argument("--segment-id", type=int, required=True)

    # segments search
    s_search = seg_sub.add_parser("search", help="检索 segment（task_id 必填）")
    s_search.add_argument("--task-id", type=int, required=True)
    s_search.add_argument("--min-reward", type=float, default=None)
    s_search.add_argument("--keyword", default=None)
    s_search.add_argument("--page", type=int, default=1)
    s_search.add_argument("--page-size", type=int, default=50)

    # ----- nanhu -----
    nan_p = sub.add_parser("nanhu", help="南湖（nanhu）作业相关")
    nan_sub = nan_p.add_subparsers(dest="nanhu_cmd", required=True)

    n_jobs = nan_sub.add_parser("jobs", help="列作业")
    n_jobs.add_argument("--job-type", default=None, help="AI_TRAIN | AI_ONLINE_INFERENCE_SERVICE")
    n_jobs.add_argument("--status", default=None, help="running / finished / failed ...")
    n_jobs.add_argument("--user-id", default=None)

    n_restart = nan_sub.add_parser("restart", help="重启作业（dry-run）")
    n_restart.add_argument("--job-id", required=True)
    n_restart.add_argument("--submit", action="store_true")

    n_link = nan_sub.add_parser("link", help="绑定南湖 job 到本地 task（dry-run）")
    n_link.add_argument("--job-id", required=True)
    n_link.add_argument("--task-name", required=True)
    n_link.add_argument("--submit", action="store_true")

    # ----- tokenizer -----
    tok_p = sub.add_parser("tokenizer", help="tokenizer 路径配置")
    tok_sub = tok_p.add_subparsers(dest="tokenizer_cmd", required=True)

    tok_set = tok_sub.add_parser("set", help="设置 config.json 的 tokenizer_dir（dry-run）")
    tok_set.add_argument("--path", required=True, help="新 tokenizer 目录")
    tok_set.add_argument("--config", default=None, help="config.json 路径（默认 $ROLLOUT_INSIGHT_CONFIG_PATH 或 ./config.json）")
    tok_set.add_argument("--submit", action="store_true")

    tok_test = tok_sub.add_parser("test", help="验证 tokenizer 目录是否合法")
    tok_test.add_argument("--path", required=True)

    return parser


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def _emit(payload: Any, args: argparse.Namespace) -> None:
    fmt.emit(payload, fmt=args.format, pretty=args.pretty)


def _emit_error(msg: str, code: int = -1) -> None:
    fmt.emit_error(msg, code=code, fmt="json")
    sys.stderr.write(f"错误: {msg}\n")
    sys.stderr.flush()


def _run_tasks(args: argparse.Namespace) -> int:
    base = api.resolve_api_base(args.api_base)
    sub = args.tasks_cmd

    if sub == "list":
        payload = api.list_tasks(
            base,
            task_name=args.task_name,
            status=args.status,
            page=args.page,
            page_size=args.page_size,
            limit=args.limit,
            offset=args.offset,
        )
        _emit(payload, args)
        return 0

    if sub == "create":
        plan = {
            "action": "create task",
            "api_base": base,
            "task_name": args.task_name,
            "rollout_root": args.rollout_root,
            "tokenizer_dir": args.tokenizer_dir,
            "nanhu_job_id": args.nanhu_job_id,
            "description": args.description,
            "source_type": args.source_type,
            "recursive": args.recursive,
            "submit": args.submit,
        }
        if not args.submit:
            plan["dry_run"] = True
            plan["note"] = "dry-run：未提交，加 --submit 才执行"
            _emit(plan, args)
            return 0
        payload = api.create_task(
            base,
            task_name=args.task_name,
            rollout_root=args.rollout_root,
            description=args.description,
            tokenizer_dir=args.tokenizer_dir,
            nanhu_job_id=args.nanhu_job_id,
            source_type=args.source_type,
            recursive=args.recursive,
        )
        _emit(payload, args)
        return 0

    if sub == "rescan":
        plan = {
            "action": "rescan task",
            "api_base": base,
            "task_id": args.task_id,
            "submit": args.submit,
        }
        if not args.submit:
            plan["dry_run"] = True
            plan["note"] = "dry-run：未提交"
            _emit(plan, args)
            return 0
        payload = api.rescan_task(base, args.task_id)
        _emit(payload, args)
        return 0

    if sub == "delete":
        plan = {
            "action": "delete task",
            "api_base": base,
            "task_id": args.task_id,
            "submit": args.submit,
        }
        if not args.submit:
            plan["dry_run"] = True
            plan["note"] = "dry-run：未提交"
            _emit(plan, args)
            return 0
        payload = api.delete_task(base, args.task_id)
        _emit(payload, args)
        return 0

    _emit_error(f"未知 tasks 子命令: {sub}")
    return 2


def _run_segments(args: argparse.Namespace) -> int:
    base = api.resolve_api_base(args.api_base)
    sub = args.segments_cmd

    if sub == "list":
        payload = api.list_segments(
            base,
            task_id=args.task_id,
            min_reward=args.min_reward,
            keyword=args.keyword,
            page=args.page,
            page_size=args.page_size,
        )
        _emit(payload, args)
        return 0

    if sub == "show":
        payload = api.show_segment(base, args.segment_id)
        _emit(payload, args)
        return 0

    if sub == "search":
        payload = api.search_segments(
            base,
            task_id=args.task_id,
            min_reward=args.min_reward,
            keyword=args.keyword,
            page=args.page,
            page_size=args.page_size,
        )
        _emit(payload, args)
        return 0

    _emit_error(f"未知 segments 子命令: {sub}")
    return 2


def _run_nanhu(args: argparse.Namespace) -> int:
    """nanhu 子命令：透传 nanhu-cli 输出（stdout 是 JSON 或 table），不做二次封装。

    - dry-run 路径：打印 plan JSON
    - 实跑路径：把 nanhu-cli 的 stdout 原样打（CLI 不再解析）
    - 错误：把 nanhu-cli 的 stderr 提升到 stderr + 非零退出码
    """
    sub = args.nanhu_cmd
    cluster = args.cluster

    if sub == "jobs":
        try:
            stdout, rc = nanhu_bridge.list_jobs(
                cluster=cluster,
                job_type=args.job_type,
                status=args.status,
                user_id=args.user_id,
                format_=args.format,
                pretty=args.pretty,
            )
        except nanhu_bridge.NanhuBridgeError as exc:
            _emit_error(str(exc), code=exc.code)
            return 1
        if rc != 0:
            _emit_error(f"nanhu-cli jobs 退出码 {rc}", code=rc)
            return 1
        sys.stdout.write(stdout)
        if stdout and not stdout.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    if sub == "restart":
        if not args.submit:
            plan = {
                "action": "restart nanhu job",
                "cluster": cluster,
                "job_id": args.job_id,
                "submit": False,
                "dry_run": True,
                "note": "dry-run：未调用 nanhu-cli，加 --submit 才执行",
            }
            _emit(plan, args)
            return 0
        try:
            stdout, rc = nanhu_bridge.restart_job(
                cluster=cluster, job_id=args.job_id, submit=True
            )
        except nanhu_bridge.NanhuBridgeError as exc:
            _emit_error(str(exc), code=exc.code)
            return 1
        if rc != 0:
            _emit_error(f"nanhu-cli restart 退出码 {rc}", code=rc)
            return 1
        sys.stdout.write(stdout)
        if stdout and not stdout.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    if sub == "link":
        base = api.resolve_api_base(args.api_base)
        payload = nanhu_bridge.link_job_to_task(
            api_base=base,
            task_name=args.task_name,
            job_id=args.job_id,
            submit=args.submit,
        )
        _emit(payload, args)
        return 0

    _emit_error(f"未知 nanhu 子命令: {sub}")
    return 2


def _run_tokenizer(args: argparse.Namespace) -> int:
    sub = args.tokenizer_cmd

    if sub == "set":
        config_path = tokenizer_cmd._resolve_config_path(args.config)
        result = tokenizer_cmd.set_tokenizer_dir(
            config_path=config_path,
            new_dir=args.path,
            submit=args.submit,
        )
        _emit(result, args)
        return 0

    if sub == "test":
        result = tokenizer_cmd.test_tokenizer_path(args.path)
        _emit(result, args)
        return 0

    _emit_error(f"未知 tokenizer 子命令: {sub}")
    return 2


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """CLI 入口；返回 int 退出码。

    用法：
        rollout-insight tasks list
        rollout-insight --format table segments search --task_id
        rollout-insight --cluster x10000 nanhu jobs --status running
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "tasks":
            return _run_tasks(args)
        if args.command == "segments":
            return _run_segments(args)
        if args.command == "nanhu":
            return _run_nanhu(args)
        if args.command == "tokenizer":
            return _run_tokenizer(args)
        _emit_error(f"未知命令: {args.command}")
        return 2
    except api.ApiError as exc:
        _emit_error(str(exc), code=exc.code)
        return 1
    except nanhu_bridge.NanhuBridgeError as exc:
        _emit_error(str(exc), code=exc.code)
        return 1
    except FileNotFoundError as exc:
        _emit_error(str(exc), code=-1)
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\n[interrupted]\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())