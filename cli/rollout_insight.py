"""``rollout-insight`` CLI 入口（v0.5.1）。

命令树::

    rollout-insight [--api-base URL] [--format json|table] [--cluster wuxi|x10000]
        tasks {list,create,rescan,delete}
        segments {list,show,search}
        nanhu {jobs,restart,link}
        tokenizer {set,test}
        rollouts {show,parse,list}                                    # v0.4.0
            parse: v0.5.0 + --backfill-turns / --clear-turn-table / --tokenizer
        agent-transcripts {report}                                    # v0.4.0
        users {list,create,set-password,activate,deactivate}         # v0.5.1

写操作默认 dry-run；加 ``--submit`` 才执行。``rollouts parse`` 是 idempotent
且用户主动触发，所以即使不带 ``--submit`` 也会真执行。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

import cli.api_client as api
import cli.formatters as fmt
import cli.nanhu_bridge as nanhu_bridge
import cli.tokenizer_cmd as tokenizer_cmd
import cli.users_cmd as users_cmd


# ---------------------------------------------------------------------------
# 顶层 arg parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """顶层 argparse parser。"""
    parser = argparse.ArgumentParser(
        prog="rollout-insight",
        description=(
            "rollout-insight CLI（v0.4.0）：本地 Web API + 南湖调度 + tokenizer + rollout 解析/列表 + agent-transcripts。\n"
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

    # ----- rollouts (v0.4.0) -----
    _build_rollouts_parser(sub)

    # ----- agent-transcripts (v0.4.0) -----
    _build_agent_transcripts_parser(sub)

    # ----- users (v0.5.1) -----
    _build_users_parser(sub)

    return parser


def _build_users_parser(sub: argparse._SubParsersAction) -> None:
    """注册 ``users`` 顶层子命令（v0.5.1 T12）。

    子命令：
    - list                  列出所有用户（默认 JSON，--format table 转 markdown 表）
    - create                新建用户（dry-run 默认；--submit 真执行）
    - set-password          改密码（dry-run 默认；--submit 真执行）
    - activate / deactivate 启用 / 禁用账号（dry-run 默认；--submit 真执行）
    """
    users_p = sub.add_parser(
        "users",
        help="用户管理（v0.5.1）",
    )
    users_sub = users_p.add_subparsers(
        dest="users_cmd", required=True, title="users 子命令"
    )

    # users list
    u_list = users_sub.add_parser(
        "list", help="列出所有用户（JSON / table）"
    )

    # users create
    u_create = users_sub.add_parser(
        "create", help="新建用户（dry-run 默认；--submit 真执行）"
    )
    u_create.add_argument("--username", required=True, help="用户名")
    u_create.add_argument("--password", required=True, help="明文密码（内部 bcrypt hash）")
    u_create.add_argument(
        "--role",
        choices=("admin", "user"),
        default="user",
        help="角色（默认 user）",
    )
    u_create.add_argument(
        "--submit",
        action="store_true",
        help="真执行（默认 dry-run）",
    )

    # users set-password
    u_setpwd = users_sub.add_parser(
        "set-password", help="改密码（dry-run 默认；--submit 真执行）"
    )
    u_setpwd.add_argument("--username", required=True)
    u_setpwd.add_argument("--password", required=True, help="新密码")
    u_setpwd.add_argument("--submit", action="store_true")

    # users activate
    u_activate = users_sub.add_parser(
        "activate", help="启用账号（dry-run 默认；--submit 真执行）"
    )
    u_activate.add_argument("--username", required=True)
    u_activate.add_argument("--submit", action="store_true")

    # users deactivate
    u_deactivate = users_sub.add_parser(
        "deactivate", help="禁用账号（dry-run 默认；--submit 真执行）"
    )
    u_deactivate.add_argument("--username", required=True)
    u_deactivate.add_argument("--submit", action="store_true")


# ---------------------------------------------------------------------------
# v0.4.0 子命令 parser
# ---------------------------------------------------------------------------


def _build_rollouts_parser(sub: argparse._SubParsersAction) -> None:
    """注册 ``rollouts`` 顶层子命令（v0.4.0 T7）。"""
    rollouts_p = sub.add_parser(
        "rollouts",
        help="Rollout 详情 / 解析 / 列表（v0.4.0）",
    )
    rollouts_sub = rollouts_p.add_subparsers(
        dest="rollouts_cmd", required=True, title="rollouts 子命令"
    )

    # rollouts show <id>
    r_show = rollouts_sub.add_parser(
        "show",
        help="展示单个 rollout 概览（pt_path / n_samples / mean_reward / attempts 摘要）",
    )
    r_show.add_argument("rollout_id", type=int, help="rollout primary_id（不是业务键）")

    # rollouts parse <id> [--tokenizer DIR]
    r_parse = rollouts_sub.add_parser(
        "parse",
        help="触发 .pt 重解析（idempotent；可选 --tokenizer 覆盖）",
    )
    r_parse.add_argument("rollout_id", type=int, help="rollout primary_id")
    r_parse.add_argument(
        "--tokenizer",
        dest="tokenizer_dir",
        default="",
        help="可选 tokenizer 目录（写到 POST body 备用）",
    )
    r_parse.add_argument(
        "--backfill-turns",
        action="store_true",
        help="v0.5.0 T5：触发 segments/{id}/turns.json 的 token 重切，并写回 turn 表",
    )
    r_parse.add_argument(
        "--clear-turn-table",
        action="store_true",
        help="v0.5.0 T5：先 DELETE FROM ri_segment_turn_table 再 parse（默认会清）",
    )
    r_parse.add_argument(
        "--submit",
        action="store_true",
        help="实际触发 parse（v0.4.0 占位：parse 本身就是 idempotent 用户主动操作，不加也执行）",
    )

    # rollouts list --task <task_id>
    r_list = rollouts_sub.add_parser(
        "list",
        help="列 task 下的 rollouts（默认走 /api/tasks/{id}/rollouts，回退到 /api/tasks 详情）",
    )
    r_list.add_argument("--task", dest="task_id", type=int, required=True, help="task primary_id")
    r_list.add_argument("--limit", type=int, default=None)
    r_list.add_argument("--offset", type=int, default=0)


def _build_agent_transcripts_parser(sub: argparse._SubParsersAction) -> None:
    """注册 ``agent-transcripts`` 顶层子命令（v0.4.0 T7）。"""
    at_p = sub.add_parser(
        "agent-transcripts",
        help="Agent transcripts 报告（v0.4.0 T6 配套）",
    )
    at_sub = at_p.add_subparsers(
        dest="agent_transcripts_cmd", required=True, title="agent-transcripts 子命令"
    )

    # agent-transcripts report <file_or_dir>
    at_report = at_sub.add_parser(
        "report",
        help="单文件或目录报告（自动判别：file → /report；dir → /batch）",
    )
    at_report.add_argument(
        "path",
        help="单个 .jsonl 文件绝对路径，或包含 .jsonl 的目录",
    )


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
# v0.4.0 dispatch
# ---------------------------------------------------------------------------


def _run_rollouts(args: argparse.Namespace) -> int:
    """``rollouts`` 子命令分发。"""
    base = api.resolve_api_base(args.api_base)
    sub = args.rollouts_cmd

    if sub == "show":
        # 调 /api/rollouts/{id}/json（plan 契约），404 时回退到 /api/rollouts/{id}
        payload = api.get_rollout_json(base, args.rollout_id)
        _emit(payload, args)
        return 0

    if sub == "parse":
        # parse 是 idempotent + 用户主动触发：不加 --submit 时打印 plan
        if not args.submit:
            plan = {
                "action": "parse rollout",
                "api_base": base,
                "rollout_id": args.rollout_id,
                "tokenizer_dir": args.tokenizer_dir,
                "backfill_turns": bool(args.backfill_turns),
                "clear_turn_table": bool(args.clear_turn_table),
                "submit": False,
                "dry_run": True,
                "note": (
                    "dry-run：未调用 parse（v0.4.0 占位）。"
                    "parse 是 idempotent + 用户主动操作，不加 --submit 也会真执行；"
                    "此处仅打印 plan 供 agent 二次确认。"
                ),
            }
            _emit(plan, args)
            return 0
        # v0.5.0 T5：parse 后若 --backfill-turns，逐 segment 触发一次 turn 重新切分（token-based）。
        # 这步是为了让「先 parse 写完 (group, segment) 落库」与「token 切分写回 ri_segment_turn_table」
        # 解耦；默认行为 parse 已写 turn，无需再调。
        payload = api.parse_rollout(
            base,
            args.rollout_id,
            tokenizer_dir=args.tokenizer_dir,
            force=True,
        )
        _emit(payload, args)

        if args.backfill_turns:
            # 1) 清 turn 表（先本地可选做，由 --clear-turn-table 控制）
            if args.clear_turn_table:
                clear_resp = api.clear_turn_table(base, args.rollout_id)
                _emit(
                    {
                        "action": "backfill_turns: clear turn table",
                        "rollout_id": args.rollout_id,
                        "result": clear_resp,
                    },
                    args,
                )
            # 2) 列 segments（rollouts list → task segments search），逐个调 turns.json
            seg_ids = api.list_segment_ids_for_rollout(base, args.rollout_id)
            _emit(
                {
                    "action": "backfill_turns: 发现 segments",
                    "rollout_id": args.rollout_id,
                    "segment_count": len(seg_ids),
                    "sample_segment_ids": seg_ids[:5],
                },
                args,
            )
            # 3) 逐个拉 turns.json
            turn_counts: Dict[int, int] = {}
            for sid in seg_ids:
                try:
                    resp = api.get_segment_turns(base, sid)
                    turn_counts[sid] = int(
                        (resp.get("data") or {}).get("turn_count", 0) or 0
                    )
                except Exception as exc:  # noqa: BLE001
                    _emit_error(
                        f"segment {sid} 拉 turns 失败: {exc}",
                        code=-1,
                    )
                    turn_counts[sid] = -1
            _emit(
                {
                    "action": "backfill_turns: 完成",
                    "rollout_id": args.rollout_id,
                    "segment_total": len(seg_ids),
                    "turn_total": sum(max(v, 0) for v in turn_counts.values()),
                    "per_segment_turn_count": turn_counts,
                    "tokenizer_dir": args.tokenizer_dir,
                },
                args,
            )
        return 0

    if sub == "list":
        payload = api.list_rollouts_by_task(
            base,
            args.task_id,
            limit=args.limit,
            offset=args.offset or 0,
        )
        _emit(payload, args)
        return 0

    _emit_error(f"未知 rollouts 子命令: {sub}")
    return 2


def _run_agent_transcripts(args: argparse.Namespace) -> int:
    """``agent-transcripts`` 子命令分发。

    - ``report <file>``  → GET /api/agent-transcripts/report?path=<file>
    - ``report <dir>``   → GET /api/agent-transcripts/batch?dir=<dir>
    """
    base = api.resolve_api_base(args.api_base)
    sub = args.agent_transcripts_cmd
    path = args.path

    if sub == "report":
        if not path:
            _emit_error("agent-transcripts report: path 不能为空")
            return 2
        if not os.path.exists(path):
            _emit_error(f"路径不存在: {path}")
            return 2
        if os.path.isfile(path):
            payload = api.agent_transcript_report(base, path)
        else:
            # 目录：批量
            payload = api.agent_transcript_batch(base, path)
        _emit(payload, args)
        return 0

    _emit_error(f"未知 agent-transcripts 子命令: {sub}")
    return 2


def _run_users(args: argparse.Namespace) -> int:
    """``users`` 子命令分发（v0.5.1 T12）。"""
    sub = args.users_cmd

    if sub == "list":
        payload = users_cmd.list_users()
        _emit(payload, args)
        return 0

    if sub == "create":
        payload = users_cmd.create_user(
            username=args.username,
            password=args.password,
            role=args.role,
            submit=args.submit,
        )
        _emit(payload, args)
        return 0 if payload.get("code") == 0 else 1

    if sub == "set-password":
        payload = users_cmd.set_password(
            username=args.username,
            password=args.password,
            submit=args.submit,
        )
        _emit(payload, args)
        return 0 if payload.get("code") == 0 else 1

    if sub == "activate":
        payload = users_cmd.set_active(
            username=args.username,
            is_active=True,
            submit=args.submit,
        )
        _emit(payload, args)
        return 0 if payload.get("code") == 0 else 1

    if sub == "deactivate":
        payload = users_cmd.set_active(
            username=args.username,
            is_active=False,
            submit=args.submit,
        )
        _emit(payload, args)
        return 0 if payload.get("code") == 0 else 1

    _emit_error(f"未知 users 子命令: {sub}")
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
        if args.command == "rollouts":
            return _run_rollouts(args)
        if args.command == "agent-transcripts":
            return _run_agent_transcripts(args)
        if args.command == "users":
            return _run_users(args)
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