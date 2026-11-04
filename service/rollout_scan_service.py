"""RolloutScanService：扫描 .pt dump 文件并入库。

主要职责：
- 解析 task 对应的 .pt 文件路径（OUTPUT_DIR / ROLLOUT_DUMP_DIR / <dump_root>/<task_name>/rollout_dumps/）
- 调用 utils.analyze_dump 读取每个 .pt 的元数据与样本摘要
- 通过 DBRolloutService / DBAttemptService / DBSegmentService 入库
- 更新 DBTaskService 状态：SCANNING -> READY | FAILED
- 写 log/scan_<task_id>.log 记录扫描过程

设计约定：
- 不写 DBSegmentTranscriptService（懒加载，留给 stage 2D）
- 每个 sample 的 idx 直接当作 attempt_id（v0.2 简化模型）
- 对 ORM/Service 的依赖通过接口契约接入（避免循环依赖，方便测试 mock）

依赖（由 2A worker 提供）：
- db.db_task_service.DBTaskService
- db.db_rollout_service.DBRolloutService
- db.db_attempt_service.DBAttemptService
- db.db_segment_service.DBSegmentService
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from data.rollout_insight_exception import RolloutInsightException
from utils.analyze_dump import load_dump_meta, summarize_dump
from utils.config_loader import config

# ---- 任务状态常量（与 2A worker 的 TaskStatus 约定对齐）----
TASK_STATUS_PENDING = "PENDING"
TASK_STATUS_SCANNING = "SCANNING"
TASK_STATUS_READY = "READY"
TASK_STATUS_FAILED = "FAILED"


class RolloutScanService:
    """扫描 .pt dump -> 入库主服务。

    单次扫描调用：``scan_task(task_id)``。
    强制重扫：``rescan_task(task_id)``。
    进度查询：``get_progress(task_id)``。
    """

    def __init__(self) -> None:
        self.log_dir: Path = Path(config.log_dir)
        self.dump_root: str = config.rollout_dump_root or ""
        # task_id -> sink_id：跟踪每个 task 已注册的 scan 日志 sink，避免重复
        self._active_scan_sinks: Dict[int, int] = {}

    # ---------- DB Service 延迟导入 ----------

    @staticmethod
    def _db_task_service():
        from db.db_task_service import DBTaskService

        return DBTaskService

    @staticmethod
    def _db_rollout_service():
        from db.db_rollout_service import DBRolloutService

        return DBRolloutService

    @staticmethod
    def _db_attempt_service():
        from db.db_attempt_service import DBAttemptService

        return DBAttemptService

    @staticmethod
    def _db_segment_service():
        from db.db_segment_service import DBSegmentService

        return DBSegmentService

    @staticmethod
    def _session_factory():
        try:
            from db.db_element import Session

            return Session
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _task_model():
        try:
            from db import Task as DBTask

            return DBTask
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _rollout_model():
        try:
            from db import Rollout as DBRollout

            return DBRollout
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _attempt_model():
        try:
            from db import Attempt as DBAttempt

            return DBAttempt
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _segment_model():
        try:
            from db import Segment as DBSegment

            return DBSegment
        except Exception:  # noqa: BLE001
            return None

    # ---------- 主入口 ----------

    def scan_task(self, task_id: int, force: bool = False) -> Dict[str, Any]:
        """扫描指定 task 下的所有 .pt 文件并入库。

        Args:
            task_id: DBTask.primary_id
            force: True 时先清空该 task 下旧数据，再扫

        Returns:
            形如 ``{
                "task_id": int,
                "status": "READY" | "FAILED",
                "scanned": int,  # 扫描成功的 .pt 数量
                "total": int,    # 待扫描 .pt 数量
                "rollouts": int, # 入库 rollout 行数
                "attempts": int, # 入库 attempt 行数
                "segments": int, # 入库 segment 行数
                "errors": List[str],
            }``
        """
        log_path = self._scan_log_path(task_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 单独为本次扫描接一个文件 sink，便于事后查看。
        # 若同一 task 重复扫描，复用已有 sink（按 task_id 去重）。
        if task_id not in self._active_scan_sinks:
            sink_id = logger.add(
                str(log_path),
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
                level="INFO",
                encoding="utf-8",
                enqueue=True,
                filter=lambda record, _tid=task_id: True,
            )
            self._active_scan_sinks[task_id] = sink_id

        DBTaskService = self._db_task_service()

        logger.info("[scan_task] start task_id={} force={}", task_id, force)

        result: Dict[str, Any] = {
            "task_id": task_id,
            "status": TASK_STATUS_READY,
            "scanned": 0,
            "total": 0,
            "rollouts": 0,
            "attempts": 0,
            "segments": 0,
            "errors": [],
        }

        # 0. 查 task
        task = DBTaskService.get_by_id(task_id)
        if not task:
            msg = f"task_id={task_id} 不存在"
            logger.error(msg)
            raise RolloutInsightException(
                RolloutInsightException.TASK_NOT_FOUND, msg
            )

        task_name = getattr(task, "task_name", "") or ""
        if not task_name:
            msg = f"task_id={task_id} 缺少 task_name"
            logger.error(msg)
            raise RolloutInsightException(
                RolloutInsightException.TASK_VALIDATION, msg
            )

        # 1. status -> SCANNING
        try:
            DBTaskService.update_status(task_id, TASK_STATUS_SCANNING)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "更新状态 -> SCANNING 失败 task_id={} err={}", task_id, exc
            )

        # 2. force 清空
        if force:
            self._purge_task_data(task_id)

        # 3. 解析 .pt 路径
        try:
            task_rollout_root = getattr(task, "rollout_root", "") or ""
            task_recursive = bool(getattr(task, "recursive", False))
            pt_paths = self._resolve_dump_paths(
                task_name,
                task_rollout_root,
                recursive=task_recursive,
            )
            result["total"] = len(pt_paths)
            logger.info(
                "[scan_task] task_id={} task_name={} resolved dump paths count={}",
                task_id,
                task_name,
                len(pt_paths),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[scan_task] _resolve_dump_paths 失败 task_id={}", task_id)
            result["status"] = TASK_STATUS_FAILED
            result["errors"].append(f"resolve dump paths: {exc}")
            try:
                DBTaskService.update_status(task_id, TASK_STATUS_FAILED)
            except Exception:  # noqa: BLE001
                logger.exception("更新状态 -> FAILED 失败 task_id={}", task_id)
            return result

        if not pt_paths:
            logger.warning(
                "[scan_task] task_id={} task_name={} 没有找到任何 .pt，请检查路径",
                task_id,
                task_name,
            )
            # 没有 .pt 也算成功（空扫描）
            try:
                DBTaskService.update_status(task_id, TASK_STATUS_READY)
            except Exception:  # noqa: BLE001
                logger.exception("更新状态 -> READY 失败 task_id={}", task_id)
            return result

        # 4. 逐个 .pt 扫描入库
        for pt_path in pt_paths:
            try:
                _rollout, n_att, n_seg = self._scan_single_pt(task_id, pt_path)
                result["scanned"] += 1
                result["rollouts"] += 1
                result["attempts"] += n_att
                result["segments"] += n_seg
            except RolloutInsightException as exc:
                err = f"扫描 {pt_path} 失败: [{exc.code}]{exc.name} {exc.message}"
                logger.error(err)
                result["errors"].append(err)
            except Exception as exc:  # noqa: BLE001
                err = f"扫描 {pt_path} 失败: {exc}"
                logger.exception(err)
                result["errors"].append(err)

        # 5. status -> READY / FAILED
        if result["errors"]:
            if result["scanned"] == 0:
                result["status"] = TASK_STATUS_FAILED
            else:
                result["status"] = TASK_STATUS_READY
        else:
            result["status"] = TASK_STATUS_READY

        try:
            DBTaskService.update_status(task_id, result["status"])
        except Exception:  # noqa: BLE001
            logger.exception(
                "更新状态 -> {} 失败 task_id={}", result["status"], task_id
            )

        # 5.1 回写任务级统计（rollout_count/attempt_count/segment_count/mean_reward/last_scanned_at）
        try:
            self._update_task_stats(task_id)
        except Exception:  # noqa: BLE001
            logger.exception("更新任务统计失败 task_id={}", task_id)

        logger.info(
            "[scan_task] done task_id={} status={} scanned={}/{} rollouts={} attempts={} segments={} errors={}",
            task_id,
            result["status"],
            result["scanned"],
            result["total"],
            result["rollouts"],
            result["attempts"],
            result["segments"],
            len(result["errors"]),
        )
        return result

    def rescan_task(self, task_id: int) -> Dict[str, Any]:
        """强制重扫：``force=True`` 的封装。"""
        return self.scan_task(task_id, force=True)

    def get_progress(self, task_id: int) -> Dict[str, Any]:
        """返回扫描进度快照。"""
        DBTaskService = self._db_task_service()
        DBRolloutService = self._db_rollout_service()
        DBAttemptService = self._db_attempt_service()
        DBSegmentService = self._db_segment_service()

        task = DBTaskService.get_by_id(task_id)
        if not task:
            raise RolloutInsightException(
                RolloutInsightException.TASK_NOT_FOUND,
                f"task_id={task_id} 不存在",
            )
        # 通过 list_by_task 做计数（DBService 未提供 count_by_task）
        rollouts = DBRolloutService.list_by_task(task_id, limit=10000)
        attempts = DBAttemptService.list_by_task(task_id, limit=10000)
        segments = DBSegmentService.list_by_task(task_id, limit=20000)
        last_rollout_id = rollouts[-1].rollout_id if rollouts else None
        return {
            "task_id": task_id,
            "status": getattr(task, "status", TASK_STATUS_PENDING),
            "rollouts": len(rollouts),
            "attempts": len(attempts),
            "segments": len(segments),
            "last_rollout_id": last_rollout_id,
            "scanned": len(rollouts),
            "total": len(rollouts),
        }

    # ---------- 工具方法 ----------

    def _scan_log_path(self, task_id: int) -> Path:
        return self.log_dir / f"scan_{task_id}.log"

    def _resolve_dump_paths(
        self,
        task_name: str,
        task_rollout_root: str = "",
        recursive: bool = False,
    ) -> List[Path]:
        """按优先级解析 .pt 列表：
        0) task.rollout_root（页面/南湖创建时填的具体路径，最优先）
        1) OUTPUT_DIR 环境变量（若含 rollout_dumps 子目录则用之，否则用其本身）
        2) ROLLOUT_DUMP_DIR 环境变量
        3) <rollout_dump_root>/<task_name>/rollout_dumps/（兜底）

        ``recursive=True`` 时，在 rollout_root 下用 ``rglob("rollout_*.pt")``
        匹配嵌套的 ``generated/rollout_dumps`` 等目录。

        Returns:
            按路径排序的 .pt 列表（来自第一个能找到 .pt 的来源）
        """
        # 0) task.rollout_root（用户在页面填的具体路径；若是目录则 glob rollout_*.pt）
        if task_rollout_root:
            root = Path(task_rollout_root)
            if root.is_dir():
                hits = sorted(root.glob("rollout_*.pt"))
                if hits:
                    logger.info(
                        "_resolve_dump_paths: 使用 task.rollout_root={} 命中 {} 个 .pt",
                        root,
                        len(hits),
                    )
                    return hits
                # 是目录但没有 rollout_*.pt → 试一下子目录 rollout_dumps/
                sub = root / "rollout_dumps"
                if sub.is_dir():
                    hits = sorted(sub.glob("rollout_*.pt"))
                    if hits:
                        logger.info(
                            "_resolve_dump_paths: 使用 task.rollout_root/rollout_dumps={} 命中 {} 个 .pt",
                            sub,
                            len(hits),
                        )
                        return hits
                if recursive:
                    hits = sorted(root.rglob("rollout_*.pt"))
                    if hits:
                        logger.info(
                            "_resolve_dump_paths: recursive rglob under {} 命中 {} 个 .pt",
                            root,
                            len(hits),
                        )
                        return hits
            elif root.is_file() and root.suffix == ".pt":
                # 用户填的就是单个 .pt 路径
                logger.info(
                    "_resolve_dump_paths: task.rollout_root 是单个 .pt path={}",
                    root,
                )
                return [root]
            else:
                logger.warning(
                    "_resolve_dump_paths: task.rollout_root 不存在或不是目录 path={}",
                    root,
                )
        # 1) OUTPUT_DIR：若 .pt 目录在子目录则用之，否则用 OUTPUT_DIR 本身
        out_dir = os.environ.get("OUTPUT_DIR")
        if out_dir:
            out_p = Path(out_dir)
            if out_p.is_dir():
                sub = out_p / "rollout_dumps"
                roots: List[Path] = [sub, out_p] if sub.is_dir() else [out_p]
                hits: List[Path] = []
                for r in roots:
                    hits.extend(sorted(r.glob("rollout_*.pt")))
                if hits:
                    logger.info(
                        "_resolve_dump_paths: 使用 OUTPUT_DIR={} 命中 {} 个 .pt",
                        out_p,
                        len(hits),
                    )
                    return hits
            else:
                logger.warning(
                    "_resolve_dump_paths: OUTPUT_DIR 不存在 path={}", out_p
                )

        # 2) ROLLOUT_DUMP_DIR：直接当根目录 glob
        rd_dir = os.environ.get("ROLLOUT_DUMP_DIR")
        if rd_dir:
            rd_p = Path(rd_dir)
            if rd_p.is_dir():
                hits = sorted(rd_p.glob("rollout_*.pt"))
                if hits:
                    logger.info(
                        "_resolve_dump_paths: 使用 ROLLOUT_DUMP_DIR={} 命中 {} 个 .pt",
                        rd_p,
                        len(hits),
                    )
                    return hits
            else:
                logger.warning(
                    "_resolve_dump_paths: ROLLOUT_DUMP_DIR 不存在 path={}", rd_p
                )

        # 3) <rollout_dump_root>/<task_name>/rollout_dumps/：兜底
        if self.dump_root and task_name:
            base = Path(self.dump_root) / task_name
            if base.is_dir():
                dumps = base / "rollout_dumps"
                roots: List[Path] = (
                    [dumps, base] if dumps.is_dir() else [base]
                )
                hits: List[Path] = []
                for r in roots:
                    hits.extend(sorted(r.glob("rollout_*.pt")))
                if hits:
                    logger.info(
                        "_resolve_dump_paths: 使用 dump_root={} task_name={} 命中 {} 个 .pt",
                        self.dump_root,
                        task_name,
                        len(hits),
                    )
                    return hits
            else:
                logger.warning(
                    "_resolve_dump_paths: dump_root/task_name 不存在 base={}", base
                )

        return []

    def _scan_single_pt(
        self, task_id: int, pt_path: Path
    ) -> Tuple[Any, int, int]:
        """扫描单个 .pt 文件：upsert rollout + attempts + segments。

        Returns:
            ``(DBRollout, n_attempts, n_segments)``
        """
        logger.info("[_scan_single_pt] task_id={} pt={}", task_id, pt_path)

        DBRolloutService = self._db_rollout_service()
        DBAttemptService = self._db_attempt_service()
        DBSegmentService = self._db_segment_service()

        # 1. 元数据
        try:
            meta = load_dump_meta(pt_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("load_dump_meta 失败 pt={}", pt_path)
            raise RolloutInsightException(
                RolloutInsightException.PARSE_DUMP_CORRUPTED,
                f"load_dump_meta 失败 {pt_path}: {exc}",
            ) from exc

        # 2. summary rows
        try:
            rows = summarize_dump(pt_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("summarize_dump 失败 pt={}", pt_path)
            raise RolloutInsightException(
                RolloutInsightException.PARSE_DUMP_CORRUPTED,
                f"summarize_dump 失败 {pt_path}: {exc}",
            ) from exc

        # 3. 计算 fingerprint 所需的 pt_size_bytes / pt_mtime
        try:
            st = pt_path.stat()
            pt_size_bytes = int(st.st_size)
            pt_mtime = float(st.st_mtime)
        except OSError as exc:
            logger.warning(
                "_scan_single_pt: stat() 失败 pt={} err={}，使用 0 兜底",
                pt_path,
                exc,
            )
            pt_size_bytes = 0
            pt_mtime = 0.0

        # 4. upsert rollout（fingerprint 由 service 内部计算）
        rollout = DBRolloutService.upsert_with_fingerprint(
            task_id=task_id,
            rollout_id=meta.get("rollout_id", 0),
            pt_path=str(pt_path),
            pt_size_bytes=pt_size_bytes,
            pt_mtime=pt_mtime,
            mean_reward=float(meta.get("mean_reward") or 0.0),
            n_samples=int(meta.get("n_samples") or 0),
            n_attempts=int(meta.get("n_attempts") or 0),
            instance_count=int(meta.get("n_attempts") or 0),
        )

        # 5. 每个 row: upsert attempt + segment
        n_att = 0
        n_seg = 0
        rollout_id_value = meta.get("rollout_id")
        for row in rows:
            # 5.1 attempt 业务键 = (task_id, dump.rollout_id, group_index, sample.rollout_id)
            # 与 view_trajectory._attempt_layout 一致：同一 OpenHands 会话的多 segment
            # 共用 sample.rollout_id，不能再用 sample 下标当 attempt_id（否则 10 段会变成 10 个 attempt）。
            sample_attempt_id = row.get("rollout_id")
            if sample_attempt_id is None:
                sample_attempt_id = row.get("idx")
            row_idx = int(row.get("idx") or 0)
            group_index = row.get("group_index")
            if group_index is None:
                group_index = 0
            attempt = DBAttemptService.upsert(
                task_id=task_id,
                rollout_id=rollout_id_value,
                group_index=int(group_index),
                attempt_id=int(sample_attempt_id),
                sandbox_id=row.get("sandbox_id") or "",
                instance_id=row.get("instance_id") or "",
                status=row.get("status") or "",
                reward=row.get("reward"),
                agent_turns=int(row.get("agent_turns") or 0),
                tool_calls=int(row.get("tool_calls") or 0),
                diff_line_count=int(row.get("diff_line_count") or 0),
                response_length=int(row.get("response_length") or 0),
            )
            if attempt is not None:
                n_att += 1

            # 5.2 segment 一个 sample 一行，attempt_id 关联 attempt。
            # metadata.segment_index 可能显式为 None（.get 默认值不生效），回退到 sample 下标。
            attempt_pk = getattr(attempt, "primary_id", None)
            seg_index = row.get("segment_index")
            if seg_index is None:
                seg_index = row_idx
            seg = DBSegmentService.upsert(
                attempt_id=attempt_pk,
                segment_index=int(seg_index),
                segment_count=int(row.get("segment_count") or 1),
                status=row.get("status") or "",
                reward=row.get("reward"),
                response_length=int(row.get("response_length") or 0),
                mask_sum=int(row.get("mask_sum") or 0),
                sandbox_id=row.get("sandbox_id") or "",
            )
            if seg is not None:
                n_seg += 1

        logger.info(
            "[_scan_single_pt] done task_id={} pt={} attempts={} segments={}",
            task_id,
            pt_path,
            n_att,
            n_seg,
        )
        return rollout, n_att, n_seg

    def _purge_task_data(self, task_id: int) -> None:
        """清空该 task 下的所有 rollout/attempt/segment（保留 task 本身）。

        删除顺序：segment(逐 attempt) -> attempt -> rollout

        注：DBSegmentService.delete_by_task 在某些 SQLAlchemy 版本下会因
        join + delete() 不兼容而抛 InvalidRequestError；此处按 attempt 逐个删除，
        绕过 JOIN 路径。
        """
        DBRolloutService = self._db_rollout_service()
        DBAttemptService = self._db_attempt_service()
        DBSegmentService = self._db_segment_service()

        try:
            # 1. 收集 task 下所有 attempt，按 attempt_id 逐个删除 segment
            attempts = DBAttemptService.list_by_task(
                task_id, limit=100000
            )
            n_seg = 0
            for att in attempts:
                att_id = getattr(att, "primary_id", None)
                if att_id is None:
                    continue
                n_seg += DBSegmentService.delete_by_attempt(att_id)

            # 2. 删除 attempt
            n_att = DBAttemptService.delete_by_task(task_id)

            # 3. 删除 rollout
            n_ro = DBRolloutService.delete_by_task(task_id)
            logger.info(
                "[_purge_task_data] task_id={} purged segments={} attempts={} rollouts={}",
                task_id,
                n_seg,
                n_att,
                n_ro,
            )
        except Exception:  # noqa: BLE001
            logger.exception("[_purge_task_data] failed task_id={}", task_id)
            raise

    def _update_task_stats(self, task_id: int) -> None:
        """回写任务的聚合统计到 ri_task_table。

        字段：rollout_count, attempt_count, segment_count, mean_reward, last_scanned_at。
        调用时机：scan_task 完成后。
        """
        Session = self._session_factory()
        if Session is None:
            logger.warning("_update_task_stats: Session 工厂不可用")
            return
        DBRollout = self._rollout_model()
        DBAttempt = self._attempt_model()
        DBSegment = self._segment_model()
        if DBRollout is None or DBAttempt is None or DBSegment is None:
            logger.warning("_update_task_stats: ORM 不可用")
            return
        with Session() as session:
            n_rollouts = session.query(DBRollout).filter(DBRollout.task_id == task_id).count()
            n_attempts = (
                session.query(DBAttempt)
                .filter(DBAttempt.task_id == task_id)
                .count()
            )
            n_segments = (
                session.query(DBSegment)
                .join(DBAttempt, DBSegment.attempt_id == DBAttempt.primary_id)
                .filter(DBAttempt.task_id == task_id)
                .count()
            )
            mean_reward = (
                session.query(DBRollout.mean_reward)
                .filter(DBRollout.task_id == task_id)
                .all()
            )
            mean_reward_val = (
                sum(r[0] for r in mean_reward if r[0] is not None) / len(mean_reward)
                if mean_reward
                else 0.0
            )
            # 写回 DBTask
            DBTask = self._task_model()
            if DBTask is None:
                return
            task_obj = session.query(DBTask).filter(DBTask.primary_id == task_id).first()
            if task_obj is None:
                return
            task_obj.rollout_count = n_rollouts
            task_obj.attempt_count = n_attempts
            task_obj.segment_count = n_segments
            # mean_reward 列是 Integer，平均值转 int（页面/JSON 输出时再 float）
            task_obj.mean_reward = int(round(mean_reward_val * 1000))  # ×1000 存为定点
            task_obj.last_scanned_at = datetime.now()
            session.commit()
            logger.info(
                "[_update_task_stats] task_id={} rollouts={} attempts={} segments={} mean_reward={:.3f}",
                task_id,
                n_rollouts,
                n_attempts,
                n_segments,
                mean_reward_val,
            )