"""rollout-insight v0.2.0 阶段 —— 5 张表 ORM + Service 测试。

测试范围：
- 建表（Base.metadata.create_all）
- 完整链路：1 task → 1 rollout → 2 attempts → 3 segments
- 唯一约束：重复 (task_id, rollout_id) 必须抛 IntegrityError
- 唯一约束：重复 (attempt_id, segment_index) 必须抛 IntegrityError
- 唯一约束：重复 segment_id 在 transcript 表必须抛 IntegrityError
- upsert_with_fingerprint：相同指纹命中已存在行，更新摘要字段
- upsert：相同唯一键命中，更新可写字段

约定：使用临时 SQLite 文件，不污染 data/rollout_insight.db；
每个 test 用独立的 task_name 避免名称冲突。
"""
import os
import tempfile
import unittest
from typing import Any, Dict

from sqlalchemy.exc import IntegrityError


# 在 import 任何 db_*.py 之前，先把 db_path 切到临时文件，避免污染默认 db
_TMP_DIR = tempfile.mkdtemp(prefix="rollout-insight-test-db-")
_TMP_DB_PATH = os.path.join(_TMP_DIR, "test_rollout_insight.db")
os.environ.setdefault("ROLLOUT_INSIGHT_CONFIG_PATH", "")  # 不改外部 config
TMP_DB_URL = f"sqlite:///{_TMP_DB_PATH}"

# 直接 override config.db_path / config.database_url，再让 db_element 走我们的临时路径
from utils.config_loader import config  # noqa: E402

config._data["db_path"] = _TMP_DB_PATH  # type: ignore[attr-defined]
config._data["database_url"] = TMP_DB_URL  # type: ignore[attr-defined]


# 此时再 import db/db_element 等，会拿到新的 db_path
from db.db_element import Base, Session, engine  # noqa: E402
from db.db_attempt import DBAttempt  # noqa: E402
from db.db_attempt_service import DBAttemptService  # noqa: E402
from db.db_rollout import DBRollout  # noqa: E402
from db.db_rollout_service import DBRolloutService  # noqa: E402
from db.db_segment import DBSegment  # noqa: E402
from db.db_segment_service import DBSegmentService  # noqa: E402
from db.db_segment_transcript import DBSegmentTranscript  # noqa: E402
from db.db_segment_transcript_service import DBSegmentTranscriptService  # noqa: E402
from db.db_task import DBTask  # noqa: E402
from db.db_task_service import DBTaskService  # noqa: E402
from data.rollout_insight_exception import RolloutInsightException  # noqa: E402


_COUNTER = {"n": 0}


def _unique_suffix() -> str:
    """每个测试用的唯一后缀。"""
    _COUNTER["n"] += 1
    return f"{_COUNTER['n']}_{os.getpid()}"


def _build_test_entities(task_name: str = "") -> Dict[str, Any]:
    """构造 1 task + 1 rollout + 2 attempts + 3 segments 的最小数据集。

    参数 task_name 允许每个测试传入独立的名字，避免名称冲突。
    """
    if not task_name:
        task_name = f"test_task_{_unique_suffix()}"
    task = DBTaskService.create_task(
        task_name=task_name,
        description="集成测试任务",
        source_type="manual",
        created_by="unittest",
        status="PENDING",
    )
    rollout = DBRolloutService.upsert_with_fingerprint(
        task_id=task.primary_id,
        rollout_id=0,
        pt_path=f"/tmp/test/{_unique_suffix()}/rollout_0.pt",
        pt_size_bytes=1024,
        pt_mtime=1718000000.0,
        mean_reward=0.75,
        n_samples=10,
        n_attempts=2,
        instance_count=2,
    )
    attempt_a = DBAttemptService.upsert(
        task_id=task.primary_id,
        rollout_id=0,
        group_index=0,
        attempt_id=0,
        sandbox_id="sandbox-A",
        instance_id="inst-A",
        status="completed",
        reward=1.0,
        agent_turns=5,
        tool_calls=3,
        diff_line_count=12,
        response_length=2048,
        trainable_tokens=512,
        finalized=True,
    )
    attempt_b = DBAttemptService.upsert(
        task_id=task.primary_id,
        rollout_id=0,
        group_index=0,
        attempt_id=1,
        sandbox_id="sandbox-B",
        instance_id="inst-B",
        status="failed",
        reward=0.0,
        agent_turns=2,
        tool_calls=0,
        diff_line_count=0,
        response_length=512,
        trainable_tokens=64,
        finalized=False,
    )
    segments = []
    for idx, attempt in enumerate((attempt_a, attempt_a, attempt_b)):
        seg = DBSegmentService.upsert(
            attempt_id=attempt.primary_id,
            segment_index=idx,
            segment_count=3,
            status="ok",
            reward=attempt.reward,
            response_length=512,
            trainable_tokens=128,
            mask_sum=128,
            sandbox_id=attempt.sandbox_id,
        )
        segments.append(seg)
    return {
        "task": task,
        "rollout": rollout,
        "attempt_a": attempt_a,
        "attempt_b": attempt_b,
        "segments": segments,
    }


class TestDbModels(unittest.TestCase):
    """5 张表 ORM + Service 集成测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        """建表：5 张表全部由 Base 自动创建。"""
        Base.metadata.create_all(bind=engine)

    @classmethod
    def tearDownClass(cls) -> None:
        """关闭引擎以便 unittest runner 能干净退出。"""
        engine.dispose()

    def test_tables_created(self) -> None:
        """5 张表必须存在于 metadata 中。"""
        tables = set(Base.metadata.tables.keys())
        for name in (
            "ri_task_table",
            "ri_rollout_table",
            "ri_attempt_table",
            "ri_segment_table",
            "ri_segment_transcript_table",
        ):
            self.assertIn(name, tables, f"{name} 未创建")

    def test_chain_creation_and_lookup(self) -> None:
        """完整链路创建后能依次按主键查到 5 张表的记录。"""
        suffix = _unique_suffix()
        task_name = f"chain_task_{suffix}"
        ids = _build_test_entities(task_name=task_name)
        task = DBTaskService.get_by_id(ids["task"].primary_id)
        self.assertIsNotNone(task)
        self.assertEqual(task.task_name, task_name)

        rollout = DBRolloutService.get_by_id(ids["rollout"].primary_id)
        self.assertIsNotNone(rollout)
        self.assertEqual(rollout.rollout_id, 0)
        self.assertEqual(rollout.task_id, task.primary_id)

        attempts = DBAttemptService.list_by_rollout(task.primary_id, 0)
        self.assertEqual(len(attempts), 2)

        segments = DBSegmentService.list_by_attempt(ids["attempt_a"].primary_id)
        self.assertEqual(len(segments), 2)
        # 3 号 attempt 的 1 个 segment 也必须存在
        seg_b = DBSegmentService.list_by_attempt(ids["attempt_b"].primary_id)
        self.assertEqual(len(seg_b), 1)

    def test_count_by_rollout_for_task(self) -> None:
        """count_by_rollout_for_task 按业务键 rollout_id 聚合 segment 数。"""
        ids = _build_test_entities(task_name=f"count_seg_{_unique_suffix()}")
        task = ids["task"]
        counts = DBSegmentService.count_by_rollout_for_task(task.primary_id)
        # _build_test_entities：rollout_id=0 下共 3 个 segment（attempt_a×2 + attempt_b×1）
        self.assertEqual(counts.get(0), 3)
        self.assertEqual(DBSegmentService.count_by_rollout_for_task(-1), {})

    def test_unique_constraint_rollout(self) -> None:
        """rollout 表 (task_id, rollout_id) 唯一约束必须抛 IntegrityError。"""
        ids = _build_test_entities(task_name=f"uq_rollout_{_unique_suffix()}")
        with Session() as session:
            dup = DBRollout(
                task_id=ids["task"].primary_id,
                rollout_id=0,
                pt_path=f"/tmp/test/{_unique_suffix()}/rollout_0_dup.pt",
                pt_size_bytes=2048,
                pt_mtime=1718999999.0,
            )
            session.add(dup)
            with self.assertRaises(IntegrityError):
                session.commit()
            session.rollback()

    def test_unique_constraint_attempt(self) -> None:
        """attempt 表 (task, rollout, group, attempt) 唯一约束必须抛 IntegrityError。"""
        ids = _build_test_entities(task_name=f"uq_attempt_{_unique_suffix()}")
        with Session() as session:
            dup = DBAttempt(
                task_id=ids["task"].primary_id,
                rollout_id=0,
                group_index=0,
                attempt_id=0,
            )
            session.add(dup)
            with self.assertRaises(IntegrityError):
                session.commit()
            session.rollback()

    def test_unique_constraint_segment(self) -> None:
        """segment 表 (attempt_id, segment_index) 唯一约束必须抛 IntegrityError。"""
        ids = _build_test_entities(task_name=f"uq_segment_{_unique_suffix()}")
        with Session() as session:
            dup = DBSegment(
                attempt_id=ids["attempt_a"].primary_id,
                segment_index=0,
            )
            session.add(dup)
            with self.assertRaises(IntegrityError):
                session.commit()
            session.rollback()

    def test_unique_constraint_segment_transcript(self) -> None:
        """transcript 表 segment_id 唯一约束必须抛 IntegrityError。"""
        ids = _build_test_entities(task_name=f"uq_transcript_{_unique_suffix()}")
        seg_id = ids["segments"][0].primary_id
        # 首条 transcript
        DBSegmentTranscriptService.upsert_transcript(
            segment_id=seg_id,
            prompt_text="p",
            response_text="r",
        )
        with Session() as session:
            dup = DBSegmentTranscript(
                segment_id=seg_id,
                prompt_text="dup",
                response_text="dup",
            )
            session.add(dup)
            with self.assertRaises(IntegrityError):
                session.commit()
            session.rollback()

    def test_unique_constraint_task_name(self) -> None:
        """task 表 task_name 唯一约束：create_task 抛 RolloutInsightException。"""
        name = f"dup_task_{_unique_suffix()}"
        DBTaskService.create_task(task_name=name, status="PENDING")
        with self.assertRaises(RolloutInsightException):
            DBTaskService.create_task(task_name=name, status="PENDING")

    def test_upsert_with_fingerprint_hit(self) -> None:
        """相同指纹二次 upsert 应命中已存在行并更新摘要字段，不增加新行。"""
        task = DBTaskService.create_task(
            task_name=f"fingerprint_task_{_unique_suffix()}", status="PENDING"
        )
        # 注意：相同指纹调用必须使用完全一致的 pt_path，因此这里把 path 提到外面
        pt_path = f"/tmp/fp/{_unique_suffix()}/rollout_42.pt"
        r1 = DBRolloutService.upsert_with_fingerprint(
            task_id=task.primary_id,
            rollout_id=42,
            pt_path=pt_path,
            pt_size_bytes=128,
            pt_mtime=1.0,
            mean_reward=0.1,
            n_samples=1,
            n_attempts=1,
            instance_count=1,
        )
        r2 = DBRolloutService.upsert_with_fingerprint(
            task_id=task.primary_id,
            rollout_id=42,
            pt_path=pt_path,
            pt_size_bytes=128,
            pt_mtime=1.0,
            mean_reward=0.9,
            n_samples=10,
            n_attempts=3,
            instance_count=4,
        )
        self.assertEqual(r1.primary_id, r2.primary_id)
        self.assertAlmostEqual(r2.mean_reward, 0.9, places=6)
        self.assertEqual(r2.n_samples, 10)

    def test_attempt_upsert_updates(self) -> None:
        """attempt.upsert 已存在行应更新 status / reward。"""
        task = DBTaskService.create_task(
            task_name=f"attempt_upsert_task_{_unique_suffix()}", status="PENDING"
        )
        DBAttemptService.upsert(
            task_id=task.primary_id,
            rollout_id=0,
            group_index=0,
            attempt_id=0,
            status="running",
            reward=0.1,
        )
        again = DBAttemptService.upsert(
            task_id=task.primary_id,
            rollout_id=0,
            group_index=0,
            attempt_id=0,
            status="completed",
            reward=0.9,
        )
        self.assertEqual(again.status, "completed")
        self.assertAlmostEqual(again.reward or 0.0, 0.9, places=6)

    def test_transcript_upsert_and_mark(self) -> None:
        """transcript upsert 后能命中更新；mark_fetched 不存在行返回 False。"""
        task = DBTaskService.create_task(
            task_name=f"transcript_task_{_unique_suffix()}", status="PENDING"
        )
        DBRolloutService.upsert_with_fingerprint(
            task_id=task.primary_id,
            rollout_id=0,
            pt_path=f"/tmp/tr/{_unique_suffix()}/rollout_0.pt",
            pt_size_bytes=10,
            pt_mtime=2.0,
        )
        attempt = DBAttemptService.upsert(
            task_id=task.primary_id,
            rollout_id=0,
            group_index=0,
            attempt_id=0,
            status="ok",
        )
        seg = DBSegmentService.upsert(
            attempt_id=attempt.primary_id,
            segment_index=0,
            status="ok",
            response_length=42,
            trainable_tokens=10,
            mask_sum=10,
        )
        # upsert
        row = DBSegmentTranscriptService.upsert_transcript(
            segment_id=seg.primary_id,
            prompt_text="P",
            response_text="R",
            turns=[{"role": "user", "content": "P"}],
            tool_calls=[],
            mark_as_fetched=True,
        )
        self.assertEqual(row.full_text_fetched, 1)
        self.assertEqual(row.response_chars, 1)
        # 更新并显式标记为非 fetched
        row2 = DBSegmentTranscriptService.upsert_transcript(
            segment_id=seg.primary_id,
            prompt_text="P2",
            response_text="RR",
            mark_as_fetched=False,
        )
        self.assertEqual(row2.full_text_fetched, 0)
        self.assertEqual(row2.response_text, "RR")
        # mark_fetched
        ok = DBSegmentTranscriptService.mark_fetched(seg.primary_id)
        self.assertTrue(ok)
        # 不存在 segment_id
        self.assertFalse(DBSegmentTranscriptService.mark_fetched(99999999))

    def test_search_by_keyword(self) -> None:
        """关键词检索：仅在 full_text_fetched=1 的 transcript 上命中。"""
        task = DBTaskService.create_task(
            task_name=f"search_task_{_unique_suffix()}", status="PENDING"
        )
        DBRolloutService.upsert_with_fingerprint(
            task_id=task.primary_id,
            rollout_id=0,
            pt_path=f"/tmp/sk/{_unique_suffix()}/rollout_0.pt",
            pt_size_bytes=10,
            pt_mtime=3.0,
        )
        attempt = DBAttemptService.upsert(
            task_id=task.primary_id,
            rollout_id=0,
            group_index=0,
            attempt_id=0,
        )
        seg_yes = DBSegmentService.upsert(
            attempt_id=attempt.primary_id,
            segment_index=0,
            response_length=10,
            trainable_tokens=2,
            mask_sum=2,
        )
        DBSegmentTranscriptService.upsert_transcript(
            segment_id=seg_yes.primary_id,
            response_text="hello world",
            mark_as_fetched=True,
        )
        seg_no = DBSegmentService.upsert(
            attempt_id=attempt.primary_id,
            segment_index=1,
            response_length=10,
            trainable_tokens=2,
            mask_sum=2,
        )
        DBSegmentTranscriptService.upsert_transcript(
            segment_id=seg_no.primary_id,
            response_text="hello world",
            mark_as_fetched=False,  # 不入库
        )
        hits = DBSegmentTranscriptService.search_by_keyword("hello")
        hit_segment_ids = {row.segment_id for row in hits}
        self.assertIn(seg_yes.primary_id, hit_segment_ids)
        self.assertNotIn(seg_no.primary_id, hit_segment_ids)

    def test_list_tasks_with_filters(self) -> None:
        """list_tasks 支持 status / keyword 过滤 + 分页。"""
        alpha = f"filter_task_alpha_{_unique_suffix()}"
        beta = f"filter_task_beta_{_unique_suffix()}"
        DBTaskService.create_task(task_name=alpha, status="READY")
        DBTaskService.create_task(task_name=beta, status="FAILED")
        ready_only = DBTaskService.list_tasks(status="READY")
        names = [t.task_name for t in ready_only]
        self.assertIn(alpha, names)
        self.assertNotIn(beta, names)
        # keyword
        kw_hit = DBTaskService.list_tasks(keyword="alpha")
        self.assertTrue(all("alpha" in (t.task_name or "") for t in kw_hit))

    def test_update_and_delete(self) -> None:
        """update_status / delete_task 行为正确。"""
        task = DBTaskService.create_task(
            task_name=f"lifecycle_task_{_unique_suffix()}", status="PENDING"
        )
        self.assertTrue(DBTaskService.update_status(task.primary_id, "READY"))
        again = DBTaskService.get_by_id(task.primary_id)
        self.assertEqual(again.status, "READY")
        self.assertTrue(DBTaskService.delete_task(task.primary_id))
        self.assertIsNone(DBTaskService.get_by_id(task.primary_id))
        # 删除不存在的
        self.assertFalse(DBTaskService.delete_task(88888888))


if __name__ == "__main__":
    unittest.main()
