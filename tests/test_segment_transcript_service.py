#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: segment_transcript_service 的集成测试（基于真实 rollout_0.pt）。

依赖：
- 真实 .pt 文件 /mnt/zj-data/data/songquanheng/output/slime/021_32b_agentic_scale_swe_1000_optim_20261001_195820/rollout_dumps/rollout_0.pt
- 内存/临时 SQLite 库：通过 TEST_DB_PATH= 控制；缺省走 data/rollout_insight.db

为不污染开发库，测试在临时 SQLite 上跑，run 完自行清理。
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# (shutil/json 在 tearDownClass/调试预留，目前未直接使用)


# 强制优先用一个临时 SQLite（避免污染 .db），导入后再设置 config
_TEST_DB_DIR = tempfile.mkdtemp(prefix="rollout_insight_transcript_test_")
_TEST_DB_PATH = os.path.join(_TEST_DB_DIR, "transcript_test.db")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


DUMP_PATH = Path(
    "/mnt/zj-data/data/songquanheng/output/slime/"
    "021_32b_agentic_scale_swe_1000_optim_20261001_195820/rollout_dumps/rollout_0.pt"
)


# 在 import 任何 db_*.py 之前，先把 config 切到临时 SQLite 路径 + database_url，
# 这样后续导入的 db_element 会用我们的临时库做 engine。
from utils.config_loader import config  # noqa: E402

config._data["db_path"] = _TEST_DB_PATH  # type: ignore[attr-defined]
_TEST_DB_URL = f"sqlite:///{_TEST_DB_PATH}"
config._data["database_url"] = _TEST_DB_URL  # type: ignore[attr-defined]


def _reload_db_modules() -> Any:
    """把 db.db_element 的 engine/Session 重新绑定到当前 config 的 SQLite 路径并建表。

    注意：不能清掉 ``db.db_element`` 模块本身 —— 否则 ``Base = declarative_base()``
    会重新实例化，丢失所有 ORM 表元数据。我们只重新创建 engine 并修改模块级
    engine/Session 引用。
    """
    import db.db_element as dbel
    from db.db_element import Base

    # 解析当前 config 期望的 db 路径
    current_url = config._data.get("database_url") or ""
    current_path = config._data.get("db_path") or ""
    if current_url.startswith("sqlite:///"):
        target_db_path = current_url[len("sqlite:///"):]
    else:
        target_db_path = current_path

    # 若当前 engine 已指向 target_db_path，直接 create_all；否则新建 engine。
    url_now = str(dbel.engine.url) if dbel.engine else ""
    target_url = f"sqlite:///{target_db_path}"
    if url_now != target_url:
        # 删除旧 db 文件，避免残留
        try:
            if target_db_path and os.path.exists(target_db_path):
                os.remove(target_db_path)
        except OSError:
            pass
        os.makedirs(os.path.dirname(target_db_path), exist_ok=True)
        new_engine = create_engine(
            target_url,
            connect_args={"check_same_thread": False},
        )
        NewSession = sessionmaker(
            bind=new_engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        dbel.engine = new_engine
        dbel.Session = NewSession
        dbel.db_path = target_db_path
    # 清掉 service.* 和 db.*_service.* 模块缓存，否则它们 import 时拿到的 Session
    # 会固定到模块 globals 中，导致 db_element 切 engine 后这些 service 仍在用旧 engine。
    # 注意：只清 service 层模块，不动 ORM 模型（db.db_task / db.db_element 等），
    # 否则会丢失 Base.metadata 上的表元数据。
    service_modules = [
        name for name in list(sys.modules) if name.startswith("service.segment_")
    ]
    db_service_modules = [
        name for name in list(sys.modules)
        if (name.startswith("db.db_") and name.endswith("_service"))
    ]
    for name in service_modules + db_service_modules:
        sys.modules.pop(name, None)
    Base.metadata.create_all(bind=dbel.engine)
    return dbel.engine


def _build_env() -> Dict[str, Any]:
    """重写 config.db_path 指向临时库，再建表。"""
    return {"engine": _reload_db_modules()}


def _bootstrap_db(rollout_pt_path: str) -> Dict[str, int]:
    """往里写一条 fake task / rollout / attempt / segment，返回关键 id。

    segment 行选用 sample idx=0（group=0、rollout_id=0、segment_index=0、reward=0）。
    """
    from db.db_task_service import DBTaskService
    from db.db_rollout_service import DBRolloutService
    from db.db_attempt_service import DBAttemptService
    from db.db_segment_service import DBSegmentService

    task = DBTaskService.create_task(
        task_name="transcript_test_task",
        description="unit test for ensure_transcript",
        source_type="manual",
    )
    rollout = DBRolloutService.upsert_with_fingerprint(
        task_id=task.primary_id,
        rollout_id=0,
        pt_path=rollout_pt_path,
        pt_size_bytes=int(os.path.getsize(rollout_pt_path)),
        pt_mtime=os.path.getmtime(rollout_pt_path),
        mean_reward=0.0,
        n_samples=1,
        n_attempts=1,
        instance_count=1,
    )
    attempt = DBAttemptService.upsert(
        task_id=task.primary_id,
        rollout_id=0,
        group_index=0,
        attempt_id=0,
        sandbox_id="sandbox-test",
        instance_id="inst-test",
        status="completed",
        reward=0.0,
        agent_turns=1,
        tool_calls=0,
        diff_line_count=0,
        response_length=0,
        trainable_tokens=0,
        finalized=True,
    )
    segment = DBSegmentService.upsert(
        attempt_id=attempt.primary_id,
        segment_index=0,
        segment_count=3,
        status="completed",
        reward=0.0,
        response_length=28786,
        trainable_tokens=5024,
        mask_sum=5024,
        sandbox_id="sandbox-test",
    )
    return {
        "task_id": task.primary_id,
        "rollout_id": rollout.primary_id,
        "attempt_id": attempt.primary_id,
        "segment_id": segment.primary_id,
    }


class TestParseTranscriptMd(unittest.TestCase):
    """_parse_transcript_md 单元测试（不依赖 db）。"""

    def test_empty(self) -> None:
        from service.segment_transcript_service import _parse_transcript_md

        turns, tool_calls = _parse_transcript_md("")
        self.assertEqual(turns, [])
        self.assertEqual(tool_calls, [])

    def test_simple_turn_split(self) -> None:
        from service.segment_transcript_service import _parse_transcript_md

        md = (
            "header1\n\n### turn 1 · assistant\n"
            "hello\n\n"
            "### turn 2 · user\n"
            "world\n"
        )
        turns, tool_calls = _parse_transcript_md(md)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["role"], "assistant")
        self.assertIn("hello", turns[0]["content"])
        self.assertEqual(turns[1]["role"], "user")
        self.assertEqual(tool_calls, [])

    def test_tool_call_extraction(self) -> None:
        from service.segment_transcript_service import _parse_transcript_md

        md = (
            "### turn 1 · assistant\n"
            "<tool_call>opensandbox_terminal<arg_key>command</arg_key>"
            "<arg_value>ls</arg_value>"
            "<arg_key>security_risk</arg_key><arg_value>LOW</arg_value></tool_call>\n"
        )
        turns, tool_calls = _parse_transcript_md(md)
        self.assertEqual(len(turns), 1)
        self.assertEqual(len(tool_calls), 1)
        tc = tool_calls[0]
        self.assertEqual(tc["name"], "opensandbox_terminal")
        self.assertEqual(tc["arguments"]["command"], "ls")
        self.assertEqual(tc["arguments"]["security_risk"], "LOW")
        self.assertEqual(tc["turn_role"], "assistant")


class TestEnsureTranscriptIntegration(unittest.TestCase):
    """端到端：依赖真实 .pt 文件。"""

    @classmethod
    def setUpClass(cls) -> None:
        if not DUMP_PATH.is_file():
            raise unittest.SkipTest(f"rollout dump 不存在: {DUMP_PATH}")
        _build_env()
        cls.ids = _bootstrap_db(str(DUMP_PATH))
        cls.segment_id = cls.ids["segment_id"]

    @classmethod
    def tearDownClass(cls) -> None:
        # 删除临时 SQLite
        try:
            shutil.rmtree(_TEST_DB_DIR, ignore_errors=True)
        except Exception:
            pass

    def test_ensure_first_time_writes_transcript(self) -> None:
        from db.db_segment_transcript_service import DBSegmentTranscriptService
        from service.segment_transcript_service import ensure_transcript

        result = ensure_transcript(self.segment_id, tokenizer_dir="", force=False)
        self.assertEqual(result["segment_id"], self.segment_id)
        self.assertEqual(result["full_text_fetched"], 1)
        self.assertGreater(len(result["turns_json"]), 0)
        self.assertIsInstance(result["turns_json"], list)
        # tool_calls_json 应是 list（即便为空）
        self.assertIsInstance(result["tool_calls_json"], list)
        self.assertGreater(result["response_chars"], 0)

        # 校验 DB 已落库
        row = DBSegmentTranscriptService.get_by_segment_id(self.segment_id)
        self.assertIsNotNone(row)
        self.assertEqual(row.full_text_fetched, 1)

    def test_ensure_second_call_uses_cache(self) -> None:
        """第二次调用不应该触发 .pt 解析（缓存命中）。"""
        from db.db_element import Session
        from db.db_segment_transcript import DBSegmentTranscript
        from service.segment_transcript_service import ensure_transcript

        # 先保证行存在
        ensure_transcript(self.segment_id, tokenizer_dir="", force=False)

        # 修改 prompt_text 为 sentinel，验证第二次调用不会重写
        with Session() as session:
            row = (
                session.query(DBSegmentTranscript)
                .filter(DBSegmentTranscript.segment_id == self.segment_id)
                .first()
            )
            self.assertIsNotNone(row)
            row.prompt_text = "SENTINEL_FROM_CACHE"
            session.commit()

        result = ensure_transcript(self.segment_id, tokenizer_dir="", force=False)
        # 缓存命中：prompt_text 应保持 SENTINEL_FROM_CACHE
        self.assertEqual(result["prompt_text"], "SENTINEL_FROM_CACHE")
        self.assertEqual(result["full_text_fetched"], 1)


class TestExtractRolloutSampleIndex(unittest.TestCase):
    """_extract_rollout_sample_index 单元测试。"""

    @unittest.skipUnless(DUMP_PATH.is_file(), f"dump 缺失: {DUMP_PATH}")
    def test_lookup_first_segment(self) -> None:
        from service.segment_transcript_service import _extract_rollout_sample_index

        idx = _extract_rollout_sample_index(str(DUMP_PATH), 0, 0, 0)
        self.assertIsInstance(idx, int)
        self.assertEqual(idx, 0)  # 第一段应在行 0

    @unittest.skipUnless(DUMP_PATH.is_file(), f"dump 缺失: {DUMP_PATH}")
    def test_lookup_missing_segment_raises(self) -> None:
        from data.rollout_insight_exception import RolloutInsightException
        from service.segment_transcript_service import _extract_rollout_sample_index

        with self.assertRaises(RolloutInsightException):
            _extract_rollout_sample_index(str(DUMP_PATH), 999, 99, 99)


if __name__ == "__main__":
    unittest.main()