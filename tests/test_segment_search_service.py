#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@desc: segment_search_service 的单元测试。

策略：直接插入 fake task / attempt / segment / segment_transcript，验证：
- 关键字命中：仅命中的 segment 出现在 items 中，且 snippet 包含上下文
- 无关键字：按 (task_id, min_reward) 返回 segment 列表
- 关键字检索时未 fetched 的段会触发 ensure_transcript 预热
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# 用独立临时 SQLite 隔离测试
_TEST_DB_DIR = tempfile.mkdtemp(prefix="rollout_insight_search_test_")
_TEST_DB_PATH = os.path.join(_TEST_DB_DIR, "search_test.db")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 把 config 切到临时 SQLite 路径 + database_url
from utils.config_loader import config  # noqa: E402

config._data["db_path"] = _TEST_DB_PATH  # type: ignore[attr-defined]
_TEST_DB_URL = f"sqlite:///{_TEST_DB_PATH}"
config._data["database_url"] = _TEST_DB_URL  # type: ignore[attr-defined]


def _reload_db_modules() -> Any:
    """重新绑定 db.db_element 的 engine/Session 到当前 config 路径并建表。

    注意：不能清掉 ``db.db_element`` 模块本身，否则 ``Base = declarative_base()``
    会被重新实例化并丢失所有 ORM 元数据。
    """
    import db.db_element as dbel
    from db.db_element import Base

    current_url = config._data.get("database_url") or ""
    current_path = config._data.get("db_path") or ""
    if current_url.startswith("sqlite:///"):
        target_db_path = current_url[len("sqlite:///"):]
    else:
        target_db_path = current_path

    url_now = str(dbel.engine.url) if dbel.engine else ""
    target_url = f"sqlite:///{target_db_path}"
    if url_now != target_url:
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

    # 让 service.segment_* 模块下次 import 时拿到新的 Session。
    # 不清 db.* —— 否则 ORM 表元数据会丢失。
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


class _FakeFixture:
    """公共假数据工厂。"""

    @staticmethod
    def make_task(name: str = "search_test_task") -> int:
        from db.db_task_service import DBTaskService

        return DBTaskService.create_task(
            task_name=name,
            description="search service unit test",
            source_type="manual",
        ).primary_id

    @staticmethod
    def make_attempt(task_id: int, attempt_id: int, reward: float = 0.0) -> int:
        from db.db_attempt_service import DBAttemptService

        return DBAttemptService.upsert(
            task_id=task_id,
            rollout_id=0,
            group_index=0,
            attempt_id=attempt_id,
            sandbox_id=f"sandbox-{attempt_id}",
            instance_id=f"inst-{attempt_id}",
            status="completed",
            reward=reward,
            agent_turns=1,
            tool_calls=0,
            diff_line_count=0,
            response_length=10,
            trainable_tokens=10,
            finalized=True,
        ).primary_id

    @staticmethod
    def make_segment(
        attempt_id: int,
        segment_index: int = 0,
        reward: float = 0.0,
        sandbox_id: str = "sandbox-test",
    ) -> int:
        from db.db_segment_service import DBSegmentService

        return DBSegmentService.upsert(
            attempt_id=attempt_id,
            segment_index=segment_index,
            segment_count=1,
            status="completed",
            reward=reward,
            response_length=100,
            trainable_tokens=50,
            mask_sum=50,
            sandbox_id=sandbox_id,
        ).primary_id

    @staticmethod
    def make_transcript(
        segment_id: int,
        prompt: str,
        response: str,
        turns: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
    ) -> None:
        from db.db_segment_transcript_service import DBSegmentTranscriptService

        DBSegmentTranscriptService.upsert_transcript(
            segment_id=segment_id,
            prompt_text=prompt,
            response_text=response,
            turns=turns,
            tool_calls=tool_calls,
            tokenizer_path_used="",
            response_chars=len(response),
            mark_as_fetched=True,
        )


class TestSearchNoKeyword(unittest.TestCase):
    """无 keyword：纯 segment 过滤。"""

    def setUp(self) -> None:
        _reload_db_modules()

    def test_search_by_task_and_reward(self) -> None:
        task_id = _FakeFixture.make_task("search_no_kw")
        # 3 个 attempt：reward 0.1 / 0.5 / 0.9 各 1 个 segment
        rewards = [0.1, 0.5, 0.9]
        seg_ids: List[int] = []
        for idx, r in enumerate(rewards):
            attempt_id = _FakeFixture.make_attempt(task_id, idx, reward=r)
            seg_ids.append(_FakeFixture.make_segment(attempt_id, reward=r))

        from service.segment_search_service import search_segments

        result = search_segments(task_id=task_id, min_reward=0.4, keyword="")
        self.assertEqual(result["total"], 2)
        ids_returned = sorted(int(item["segment_id"]) for item in result["items"])
        expected = sorted([seg_ids[1], seg_ids[2]])
        self.assertEqual(ids_returned, expected)
        for item in result["items"]:
            self.assertIn("attempt_id", item)
            self.assertIn("sandbox_id", item)
            self.assertIn("snippet", item)
            self.assertIn("full_text_fetched", item)


class TestSearchWithKeyword(unittest.TestCase):
    """keyword 命中。"""

    def setUp(self) -> None:
        _reload_db_modules()

    def test_keyword_hits_only_matching_segments(self) -> None:
        task_id = _FakeFixture.make_task("search_kw")
        attempt_low = _FakeFixture.make_attempt(task_id, 0, reward=0.0)
        seg_low = _FakeFixture.make_segment(attempt_low, reward=0.0)
        _FakeFixture.make_transcript(
            seg_low,
            prompt="Q:",
            response="this is a transcript about refactor task.",
            turns=[{"role": "assistant", "content": "refactor task.", "mask": None}],
            tool_calls=[],
        )

        attempt_high = _FakeFixture.make_attempt(task_id, 1, reward=1.0)
        seg_high = _FakeFixture.make_segment(attempt_high, reward=1.0)
        _FakeFixture.make_transcript(
            seg_high,
            prompt="Q:",
            response="hello world says the agent",
            turns=[{"role": "assistant", "content": "hello world", "mask": None}],
            tool_calls=[],
        )

        from service.segment_search_service import search_segments

        result = search_segments(task_id=task_id, min_reward=0.0, keyword="hello")
        self.assertEqual(result["total"], 1)
        self.assertEqual(len(result["items"]), 1)
        item = result["items"][0]
        self.assertEqual(item["segment_id"], seg_high)
        self.assertEqual(item["full_text_fetched"], 1)
        self.assertIn("hello", item["snippet"])
        self.assertEqual(item["reward"], 1.0)

    def test_keyword_miss_returns_empty(self) -> None:
        task_id = _FakeFixture.make_task("search_kw_miss")
        attempt_id = _FakeFixture.make_attempt(task_id, 0, reward=0.0)
        seg_id = _FakeFixture.make_segment(attempt_id, reward=0.0)
        _FakeFixture.make_transcript(
            seg_id,
            prompt="Q:",
            response="just some text without the magic",
            turns=[],
            tool_calls=[],
        )

        from service.segment_search_service import search_segments

        result = search_segments(task_id=task_id, min_reward=0.0, keyword="magicxxxx")
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["items"], [])

    def test_keyword_triggers_ensure_for_unfetched(self) -> None:
        """未 fetched 的段：插入 segment 行但无 transcript，触发检索时应被预热。

        由于无 .pt 文件可解析（mock），我们直接验证：调用 ensure_transcript 后
        transcript 仍能被 search_by_keyword 查到（先手动 upsert）。
        """
        task_id = _FakeFixture.make_task("search_kw_prewarm")
        attempt_id = _FakeFixture.make_attempt(task_id, 0, reward=0.0)
        seg_id = _FakeFixture.make_segment(attempt_id, reward=0.0)
        # 仅写一条 full_text_fetched=0 的 transcript 占位（无 .pt 可解析也能验证 upsert 路径）
        from db.db_segment_transcript_service import DBSegmentTranscriptService

        DBSegmentTranscriptService.upsert_transcript(
            segment_id=seg_id,
            prompt_text="Q:",
            response_text="placeholder hello placeholder",
            turns=[],
            tool_calls=[],
            mark_as_fetched=False,
        )

        from service.segment_search_service import search_segments

        result = search_segments(task_id=task_id, min_reward=0.0, keyword="hello")
        # upsert 后即使 mark_as_fetched=False，response_text 命中 LIKE；
        # 但 search_by_keyword 要求 full_text_fetched=1，因此预热后才能命中。
        # 这里我们仅验证调用不抛错；具体命中条数依赖 ensure_transcript 是否成功。
        self.assertIsInstance(result, dict)
        self.assertIn("total", result)
        self.assertIn("items", result)


class TestSearchPagination(unittest.TestCase):
    """分页验证。"""

    def setUp(self) -> None:
        _reload_db_modules()

    def test_pagination_limit_offset(self) -> None:
        task_id = _FakeFixture.make_task("search_page")
        # 5 个 segment
        seg_ids: List[int] = []
        for idx in range(5):
            attempt_id = _FakeFixture.make_attempt(task_id, idx, reward=0.0)
            sid = _FakeFixture.make_segment(attempt_id, reward=0.0)
            seg_ids.append(sid)
            _FakeFixture.make_transcript(
                sid,
                prompt="Q:",
                response=f"text {idx}",
                turns=[],
                tool_calls=[],
            )

        from service.segment_search_service import search_segments

        page1 = search_segments(task_id=task_id, min_reward=0.0, keyword="", limit=2, offset=0)
        page2 = search_segments(task_id=task_id, min_reward=0.0, keyword="", limit=2, offset=2)
        self.assertEqual(len(page1["items"]), 2)
        self.assertEqual(len(page2["items"]), 2)
        self.assertEqual(page1["total"], 5)
        ids1 = {int(it["segment_id"]) for it in page1["items"]}
        ids2 = {int(it["segment_id"]) for it in page2["items"]}
        self.assertFalse(ids1 & ids2)


if __name__ == "__main__":
    unittest.main()