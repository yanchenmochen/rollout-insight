"""RolloutScanService 单元测试。

由于本测试运行在 2A worker（db ORM + Service）尚未合并的阶段，
我们采用双轨策略：

1. **优先尝试真实导入**：若 db/db_task.py 等模块已就绪，则直接
   ``unittest.mock.patch`` 实际 Service，避免重建内存数据库。

2. **退路 stub 实现**：若 db 模块未就绪，则在 ``tests/_stubs`` 中动态
   注入 fake ORM + Service，再 import ``service.rollout_scan_service``。

测试流程：
- 在临时目录 fake 2 个 .pt（通过 unittest.mock 拦截
  ``utils.analyze_dump.load_dump_meta`` / ``summarize_dump``）
- 跑 ``scan_task``，验证 5 张表的 mock Service 都被正确调用

最终断言：
- ``DBRolloutService.upsert_with_fingerprint`` 被调用 N 次
- ``DBAttemptService.upsert`` 被调用 M 次（每个 sample 一行 attempt）
- ``DBSegmentService.upsert`` 被调用 M 次
- ``DBTaskService.update_status`` 调用序列包含 ``SCANNING -> READY``
- ``rescan_task`` 时 ``*_Service.delete_by_task`` 都被调用
- ``get_progress`` 返回的字段非空
"""
from __future__ import annotations

import importlib
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]


def _silence():
    """屏蔽导入/解析阶段对 stdout/stderr 的噪声。"""
    return redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())


# --------------------------------------------------------------------------
# stub 层：若 db/* 不存在，注入伪造 ORM + Service，使 RolloutScanService
# 能正常 import / 运行
# --------------------------------------------------------------------------


class _StubORM:
    """最小化的 ORM 行对象，用作返回。"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _StubTaskService:
    """DBTaskService 的最小 stub。"""

    _status: Dict[int, str] = {}
    _tasks: Dict[int, Any] = {}

    @classmethod
    def reset(cls):
        cls._status.clear()
        cls._tasks.clear()

    @classmethod
    def get_by_id(cls, task_id: int):
        return cls._tasks.get(task_id)

    @classmethod
    def update_status(cls, primary_id: int, status: str) -> bool:
        t = cls._tasks.get(primary_id)
        if not t:
            return False
        t.status = status
        cls._status[primary_id] = status
        return True


class _StubRolloutService:
    counter = 0
    calls: List[Dict[str, Any]] = []

    @classmethod
    def reset(cls):
        cls.counter = 0
        cls.calls.clear()

    @classmethod
    def upsert_with_fingerprint(cls, **kwargs):
        cls.counter += 1
        cls.calls.append(kwargs)
        return _StubORM(primary_id=cls.counter, **kwargs)

    @classmethod
    def list_by_task(cls, task_id: int, limit: int = 200, offset: int = 0) -> List[Any]:
        rows = [
            _StubORM(**c) for c in cls.calls if c.get("task_id") == task_id
        ]
        return rows[offset : offset + limit]

    @classmethod
    def delete_by_task(cls, task_id: int) -> int:
        before = len(cls.calls)
        cls.calls = [c for c in cls.calls if c.get("task_id") != task_id]
        return before - len(cls.calls)


class _StubAttemptService:
    counter = 0
    calls: List[Dict[str, Any]] = []

    @classmethod
    def reset(cls):
        cls.counter = 0
        cls.calls.clear()

    @classmethod
    def upsert(cls, **kwargs):
        cls.counter += 1
        cls.calls.append(kwargs)
        return _StubORM(primary_id=cls.counter, **kwargs)

    @classmethod
    def list_by_task(cls, task_id: int, limit: int = 1000, offset: int = 0) -> List[Any]:
        rows = [
            _StubORM(**c) for c in cls.calls if c.get("task_id") == task_id
        ]
        return rows[offset : offset + limit]

    @classmethod
    def delete_by_task(cls, task_id: int) -> int:
        before = len(cls.calls)
        cls.calls = [c for c in cls.calls if c.get("task_id") != task_id]
        return before - len(cls.calls)


class _StubSegmentService:
    counter = 0
    calls: List[Dict[str, Any]] = []

    @classmethod
    def reset(cls):
        cls.counter = 0
        cls.calls.clear()

    @classmethod
    def upsert(cls, **kwargs):
        cls.counter += 1
        cls.calls.append(kwargs)
        return _StubORM(primary_id=cls.counter, **kwargs)

    @classmethod
    def list_by_task(cls, task_id: int, limit: int = 2000, offset: int = 0) -> List[Any]:
        rows = [
            _StubORM(**c) for c in cls.calls if c.get("task_id") == task_id
        ]
        return rows[offset : offset + limit]

    @classmethod
    def delete_by_task(cls, task_id: int) -> int:
        before = len(cls.calls)
        cls.calls = [c for c in cls.calls if c.get("task_id") != task_id]
        return before - len(cls.calls)


def _ensure_stub_db_modules(force: bool = False) -> bool:
    """注入 stub db service 模块。

    Args:
        force: True 时即使真实模块已存在也强制覆盖（用于单元测试中隔离真实 ORM）。
               默认 False 表示只在缺失时注入，避免破坏后续 e2e 测试的副作用。

    Returns:
        True 表示注入了 stub。
    """
    needed = [
        "db.db_task_service",
        "db.db_rollout_service",
        "db.db_attempt_service",
        "db.db_segment_service",
    ]
    if not force:
        missing = [m for m in needed if m not in sys.modules]
        if not missing:
            return False

    # 否则把 db 包和具体子模块都注册成 stub
    import types

    _STUB_SNAPSHOT["db"] = sys.modules.get("db")

    if "db" not in sys.modules:
        db_pkg = types.ModuleType("db")
        db_pkg.__path__ = [str(REPO_ROOT / "db")]
        sys.modules["db"] = db_pkg

    for mod_name in needed:
        _STUB_SNAPSHOT[mod_name] = sys.modules.get(mod_name)

    db_task_service = types.ModuleType("db.db_task_service")
    db_task_service.DBTaskService = _StubTaskService  # type: ignore[attr-defined]
    sys.modules["db.db_task_service"] = db_task_service

    db_rollout_service = types.ModuleType("db.db_rollout_service")
    db_rollout_service.DBRolloutService = _StubRolloutService  # type: ignore[attr-defined]
    sys.modules["db.db_rollout_service"] = db_rollout_service

    db_attempt_service = types.ModuleType("db.db_attempt_service")
    db_attempt_service.DBAttemptService = _StubAttemptService  # type: ignore[attr-defined]
    sys.modules["db.db_attempt_service"] = db_attempt_service

    db_segment_service = types.ModuleType("db.db_segment_service")
    db_segment_service.DBSegmentService = _StubSegmentService  # type: ignore[attr-defined]
    sys.modules["db.db_segment_service"] = db_segment_service
    return True


_STUB_SNAPSHOT: Dict[str, Any] = {}


def _restore_stub_db_modules() -> None:
    """把之前注入的 stub 还原到注入前的状态。"""
    for name, original in _STUB_SNAPSHOT.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    _STUB_SNAPSHOT.clear()


_SERVICES_MODULES = (
    "service.rollout_scan_service",
    "service",
)


def _reload_services_module() -> None:
    """让 service/ 重新基于 db 状态加载一次。"""
    for name in _SERVICES_MODULES:
        if name in sys.modules:
            importlib.reload(sys.modules[name])


def _setup_path() -> None:
    """把 repo 根目录加进 sys.path。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _make_fake_pt(tmpdir: Path, rollout_id: int, n_samples: int) -> Path:
    """在临时目录创建一个假 .pt 文件（实际内容不重要，我们 mock load/summarize）。"""
    p = tmpdir / f"rollout_{rollout_id}.pt"
    p.write_bytes(b"FAKE_PT")
    return p


def _meta_for(rollout_id: int, n_samples: int) -> Dict[str, Any]:
    return {
        "rollout_id": rollout_id,
        "n_samples": n_samples,
        "n_attempts": n_samples,
        "mean_reward": 0.5,
        "pt_path": f"<fake>/rollout_{rollout_id}.pt",
    }


def _rows_for(rollout_id: int, n_samples: int) -> List[Dict[str, Any]]:
    rows = []
    for i in range(n_samples):
        rows.append(
            {
                "idx": i,
                "group_index": i,
                "rollout_id": rollout_id,
                "segment_index": i,
                "segment_count": n_samples,
                "instance_id": f"inst-{rollout_id}-{i}",
                "reward": 0.5,
                "status": "completed",
                "agent_turns": 3,
                "tool_calls": 2,
                "diff_line_count": 10,
                "response_length": 100,
                "mask_sum": 50,
                "sandbox_id": f"sx-{rollout_id}",
            }
        )
    return rows


# --------------------------------------------------------------------------
# 测试用例
# --------------------------------------------------------------------------


class TestRolloutScanService(unittest.TestCase):
    """端到端验证：scan_task → 5 张表被填充。"""

    def setUp(self) -> None:
        _setup_path()
        # 单元测试强制覆盖真实 ORM，避免被前面的 e2e 测试污染
        _ensure_stub_db_modules(force=True)
        _reload_services_module()

        # 重新导入（确保拿到的是 stub 版本）
        from service.rollout_scan_service import RolloutScanService  # noqa: WPS433

        self.RolloutScanService = RolloutScanService

        _StubTaskService.reset()
        _StubRolloutService.reset()
        _StubAttemptService.reset()
        _StubSegmentService.reset()

        # 屏蔽 loguru 默认配置（避免多 sink 干扰）
        from loguru import logger as _lg  # noqa: WPS433

        _lg.remove()

        self.task_id = 42
        _StubTaskService._tasks[self.task_id] = _StubORM(
            primary_id=self.task_id,
            task_name="fake-task",
            status="PENDING",
        )

    def tearDown(self) -> None:
        # 恢复 db 注入前的状态，避免污染后续测试
        _restore_stub_db_modules()
        _reload_services_module()

    def _patch_dump(self, pt_paths: List[Path], sample_counts: Dict[Path, int]):
        """mock view_trajectory.load_dump + analyze_dump.load_dump_meta/summarize_dump"""
        load_dump_meta_mock = mock.MagicMock(side_effect=lambda p: _meta_for(
            int(p.stem.split("_")[-1]), sample_counts[p]
        ))
        summarize_dump_mock = mock.MagicMock(side_effect=lambda p: _rows_for(
            int(p.stem.split("_")[-1]), sample_counts[p]
        ))
        return mock.patch.multiple(
            "service.rollout_scan_service",
            load_dump_meta=load_dump_meta_mock,
            summarize_dump=summarize_dump_mock,
        )

    def test_scan_task_fills_five_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            # 模拟 <dump_root>/<task_name>/rollout_dumps/ 结构
            task_name = "fake-task"
            dumps_dir = tmpdir / task_name / "rollout_dumps"
            dumps_dir.mkdir(parents=True)
            pt_a = _make_fake_pt(dumps_dir, rollout_id=0, n_samples=3)
            pt_b = _make_fake_pt(dumps_dir, rollout_id=1, n_samples=2)
            sample_counts = {pt_a: 3, pt_b: 2}

            _StubTaskService._tasks[self.task_id].task_name = task_name

            svc = self.RolloutScanService()
            svc.dump_root = str(tmpdir)  # 指向临时根目录

            with mock.patch.dict(
                os.environ, {"OUTPUT_DIR": "", "ROLLOUT_DUMP_DIR": ""}, clear=False
            ), self._patch_dump([pt_a, pt_b], sample_counts):
                os.environ.pop("OUTPUT_DIR", None)
                os.environ.pop("ROLLOUT_DUMP_DIR", None)
                result = svc.scan_task(self.task_id)

            # 1. 返回值正确
            self.assertEqual(result["status"], "READY")
            self.assertEqual(result["total"], 2)
            self.assertEqual(result["scanned"], 2)
            self.assertEqual(result["rollouts"], 2)
            self.assertEqual(result["attempts"], 5)  # 3 + 2
            self.assertEqual(result["segments"], 5)
            self.assertEqual(result["errors"], [])

            # 2. 5 张表（Service mock）都被填了数据
            self.assertEqual(len(_StubRolloutService.calls), 2)
            self.assertEqual(len(_StubAttemptService.calls), 5)
            self.assertEqual(len(_StubSegmentService.calls), 5)

            # 3. rollout 调用参数含 task_id + pt_path + fingerprint 由 service 计算
            for call in _StubRolloutService.calls:
                self.assertEqual(call["task_id"], self.task_id)
                self.assertIn("rollout_id", call)
                self.assertIn("pt_path", call)
                self.assertIn("pt_size_bytes", call)
                self.assertIn("pt_mtime", call)
                self.assertIn("n_samples", call)

            # 4. attempt 业务键齐全
            for call in _StubAttemptService.calls:
                self.assertEqual(call["task_id"], self.task_id)
                self.assertIn("group_index", call)
                self.assertIn("attempt_id", call)
                self.assertIn("instance_id", call)
                self.assertIn("status", call)

            # 5. segment 含 attempt_id
            for call in _StubSegmentService.calls:
                self.assertIn("attempt_id", call)
                self.assertIn("segment_index", call)
                self.assertIn("segment_count", call)
                self.assertIn("mask_sum", call)

            # 6. task 状态被推进到 READY
            self.assertEqual(_StubTaskService._status[self.task_id], "READY")

    def test_scan_task_path_priority(self) -> None:
        """OUTPUT_DIR 优先于 dump_root。"""
        with tempfile.TemporaryDirectory() as env_tmp, tempfile.TemporaryDirectory() as root_tmp:
            env_dir = Path(env_tmp)
            root_dir = Path(root_tmp)

            # env 下的 rollout_*.pt
            env_pt = _make_fake_pt(env_dir, rollout_id=0, n_samples=2)
            # dump_root/<task_name>/rollout_dumps/ 下也有一个 .pt（不应被选中）
            task_name = "fake-task"
            dumps_dir = root_dir / task_name / "rollout_dumps"
            dumps_dir.mkdir(parents=True)
            root_pt = _make_fake_pt(dumps_dir, rollout_id=99, n_samples=1)

            sample_counts = {env_pt: 2}
            _StubTaskService._tasks[self.task_id].task_name = task_name

            svc = self.RolloutScanService()
            svc.dump_root = str(root_dir)

            with mock.patch.dict(os.environ, {"OUTPUT_DIR": str(env_dir)}, clear=False), \
                 self._patch_dump([env_pt], sample_counts):
                result = svc.scan_task(self.task_id)

            # 只扫到 env_pt（root_pt 不在结果里）
            self.assertEqual(result["scanned"], 1)
            self.assertEqual(result["total"], 1)
            self.assertEqual(len(_StubRolloutService.calls), 1)
            self.assertEqual(
                _StubRolloutService.calls[0]["pt_path"], str(env_pt)
            )

    def test_scan_task_missing_task(self) -> None:
        svc = self.RolloutScanService()
        with self.assertRaises(Exception):
            svc.scan_task(task_id=9999)

    def test_rescan_task_purges_old_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            task_name = "fake-task"
            dumps_dir = tmpdir / task_name / "rollout_dumps"
            dumps_dir.mkdir(parents=True)
            _StubTaskService._tasks[self.task_id].task_name = task_name
            pt_a = _make_fake_pt(dumps_dir, rollout_id=0, n_samples=2)
            pt_b = _make_fake_pt(dumps_dir, rollout_id=1, n_samples=2)
            sample_counts = {pt_a: 2, pt_b: 2}

            svc = self.RolloutScanService()
            svc.dump_root = str(tmpdir)
            with mock.patch.dict(os.environ, {}, clear=False), \
                 self._patch_dump([pt_a, pt_b], sample_counts):
                os.environ.pop("OUTPUT_DIR", None)
                os.environ.pop("ROLLOUT_DUMP_DIR", None)
                r1 = svc.scan_task(self.task_id)
                self.assertEqual(r1["scanned"], 2)
                self.assertEqual(len(_StubRolloutService.calls), 2)

                with mock.patch.object(
                    svc, "_purge_task_data", wraps=svc._purge_task_data
                ) as purge_spy:
                    r2 = svc.rescan_task(self.task_id)
                    purge_spy.assert_called_once_with(self.task_id)
                self.assertEqual(r2["scanned"], 2)
                self.assertEqual(len(_StubRolloutService.calls), 2)

    def test_scan_task_handles_parse_error_per_pt(self) -> None:
        """某个 .pt 解析失败不应影响其他 .pt。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            task_name = "fake-task"
            dumps_dir = tmpdir / task_name / "rollout_dumps"
            dumps_dir.mkdir(parents=True)
            pt_good = _make_fake_pt(dumps_dir, rollout_id=0, n_samples=2)
            pt_bad = _make_fake_pt(dumps_dir, rollout_id=1, n_samples=2)
            sample_counts = {pt_good: 2}

            _StubTaskService._tasks[self.task_id].task_name = task_name

            svc = self.RolloutScanService()
            svc.dump_root = str(tmpdir)

            def fake_meta(p):
                if p == pt_bad:
                    raise RuntimeError("corrupted dump")
                return _meta_for(0, 2)

            def fake_rows(p):
                if p == pt_bad:
                    return []
                return _rows_for(0, 2)

            with mock.patch(
                "service.rollout_scan_service.load_dump_meta", side_effect=fake_meta
            ), mock.patch(
                "service.rollout_scan_service.summarize_dump", side_effect=fake_rows
            ):
                result = svc.scan_task(self.task_id)

            self.assertEqual(result["status"], "READY")  # partial success -> READY
            self.assertEqual(result["scanned"], 1)
            self.assertEqual(result["total"], 2)
            self.assertEqual(len(result["errors"]), 1)

    def test_get_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            task_name = "fake-task"
            dumps_dir = tmpdir / task_name / "rollout_dumps"
            dumps_dir.mkdir(parents=True)
            _StubTaskService._tasks[self.task_id].task_name = task_name
            pt_a = _make_fake_pt(dumps_dir, rollout_id=0, n_samples=2)
            sample_counts = {pt_a: 2}

            svc = self.RolloutScanService()
            svc.dump_root = str(tmpdir)
            with mock.patch.dict(os.environ, {}, clear=False), \
                 self._patch_dump([pt_a], sample_counts):
                os.environ.pop("OUTPUT_DIR", None)
                os.environ.pop("ROLLOUT_DUMP_DIR", None)
                svc.scan_task(self.task_id)
                progress = svc.get_progress(self.task_id)

            self.assertEqual(progress["task_id"], self.task_id)
            self.assertEqual(progress["status"], "READY")
            self.assertEqual(progress["rollouts"], 1)
            self.assertEqual(progress["attempts"], 2)
            # segments 通过 JOIN attempt 过滤；stub 无法模拟 JOIN，只校验 attempts/rollouts
            self.assertEqual(progress["last_rollout_id"], 0)

    def test_resolve_dump_paths_priority(self) -> None:
        """直接测试 _resolve_dump_paths 的优先级：OUTPUT_DIR 优先。"""
        with tempfile.TemporaryDirectory() as env_tmp, tempfile.TemporaryDirectory() as root_tmp:
            env_dir = Path(env_tmp)
            root_dir = Path(root_tmp)
            task_name = "fake-task"

            env_pt = _make_fake_pt(env_dir, rollout_id=0, n_samples=2)
            env_sub_pt = env_dir / "rollout_dumps"
            env_sub_pt.mkdir()
            env_rollout_sub = env_sub_pt / "rollout_77.pt"
            env_rollout_sub.write_bytes(b"x")
            # dump_root/<task_name>/rollout_dumps/ 下的 .pt
            dumps_dir = root_dir / task_name / "rollout_dumps"
            dumps_dir.mkdir(parents=True)
            root_pt = _make_fake_pt(dumps_dir, rollout_id=99, n_samples=1)

            svc = self.RolloutScanService()
            svc.dump_root = str(root_dir)

            # 1) OUTPUT_DIR 优先：拿到 env_rollout_sub + env_pt，不含 root_pt
            with mock.patch.dict(
                os.environ, {"OUTPUT_DIR": str(env_dir)}, clear=False
            ):
                paths = svc._resolve_dump_paths(task_name)
            self.assertIn(env_rollout_sub, paths)
            self.assertIn(env_pt, paths)
            self.assertNotIn(root_pt, paths)

            # 2) 没有 OUTPUT_DIR/ROLLOUT_DUMP_DIR → 回退到 dump_root/<task_name>/rollout_dumps/
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("OUTPUT_DIR", None)
                os.environ.pop("ROLLOUT_DUMP_DIR", None)
                paths2 = svc._resolve_dump_paths(task_name)
            self.assertIn(root_pt, paths2)
            self.assertNotIn(env_pt, paths2)
            self.assertNotIn(env_rollout_sub, paths2)

            # 3) ROLLOUT_DUMP_DIR 优先于 dump_root
            with mock.patch.dict(
                os.environ, {"ROLLOUT_DUMP_DIR": str(env_dir)}, clear=False
            ):
                os.environ.pop("OUTPUT_DIR", None)
                paths3 = svc._resolve_dump_paths(task_name)
            self.assertIn(env_pt, paths3)
            self.assertNotIn(root_pt, paths3)


class TestImports(unittest.TestCase):
    """保证 service.rollout_scan_service 在 db 模块未就绪时也能 import。"""

    def setUp(self) -> None:
        _setup_path()
        _ensure_stub_db_modules()
        _reload_services_module()

    def tearDown(self) -> None:
        _restore_stub_db_modules()
        _reload_services_module()

    def test_import(self) -> None:
        from service.rollout_scan_service import RolloutScanService  # noqa: WPS433

        self.assertTrue(callable(RolloutScanService.scan_task))
        self.assertTrue(callable(RolloutScanService.rescan_task))
        self.assertTrue(callable(RolloutScanService.get_progress))


if __name__ == "__main__":
    unittest.main()