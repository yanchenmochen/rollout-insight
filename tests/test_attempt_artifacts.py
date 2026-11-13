"""utils.attempt_artifacts：目录定位、summary 索引 / sidecar、patch 与轨迹分页。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.attempt_artifacts import (
    clear_artifact_cache_for_tests,
    find_trajectory_path,
    load_patch,
    load_summary,
    load_summary_index,
    load_trajectory_events,
    locate_artifacts_dir,
    strip_ansi,
)

_STDOUT = (
    "=================================== FAILURES ===================================\n"
    "___________________ test_parse_order ___________________\n"
    "E       AssertionError: expected 1\n"
)


def _write_summary(root: Path, stem: str, **fields) -> None:
    (root / "summaries" / f"{stem}.json").write_text(json.dumps(fields), encoding="utf-8")


def _make_run(tmp: Path) -> Path:
    run = tmp / "run"
    dumps = run / "generated" / "rollout_dumps"
    dumps.mkdir(parents=True)
    art = run / "attempt_artifacts"
    for sub in ("summaries", "patches", "failed_trajectories"):
        (art / sub).mkdir(parents=True)
    _write_summary(
        art, "repo_pr1__aaa-1", attempt_id="aaa-1", instance_id="repo_pr1", group_index=3,
        rollout_id=24, sample_index=24, status="completed", reward=0.0, failed=True,
        diagnosis="evaluator:tests_failed", evaluation={"exit_code": 1, "stdout_tail": _STDOUT},
        phase_times={"phase_time/attempt_total": 12.5},
    )
    _write_summary(
        art, "repo_pr1__bbb-2", attempt_id="bbb-2", instance_id="repo_pr1", group_index=3,
        rollout_id=None, sample_index=25, status="aborted", reward=0.0, failed=True,
        abort_reason="exception:SandboxInternalException",
    )
    (art / "patches" / "repo_pr1__aaa-1.patch").write_text("diff --git a/x b/x\n+new\n", encoding="utf-8")
    events = [
        {"kind": "ActionEvent", "source": "agent", "tool_name": "execute_bash",
         "summary": "execute_bash: pytest", "action": {"command": "pytest -x"}, "thought": [{"text": "run tests"}]},
        {"kind": "ObservationEvent", "source": "environment", "tool_name": "execute_bash",
         "observation": {"content": [{"type": "text", "text": "\x1b[31mFAILED\x1b[0m"}], "exit_code": 1}},
        {"kind": "ConversationErrorEvent", "source": "environment", "code": "LLMBadRequestError",
         "detail": "context limit reached"},
    ]
    with open(art / "failed_trajectories" / "repo_pr1__aaa-1.history.jsonl", "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    return dumps


class TestAttemptArtifacts(unittest.TestCase):
    def setUp(self) -> None:
        clear_artifact_cache_for_tests()
        self._tmp = tempfile.TemporaryDirectory()
        self.dumps = _make_run(Path(self._tmp.name))
        self.art = locate_artifacts_dir(str(self.dumps))

    def tearDown(self) -> None:
        self._tmp.cleanup()
        clear_artifact_cache_for_tests()

    def test_locate_from_rollout_dumps(self):
        self.assertIsNotNone(self.art)
        self.assertEqual(self.art.name, "attempt_artifacts")
        self.assertIsNone(locate_artifacts_dir(""))

    def test_index_fields_and_trajectory_flag(self):
        entries = {e["stem"]: e for e in load_summary_index(self.art)}
        fail = entries["repo_pr1__aaa-1"]
        self.assertEqual((fail["group_index"], fail["rollout_id"]), (3, 24))
        self.assertEqual(fail["failed_test"], "test_parse_order")
        self.assertEqual(fail["error_summary"], "AssertionError: expected 1")
        self.assertEqual(fail["trajectory_kind"], "failed_trajectories")
        abort = entries["repo_pr1__bbb-2"]
        self.assertIsNone(abort["rollout_id"])
        self.assertEqual(abort["abort_reason"], "exception:SandboxInternalException")
        self.assertIsNone(abort["trajectory_kind"])

    def test_sidecar_reused_after_cache_clear(self):
        load_summary_index(self.art)
        self.assertTrue((self.art / ".ri_artifacts_index.json").is_file())
        clear_artifact_cache_for_tests()
        (self.art / "summaries" / "repo_pr1__aaa-1.json").write_text("{}", encoding="utf-8")
        entries = {e["stem"]: e for e in load_summary_index(self.art)}
        self.assertEqual(entries["repo_pr1__aaa-1"]["failed_test"], "test_parse_order")

    def test_patch_and_summary(self):
        patch = load_patch(self.art, "repo_pr1__aaa-1")
        self.assertTrue(patch["exists"])
        self.assertIn("+new", patch["text"])
        self.assertFalse(load_patch(self.art, "repo_pr1__bbb-2")["exists"])
        self.assertEqual(load_summary(self.art, "repo_pr1__aaa-1")["attempt_id"], "aaa-1")
        with self.assertRaises(ValueError):
            load_summary(self.art, "../etc")

    def test_trajectory_events_and_errors(self):
        path = find_trajectory_path(self.art, "repo_pr1__aaa-1")
        data = load_trajectory_events(path, limit=10)
        self.assertEqual(data["total"], 3)
        self.assertEqual(data["n_errors"], 2)
        self.assertEqual(data["tool_counts"], {"execute_bash": 1})
        obs = data["events"][1]
        self.assertTrue(obs["is_error"])
        self.assertEqual(obs["text"], "FAILED")
        self.assertEqual(data["last_error"]["title"], "LLMBadRequestError")
        only = load_trajectory_events(path, errors_only=True, offset=1, limit=1)
        self.assertEqual(only["n_matched"], 2)
        self.assertEqual([e["index"] for e in only["events"]], [2])

    def test_strip_ansi(self):
        self.assertEqual(strip_ansi("\x1b[1mok\x1b[0m"), "ok")


if __name__ == "__main__":
    unittest.main()
