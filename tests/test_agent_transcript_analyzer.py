"""v0.4.0 T6：agent-transcripts 分布分析单测。

对 2 个真实数据文件验证：
- analyze_transcript 返回 dict 且 JSON 可序列化
- 关键字段：total_events / role_distribution / char_distribution / implicit_rewards
- 隐式 reward 在 [0, 1]
- 端点 /api/agent-transcripts/page 渲染 HTML 含 <table> + <svg>
"""
import json
import os
import unittest
from pathlib import Path


EF2A8B4D = (
    "/root/.cursor/projects/mnt-zj-data-data-songquanheng-slime/"
    "agent-transcripts/ef2a8b4d-129d-472b-ba10-9cae83f4d080/"
    "ef2a8b4d-129d-472b-ba10-9cae83f4d080.jsonl"
)
F40B0D2F3 = (
    "/root/.cursor/projects/mnt-zj-data-data-songquanheng-slime/"
    "agent-transcripts/40b0d2f3-8432-433d-981f-61b46216f249/"
    "40b0d2f3-8432-433d-981f-61b46216f249.jsonl"
)


class TestAgentTranscriptAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import service.agent_transcript_analyzer as mod
        cls.mod = mod

    def test_extract_text_simple(self) -> None:
        m = self.mod
        msg = {"content": [{"type": "text", "text": "hello"}, {"type": "tool_use", "name": "x"}]}
        self.assertEqual(m.extract_text_from_message(msg), "hello")
        self.assertEqual(m.count_tool_calls(msg), 1)

    def test_extract_text_empty(self) -> None:
        m = self.mod
        self.assertEqual(m.extract_text_from_message(None), "")
        self.assertEqual(m.extract_text_from_message({}), "")
        self.assertEqual(m.extract_text_from_message({"content": "abc"}), "abc")

    def test_classify_user_feedback(self) -> None:
        m = self.mod
        self.assertEqual(m.classify_user_feedback("完成了"), 1.0)
        self.assertEqual(m.classify_user_feedback("好的，没问题"), 1.0)
        self.assertEqual(m.classify_user_feedback("approved"), 1.0)
        self.assertEqual(m.classify_user_feedback("不对，重来"), 0.0)
        self.assertEqual(m.classify_user_feedback("失败"), 0.0)
        self.assertEqual(m.classify_user_feedback("请分析下这个文件"), 0.5)
        self.assertEqual(m.classify_user_feedback(""), 0.5)

    def test_load_transcript_ef2a(self) -> None:
        m = self.mod
        if not Path(EF2A8B4D).is_file():
            self.skipTest(f"ef2a8b4d jsonl 不存在: {EF2A8B4D}")
        events = m.load_transcript(EF2A8B4D)
        self.assertEqual(len(events), 36)

    def test_analyze_transcript_ef2a(self) -> None:
        m = self.mod
        if not Path(EF2A8B4D).is_file():
            self.skipTest(f"ef2a8b4d jsonl 不存在: {EF2A8B4D}")
        r = m.analyze_transcript(EF2A8B4D)
        self.assertEqual(r["total_events"], 36)
        self.assertGreaterEqual(r["assistant_turn_count"], 1)
        self.assertIn("user", r["role_distribution"])
        self.assertIn("assistant", r["role_distribution"])
        # char_distribution
        cd = r["char_distribution"]
        self.assertIn("turn_indices", cd)
        self.assertIn("char_counts", cd)
        self.assertIn("token_estimates", cd)
        self.assertIn("histogram", cd)
        self.assertEqual(len(cd["turn_indices"]), len(cd["char_counts"]))
        self.assertEqual(len(cd["histogram"]["edges"]), 11)  # 10 bins + 1
        # implicit rewards
        ir = r["implicit_rewards"]
        self.assertIn("user_turn_indices", ir)
        self.assertIn("rewards", ir)
        self.assertIn("mean_reward", ir)
        self.assertGreaterEqual(ir["mean_reward"], 0.0)
        self.assertLessEqual(ir["mean_reward"], 1.0)
        # top longest
        self.assertGreaterEqual(len(r["top_longest_turns"]), 1)
        self.assertLessEqual(len(r["top_longest_turns"]), 5)
        # JSON serializable
        json.dumps(r, ensure_ascii=False)

    def test_analyze_transcript_40b0(self) -> None:
        m = self.mod
        if not Path(F40B0D2F3).is_file():
            self.skipTest(f"40b0d2f3 jsonl 不存在: {F40B0D2F3}")
        r = m.analyze_transcript(F40B0D2F3)
        self.assertEqual(r["total_events"], 173)
        self.assertGreaterEqual(r["user_turn_count"], 1)
        self.assertGreaterEqual(r["tool_call_count"], 1)
        self.assertGreaterEqual(r["implicit_rewards"]["mean_reward"], 0.0)
        self.assertLessEqual(r["implicit_rewards"]["mean_reward"], 1.0)
        json.dumps(r, ensure_ascii=False)

    def test_batch_analyze(self) -> None:
        m = self.mod
        paths = [p for p in (EF2A8B4D, F40B0D2F3) if Path(p).is_file()]
        if not paths:
            self.skipTest("无可用 jsonl 文件")
        results = m.batch_analyze(paths)
        self.assertEqual(len(results), len(paths))
        for r in results:
            self.assertIn("path", r)
            self.assertTrue("error" in r or "total_events" in r)


class TestAgentTranscriptController(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from main import app
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"无法 import app: {exc}")
        self.client = TestClient(app)

    def test_page_endpoint(self) -> None:
        r = self.client.get("/api/agent-transcripts/page")
        self.assertEqual(r.status_code, 200)
        self.assertIn("<table", r.text)
        self.assertIn("<svg", r.text)

    def test_report_endpoint(self) -> None:
        if not Path(EF2A8B4D).is_file():
            self.skipTest(f"ef2a8b4d jsonl 不存在")
        r = self.client.get("/api/agent-transcripts/report", params={"path": EF2A8B4D})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["code"], 0)
        self.assertIn("data", body)
        self.assertEqual(body["data"]["total_events"], 36)

    def test_batch_endpoint(self) -> None:
        r = self.client.get("/api/agent-transcripts/batch")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["code"], 0)
        self.assertGreaterEqual(body["data"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
