#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""segment 页：塌缩成单条 ASSISTANT 的 markdown turn 应被重切成多轮。"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestExplodeMarkdownTurns(unittest.TestCase):
    def test_explode_single_embedded_markdown(self):
        from controller.segment_controller import _explode_markdown_turns

        body = (
            "### turn 1 · assistant\nhello think\n"
            "### turn 2 · user\ntool result\n"
            "### turn 3 · assistant\nnext\n"
        )
        turns = [
            {
                "turn_index": 1,
                "role": "assistant",
                "role_label": "assistant",
                "body_text": body,
                "body": body,
            }
        ]
        out, source = _explode_markdown_turns(turns, "turn_table")
        self.assertGreaterEqual(len(out), 3)
        self.assertIn("markdown_turns", source)
        roles = [t.get("role_label") or t.get("role") for t in out]
        self.assertEqual(roles[:3], ["assistant", "user", "assistant"])


class TestRepairCollapsedMarkdownTurns(unittest.TestCase):
    def test_hydrate_from_turns_json_when_still_one(self):
        from controller.segment_controller import _repair_collapsed_markdown_turns

        collapsed = [
            {
                "turn_index": 1,
                "role": "assistant",
                "role_label": "assistant",
                "body_text": "only one assistant blob without headers",
                "body": "only one assistant blob without headers",
                "body_char_count": 40,
            }
        ]
        fake_row = MagicMock()
        fake_row.turns_json = (
            '[{"role":"assistant","content":"a"},'
            '{"role":"user","content":"b"},'
            '{"role":"assistant","content":"c"}]'
        )

        with patch(
            "controller.segment_controller._recover_truncated_markdown",
            return_value=(collapsed, "turn_table"),
        ), patch(
            "controller.segment_controller._explode_markdown_turns",
            return_value=(collapsed, "turn_table"),
        ), patch(
            "db.db_segment_transcript.DBSegmentTranscript",
            create=True,
        ), patch("db.db_element.Session") as session_cls, patch(
            "db.db_segment_turn_service.DBSegmentTurnService.upsert_segment_turns",
            return_value=3,
        ):
            session = MagicMock()
            session_cls.return_value.__enter__.return_value = session
            session.query.return_value.filter.return_value.first.return_value = fake_row
            out = _repair_collapsed_markdown_turns(42, collapsed)

        self.assertEqual(len(out), 3)
        self.assertEqual(
            [t["role_label"] for t in out],
            ["assistant", "user", "assistant"],
        )
        self.assertTrue(all("html" in t for t in out))


if __name__ == "__main__":
    unittest.main()
