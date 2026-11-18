#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""空 response 的 legacy 标记不能被当成一轮对话。"""
import unittest

from controller.segment_controller import _is_placeholder_turn_body, _wiped_sequence_note
from service.segment_transcript_service import _split_prompt_response
from utils.rollout_pt_parser import _segment_index_matches


class PlaceholderTranscriptTest(unittest.TestCase):
    def test_empty_response_is_not_a_turn_body(self):
        md = (
            "----- DATASET PROMPT (full token sequence unavailable) -----\n"
            "# Title\nkeep this prompt\n"
            "----- LEGACY TRANSCRIPT (response only) -----"
        )
        prompt, response = _split_prompt_response(md)
        self.assertIn("# Title", prompt)
        self.assertNotIn("full token sequence unavailable", prompt)
        self.assertEqual(response, "")
        self.assertTrue(_is_placeholder_turn_body("(response only) -----"))

    def test_real_response_after_marker_is_kept(self):
        md = (
            "----- LEGACY TRANSCRIPT (response only) -----\n"
            "<|im_start|>assistant\nhello<|im_end|>"
        )
        _prompt, response = _split_prompt_response(md)
        self.assertIn("hello", response)
        self.assertFalse(_is_placeholder_turn_body(response))

    def test_abort_sample_without_segment_index_still_matches(self):
        # 扫描把 sample 下标 85 写进 segment 表，metadata 里没有 segment_index
        self.assertTrue(_segment_index_matches(None, 85, 85))
        self.assertTrue(_segment_index_matches(0, 0, 768))
        self.assertFalse(_segment_index_matches(0, 1, 768))
        self.assertFalse(_segment_index_matches(None, 0, 85))

    def test_abort_stub_explains_missing_turns(self):
        note = _wiped_sequence_note(
            {
                "status": "aborted",
                "response": "",
                "tokens": [0, 0],
                "metadata": {
                    "abort_reason": "exception:SandboxApiException",
                    "attempt_completed_turns": 94,
                    "attempt_generated_response_tokens": 21693,
                },
            }
        )
        self.assertIn("SandboxApiException", note)
        self.assertIn("94", note)
        self.assertIsNone(
            _wiped_sequence_note(
                {
                    "status": "completed",
                    "response": "<|im_start|>assistant\nhi",
                    "tokens": [1, 2, 3],
                    "metadata": {},
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
