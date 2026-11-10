"""task_annotation_service：sidecar 命中/失效与单 rollout 载荷。"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from service import task_annotation_service as tas


def _tiny_dump():
    return {
        "samples": [
            {
                "group_index": 0,
                "rollout_id": 0,
                "status": "completed",
                "metadata": {"raw_reward": 1.0, "instance_id": "case-a"},
                "reward": 1.0,
                "prompt": "hello",
            },
            {
                "group_index": 0,
                "rollout_id": 1,
                "status": "completed",
                "metadata": {
                    "raw_reward": 0.0,
                    "diagnosis": "evaluator:tests_failed",
                },
                "reward": 0.0,
            },
            {
                "group_index": 1,
                "rollout_id": 2,
                "status": "aborted",
                "metadata": {"abort_reason": "exception:SandboxInternalException"},
                "reward": 0.0,
            },
            {
                "group_index": 1,
                "rollout_id": 3,
                "status": "completed",
                "metadata": {"raw_reward": 0.0, "train_drop_reason": "low_quality"},
                "reward": 0.0,
            },
        ]
    }


class TestTaskAnnotationSidecar(unittest.TestCase):
    def setUp(self) -> None:
        tas.clear_annotation_cache_for_tests()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.pt_path = str(Path(self._tmpdir.name) / "rollout_0.pt")
        Path(self.pt_path).write_bytes(b"fake-pt")

    def tearDown(self) -> None:
        tas.clear_annotation_cache_for_tests()
        self._tmpdir.cleanup()

    def test_load_writes_and_hits_sidecar(self):
        dump = _tiny_dump()
        with mock.patch("utils.rollout_pt_parser.load_pt", return_value=dump):
            # first load computes + writes sidecar
            first = tas.load_pt_annotations(self.pt_path)
            self.assertGreater(first["type_breakdown"]["n_attempts"], 0)
            sidecar = tas.sidecar_path_for(self.pt_path)
            self.assertTrue(os.path.isfile(sidecar))

        tas.clear_annotation_cache_for_tests()
        with mock.patch(
            "utils.rollout_pt_parser.load_pt",
            side_effect=AssertionError("should not load pt when sidecar hits"),
        ):
            second = tas.load_pt_annotations(self.pt_path)
        self.assertEqual(
            second["type_breakdown"]["n_attempts"],
            first["type_breakdown"]["n_attempts"],
        )
        self.assertEqual(
            second["train_sample_stats"]["n_train_kept"],
            first["train_sample_stats"]["n_train_kept"],
        )

    def test_sidecar_invalidated_on_mtime_change(self):
        dump = _tiny_dump()
        with mock.patch("utils.rollout_pt_parser.load_pt", return_value=dump):
            tas.load_pt_annotations(self.pt_path)
        sidecar = tas.sidecar_path_for(self.pt_path)
        with open(sidecar, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["pt_mtime"] = 1.0
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

        tas.clear_annotation_cache_for_tests()
        called = {"n": 0}

        def _load(_path):
            called["n"] += 1
            return dump

        with mock.patch("utils.rollout_pt_parser.load_pt", side_effect=_load):
            tas.load_pt_annotations(self.pt_path)
        self.assertEqual(called["n"], 1)

    def test_empty_path(self):
        out = tas.load_pt_annotations(None)
        self.assertEqual(out["type_breakdown"]["n_attempts"], 0)
        self.assertEqual(out["train_sample_stats"]["n_samples"], 0)

    def test_type_segments_payload_shape(self):
        dump = _tiny_dump()
        with mock.patch("utils.rollout_pt_parser.load_pt", return_value=dump):
            ann = tas.load_pt_annotations(self.pt_path)
        segs = tas._type_segments_from_ann(ann)
        self.assertTrue(any(s["key"] == "pass" for s in segs))
        self.assertTrue(any(str(s["key"]).startswith("fail:") for s in segs))
        summary = tas._failure_summary(ann)
        self.assertIn("by_reason", summary)
        self.assertGreaterEqual(summary["n_abort"], 1)


class TestAssignStackTypeColors(unittest.TestCase):
    def test_pass_and_fail_colors(self):
        from collections import Counter

        colors = tas.assign_stack_type_colors(
            Counter({"pass": 3, "fail:x": 2, "abort:y": 1})
        )
        self.assertEqual(colors["pass"]["color"], "#2e7d32")
        self.assertIn("失败", colors["fail:x"]["label"])
        self.assertIn("abort", colors["abort:y"]["label"])


if __name__ == "__main__":
    unittest.main()
