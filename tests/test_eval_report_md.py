"""``evaluations.md`` 解析：外部评测台账。"""
from __future__ import annotations

import unittest

from utils.eval_report_md import (
    EVALUATION_FILE_NAME,
    extract_scores,
    parse_evaluations_md,
    read_evaluations_for_rollout_root,
    read_evaluations_md,
)

SIMPLE = """# Evaluations of training run 20261118_174234_c021-v06-band64-finish-train

Training job: c021-v06-band64-finish-train-sqh-20261118-174234 (jb-aitrain-165060858749676864). Updated 2026-11-19 15:23:25.

| Training step | Evaluation | Round | Nanhu job name | Nanhu job ID | Run | Job status | Result |
|---|---|---|---|---|---|---|---|
| 10 | panel | first run | c021-v06-band64-panel-s10-20261119-152117 | jb-aitrain-165101654806916608 | 20261119_152117_c021-v06-band64-panel-s10 | scheduling |  |
"""

RICH = """# Evaluations of training run 20261113_174212_c021-v06-rand128-train

Training job: c021-v06-rand128-train-sqh-20261117-171712 (jb-aitrain-165014717581520128). Updated 2026-11-19 16:15:32.

| Training step | Evaluation | Round | Nanhu job name | Nanhu job ID | Run | Job status | Result |
|---|---|---|---|---|---|---|---|
| 10 | panel | first run | c021-v06-rand128-panel-s10-20261114-071115 | jb-aitrain-164859746868816832 | 20261114_071115_c021-v06-rand128-panel-s10 | finished | ungraded 1 ['scikit-learn__scikit-learn-12585']; weighted score on 429 tasks: 0.660; 95% selection interval +-0.039; holdout score: 0.400 over 85 of 87 tasks |
| 20 | panel | first run | c021-v06-rand128-panel-s20-20261114-210305 | jb-aitrain-164885914293660608 | 20261114_210305_c021-v06-rand128-panel-s20 | finished | ungraded 1 ['psf__requests-6028']; weighted score on 429 tasks: 0.599; 95% selection interval +-0.050; holdout score: 0.427 over 85 of 87 tasks |
| 30 | panel | failed start | c021-v06-rand128-panel-s30-20261115-102507 | jb-aitrain-164911144309910464 | 20261115_102507_c021-v06-rand128-panel-s30 | error | resubmitted |
| 50 | v500 | rerun 2 | c021-v06-rand128-v500-s50-r2-20261117-013215 | jb-aitrain-164984977929305728 | 20261117_013215_c021-v06-rand128-v500-s50-r2 | finished | first round 288/500; merged 298/500 = 0.596; 2 reruns; still bad {'error': 2} |
| 60 | panel | first run | c021-v06-rand128-panel-s60-20261119-152112 | jb-aitrain-165101652927344128 | 20261119_152112_c021-v06-rand128-panel-s60 | running |  |
"""


class TestExtractScores(unittest.TestCase):
    def test_weighted_and_holdout(self):
        scores = extract_scores(
            "ungraded 1 ['a']; weighted score on 429 tasks: 0.660; "
            "95% selection interval +-0.039; holdout score: 0.400 over 85 of 87 tasks"
        )
        self.assertAlmostEqual(scores["weighted"], 0.660)
        self.assertEqual(scores["weighted_tasks"], 429)
        self.assertAlmostEqual(scores["interval"], 0.039)
        self.assertAlmostEqual(scores["holdout"], 0.400)
        self.assertEqual(scores["holdout_tasks"], 85)
        self.assertEqual(scores["holdout_tasks_total"], 87)
        self.assertEqual(scores["ungraded"], 1)

    def test_merged_form(self):
        scores = extract_scores(
            "first round 288/500; merged 298/500 = 0.596; 2 reruns; still bad {'error': 2}"
        )
        self.assertEqual(scores["first_round_pass"], 288)
        self.assertEqual(scores["first_round_total"], 500)
        self.assertEqual(scores["merged_pass"], 298)
        self.assertEqual(scores["merged_total"], 500)
        self.assertAlmostEqual(scores["merged"], 0.596)

    def test_empty_result(self):
        scores = extract_scores("")
        self.assertIsNone(scores["weighted"])
        self.assertIsNone(scores["holdout"])
        self.assertIsNone(scores["merged"])


class TestParseEvaluationsMd(unittest.TestCase):
    def test_empty_text(self):
        data = parse_evaluations_md("")
        self.assertFalse(data["exists"])
        self.assertEqual(data["runs"], [])

    def test_header_not_found(self):
        data = parse_evaluations_md("# 标题\n\n没有表格。\n")
        self.assertTrue(data["exists"])
        self.assertEqual(data["runs"], [])

    def test_simple_table(self):
        data = parse_evaluations_md(SIMPLE)
        self.assertTrue(data["exists"])
        self.assertEqual(data["job_name"], "c021-v06-band64-finish-train-sqh-20261118-174234")
        self.assertEqual(data["job_id"], "jb-aitrain-165060858749676864")
        self.assertEqual(data["updated_at"], "2026-11-19 15:23:25")
        self.assertEqual(len(data["runs"]), 1)
        run = data["runs"][0]
        self.assertEqual(run["step"], 10)
        self.assertEqual(run["evaluation"], "panel")
        self.assertEqual(run["round"], "first run")
        self.assertEqual(run["job_status"], "scheduling")
        self.assertIsNone(run["scores"]["weighted"])

    def test_rich_table(self):
        data = parse_evaluations_md(RICH)
        self.assertEqual(len(data["runs"]), 5)
        # 按 step 排序
        self.assertEqual([r["step"] for r in data["runs"]], [10, 20, 30, 50, 60])
        first = data["runs"][0]
        self.assertAlmostEqual(first["scores"]["weighted"], 0.660)
        self.assertEqual(first["job_status"], "finished")
        # failed start 归一化为 error
        self.assertEqual(data["runs"][2]["job_status"], "error")
        # merged 形态
        v500 = [r for r in data["runs"] if r["evaluation"] == "v500"][0]
        self.assertAlmostEqual(v500["scores"]["merged"], 0.596)
        # running
        self.assertEqual(data["runs"][4]["job_status"], "running")

    def test_two_evaluations_in_one_step(self):
        """同一 step 下的两个评测（panel / v500）都要保留。"""
        data = parse_evaluations_md(RICH)
        step50 = [r for r in data["runs"] if r["step"] == 50]
        self.assertEqual(len(step50), 1)
        self.assertEqual(step50[0]["evaluation"], "v500")


class TestReadEvaluationsMd(unittest.TestCase):
    def test_missing_file(self):
        data = read_evaluations_md("/tmp/definitely-not-exists-evaluations.md")
        self.assertFalse(data["exists"])

    def test_rollout_root_without_run_dir(self):
        data = read_evaluations_for_rollout_root("")
        self.assertFalse(data["exists"])
        self.assertEqual(data["path"], "")

    def test_rollout_root_resolves_file(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "run-1"
            (run_dir / "tbs").mkdir(parents=True)
            (run_dir / EVALUATION_FILE_NAME).write_text(SIMPLE, encoding="utf-8")
            rollout_root = str(run_dir / "generated" / "rollout_dumps")
            data = read_evaluations_for_rollout_root(rollout_root)
            self.assertTrue(data["exists"])
            self.assertTrue(data["path"].endswith(EVALUATION_FILE_NAME))
            self.assertEqual(len(data["runs"]), 1)


if __name__ == "__main__":
    unittest.main()
