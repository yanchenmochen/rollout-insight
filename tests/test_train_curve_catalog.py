"""训练曲线指标目录：默认按钮组、tag 匹配、自定义指标、偏好合并。"""
from __future__ import annotations

import unittest
from unittest import mock

from service.train_curve_service import resolve_curve_keys, save_curve_preference
from utils.train_curve_catalog import (
    CURVE_METRICS,
    DEFAULT_CURVE_KEYS,
    available_curve_metrics,
    curve_metric,
    curve_metric_keys,
    is_custom_key,
    match_tag,
    normalize_curve_keys,
    resolve_metric_tags,
)


class TestCurveCatalog(unittest.TestCase):
    def test_default_keys_cover_requested_buttons(self):
        # 用户要求的默认按钮组：混合奖励 / 长度 / 梯度 / entropy_loss / context_exhausted
        self.assertEqual(
            DEFAULT_CURVE_KEYS,
            [
                "mixed_reward",
                "response_length",
                "grad_norm",
                "entropy_loss",
                "context_exhausted",
            ],
        )

    def test_labels(self):
        self.assertEqual(curve_metric("mixed_reward")["label"], "混合奖励")
        self.assertEqual(curve_metric("response_length")["label"], "长度")
        self.assertEqual(curve_metric("grad_norm")["label"], "梯度")

    def test_unknown_metric_returns_none(self):
        self.assertIsNone(curve_metric("nope"))

    def test_curve_metric_keys_matches_registry(self):
        self.assertEqual(set(curve_metric_keys()), set(CURVE_METRICS.keys()))

    def test_available_metrics_marks_defaults(self):
        items = available_curve_metrics()
        defaults = [item["key"] for item in items if item["is_default"]]
        self.assertEqual(defaults, DEFAULT_CURVE_KEYS)


class TestMatchTag(unittest.TestCase):
    def test_matches_prefixed_tags(self):
        self.assertTrue(match_tag("mixed_reward", "rollout/mixed_reward"))
        self.assertTrue(match_tag("response_length", "actor/response_length"))
        self.assertTrue(match_tag("grad_norm", "optimizer/grad_norm"))
        self.assertTrue(match_tag("entropy_loss", "actor/entropy_loss"))
        self.assertTrue(match_tag("context_exhausted", "rollout/context_exhausted"))

    def test_non_matching_tag(self):
        self.assertFalse(match_tag("mixed_reward", "actor/loss"))
        self.assertFalse(match_tag("grad_norm", ""))

    def test_custom_key_requires_exact_tag(self):
        self.assertTrue(is_custom_key("custom:my_metric"))
        self.assertFalse(is_custom_key("mixed_reward"))
        self.assertTrue(match_tag("custom:my_metric", "my_metric"))
        self.assertFalse(match_tag("custom:my_metric", "prefix/my_metric"))


class TestResolveMetricTags(unittest.TestCase):
    def test_prefers_configured_tag_order(self):
        tags = ["critic/mixed_reward", "rollout/mixed_reward", "other/mixed_reward"]
        resolved = resolve_metric_tags("mixed_reward", tags)
        self.assertEqual(resolved[0], "rollout/mixed_reward")

    def test_returns_empty_when_no_match(self):
        self.assertEqual(resolve_metric_tags("grad_norm", ["actor/loss"]), [])

    def test_custom_key_exact_match(self):
        self.assertEqual(
            resolve_metric_tags("custom:foo", ["bar", "foo"]), ["foo"]
        )

    def test_unknown_key_returns_empty(self):
        self.assertEqual(resolve_metric_tags("nope", ["a", "b"]), [])


class TestNormalizeCurveKeys(unittest.TestCase):
    def test_filters_unknown_and_dedupes(self):
        self.assertEqual(
            normalize_curve_keys(["mixed_reward", "mixed_reward", "nope", "  "]),
            ["mixed_reward"],
        )

    def test_keeps_custom_keys(self):
        self.assertEqual(
            normalize_curve_keys(["custom:a/b", "custom:"]),
            ["custom:a/b"],
        )

    def test_none_returns_empty(self):
        self.assertEqual(normalize_curve_keys(None), [])


class TestResolveCurveKeys(unittest.TestCase):
    """偏好合并：内置默认 → 用户全局 → 任务级追加。"""

    def _patch_pref(self, global_row, task_row):
        def fake_get_global(user_id):
            return global_row

        def fake_get_for_task(user_id, task_id):
            return task_row

        return (
            mock.patch(
                "db.db_train_curve_preference_service.DBTrainCurvePreferenceService.get_global",
                fake_get_global,
            ),
            mock.patch(
                "db.db_train_curve_preference_service.DBTrainCurvePreferenceService.get_for_task",
                fake_get_for_task,
            ),
        )

    def _row(self, keys, custom_keys):
        class _Row:
            def to_dict(self):
                return {"keys": list(keys), "custom_keys": list(custom_keys)}

        return _Row()

    def test_default_when_no_preference(self):
        p1, p2 = self._patch_pref(None, None)
        with p1, p2:
            result = resolve_curve_keys(1, 10)
        self.assertEqual(result["keys"], DEFAULT_CURVE_KEYS)
        self.assertFalse(result["global_saved"])
        self.assertFalse(result["has_task_override"])

    def test_global_replaces_defaults(self):
        p1, p2 = self._patch_pref(self._row(["grad_norm"], []), None)
        with p1, p2:
            result = resolve_curve_keys(1, 10)
        self.assertEqual(result["keys"], ["grad_norm"])
        self.assertTrue(result["global_saved"])

    def test_task_keys_are_appended(self):
        p1, p2 = self._patch_pref(
            self._row(["grad_norm"], ["custom:my_metric"]), self._row(["entropy_loss"], [])
        )
        with p1, p2:
            result = resolve_curve_keys(1, 10)
        self.assertEqual(result["keys"], ["grad_norm", "custom:my_metric", "entropy_loss"])
        self.assertTrue(result["has_task_override"])

    def test_anonymous_user_gets_defaults(self):
        p1, p2 = self._patch_pref(self._row(["grad_norm"], []), None)
        with p1, p2:
            result = resolve_curve_keys(None, 10)
        self.assertEqual(result["keys"], DEFAULT_CURVE_KEYS)


class TestSaveCurvePreference(unittest.TestCase):
    def test_requires_user_id(self):
        from data.rollout_insight_exception import RolloutInsightException

        with self.assertRaises(RolloutInsightException):
            save_curve_preference(1, None, keys=["grad_norm"])

    def test_rejects_invalid_scope(self):
        from data.rollout_insight_exception import RolloutInsightException

        with self.assertRaises(RolloutInsightException):
            save_curve_preference(1, 1, keys=["grad_norm"], scope="bogus")

    def test_global_scope_calls_upsert_global(self):
        with mock.patch(
            "db.db_train_curve_preference_service.DBTrainCurvePreferenceService.upsert_global"
        ) as upsert, mock.patch(
            "db.db_train_curve_preference_service.DBTrainCurvePreferenceService.get_global",
            return_value=None,
        ), mock.patch(
            "db.db_train_curve_preference_service.DBTrainCurvePreferenceService.get_for_task",
            return_value=None,
        ):
            save_curve_preference(1, 7, keys=["grad_norm", "nope"], scope="global")
        upsert.assert_called_once()
        args, kwargs = upsert.call_args
        self.assertEqual(args[0], 7)
        self.assertEqual(kwargs["keys"], ["grad_norm"])


if __name__ == "__main__":
    unittest.main()
