"""模块：任务详情里的 rollout 行 → 四层手风琴。"""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def _open_first_rollout(page):
    page.goto("/")
    task = page.locator("#taskTableBody a").first
    if task.count() == 0:
        pytest.skip("没有任务")
    task.click()
    rollout = page.locator("#rollouts tbody a").first
    if rollout.count() == 0:
        pytest.skip("任务下没有 rollout")
    rollout.click()


def test_rollout_page_shows_groups(authed_page):
    _open_first_rollout(authed_page)
    expect(authed_page.locator("#treeRoot")).to_be_visible()
    groups = authed_page.locator("#treeRoot > details.group-row")
    assert groups.count() >= 1


def test_expand_attempt_shows_stats(authed_page):
    _open_first_rollout(authed_page)
    group = authed_page.locator("#treeRoot > details.group-row").first
    if group.get_attribute("open") is None:
        group.locator("summary").first.click()
    attempt = group.locator("details.attempt-row").first
    expect(attempt).to_be_visible()
    attempt.locator("summary").first.click()
    chips = attempt.locator(".stat-chip")
    expect(chips.filter(has_text="tool")).to_be_visible()
    expect(chips.filter(has_text="diff")).to_be_visible()
    expect(chips.filter(has_text="resp_len")).to_be_visible()
    expect(chips.filter(has_text="mask")).to_be_visible()
