"""模块：segment 展开后懒加载 turns。"""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_segment_lazy_loads_turns(authed_page):
    authed_page.goto("/")
    task = authed_page.locator("#taskTableBody a").first
    if task.count() == 0:
        pytest.skip("没有任务")
    task.click()
    rollout = authed_page.locator("#rollouts tbody a").first
    if rollout.count() == 0:
        pytest.skip("没有 rollout")
    rollout.click()

    group = authed_page.locator("#treeRoot > details.group-row").first
    if group.get_attribute("open") is None:
        group.locator("summary").first.click()
    attempt = group.locator("details.attempt-row").first
    if attempt.count() == 0:
        pytest.skip("group 下没有 attempt")
    attempt.locator("summary").first.click()
    segment = attempt.locator("details.segment-row").first
    if segment.count() == 0:
        pytest.skip("attempt 下没有 segment")
    with authed_page.expect_response(lambda r: "/turns.json" in r.url and r.status == 200):
        segment.locator("summary").first.click()
    body = segment.locator(".tree-body").last
    expect(body).to_be_visible()
    # 有 turn，或明确的空提示
    assert (
        segment.locator("details.turn-row").count() >= 1
        or "暂无" in (body.inner_text() or "")
        or "turn" in (body.inner_text() or "").lower()
    )
