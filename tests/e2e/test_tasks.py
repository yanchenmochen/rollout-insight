"""模块：任务列表 / 详情 / 创建页。"""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_task_list_has_rows_or_empty(authed_page):
    authed_page.goto("/")
    expect(authed_page.locator("#taskTableBody")).to_be_visible()
    expect(authed_page.locator("#filterForm")).to_be_visible()


def test_open_first_task_detail(authed_page):
    authed_page.goto("/")
    link = authed_page.locator("#taskTableBody a").first
    if link.count() == 0:
        pytest.skip("当前没有任务，跳过详情")
    link.click()
    expect(authed_page.locator("#overview, h1").first).to_be_visible()
    expect(authed_page.locator("#rollouts")).to_be_visible()


def test_task_detail_annotations_progressive(authed_page):
    """SSR 先出结局柱；类型柱经 annotations.json 渐进填充。"""
    authed_page.goto("/")
    link = authed_page.locator("#taskTableBody a").first
    if link.count() == 0:
        pytest.skip("当前没有任务，跳过详情")
    href = link.get_attribute("href") or ""
    if not href:
        pytest.skip("任务链接无效")
    with authed_page.expect_response(
        lambda r: "/annotations/meta.json" in r.url and r.status == 200,
        timeout=60_000,
    ):
        authed_page.goto(href)
    expect(authed_page.locator("#rollout-stack")).to_be_visible()
    # 结局柱应已在 SSR HTML 中（不依赖 .pt）
    expect(authed_page.locator(".stack-bar.is-outcome").first).to_be_visible()
    rollout_rows = authed_page.locator(".stack-col[data-rollout-pk]")
    if rollout_rows.count() == 0:
        pytest.skip("任务下没有 rollout，无渐进注解目标")
    # 至少一条 per-rollout annotations
    authed_page.wait_for_response(
        lambda r: "/annotations.json" in r.url and "meta.json" not in r.url,
        timeout=120_000,
    )
    train_cell = authed_page.locator("[data-train-cell]").first
    expect(train_cell).to_be_visible()


def test_create_task_form_renders(authed_page):
    authed_page.goto("/api/tasks/create")
    expect(authed_page.locator("#task_name")).to_be_visible()
    expect(authed_page.locator("#rollout_root")).to_be_visible()
    expect(authed_page.locator("#submitBtn")).to_be_visible()
