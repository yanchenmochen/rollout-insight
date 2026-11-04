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


def test_create_task_form_renders(authed_page):
    authed_page.goto("/api/tasks/create")
    expect(authed_page.locator("#task_name")).to_be_visible()
    expect(authed_page.locator("#rollout_root")).to_be_visible()
    expect(authed_page.locator("#submitBtn")).to_be_visible()
