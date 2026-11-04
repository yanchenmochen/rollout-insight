"""模块：南湖作业页（只读渲染，不真正 restart）。"""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_nanhu_jobs_page_renders(authed_page):
    authed_page.goto("/api/nanhu/jobs/page")
    expect(authed_page.locator("#filterForm")).to_be_visible()
    expect(authed_page.locator("#jobTableBody, .empty-row").first).to_be_visible()
