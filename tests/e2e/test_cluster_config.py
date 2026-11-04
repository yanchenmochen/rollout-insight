"""模块：当前用户的集群网关配置。"""
from __future__ import annotations

import time

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_cluster_config_create_and_delete(authed_page):
    key = f"e2e-{int(time.time())}"
    authed_page.goto("/api/me/cluster-configs/new")
    expect(authed_page.locator("#cluster_key")).to_be_visible()
    authed_page.locator("#cluster_key").fill(key)
    authed_page.locator("#display_label").fill("e2e 临时集群")
    authed_page.locator("#api_base_url").fill("http://127.0.0.1:9")
    authed_page.locator("#user_id_field").fill("e2e-user")
    authed_page.get_by_role("button", name="保存").click()
    expect(authed_page.get_by_text(key)).to_be_visible()

    row = authed_page.locator("tr", has_text=key)
    authed_page.once("dialog", lambda d: d.accept())
    row.get_by_role("button", name="删除").click()
    expect(authed_page.get_by_text(key)).to_have_count(0)
