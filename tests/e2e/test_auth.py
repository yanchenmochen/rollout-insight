"""模块：登录 / 会话。"""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_anonymous_home_redirects_to_login(anon_page):
    anon_page.goto("/")
    expect(anon_page).to_have_url(lambda u: "/api/auth/login" in u)
    expect(anon_page.locator("h1")).to_contain_text("登录")


def test_wrong_password_shows_banner(anon_page):
    anon_page.goto("/api/auth/login")
    anon_page.locator("#username").fill("admin")
    anon_page.locator("#password").fill("not-the-password")
    anon_page.get_by_role("button", name="登录").click()
    expect(anon_page.locator(".auth-banner")).to_be_visible()


def test_admin_login_reaches_task_list(anon_page):
    anon_page.goto("/api/auth/login")
    anon_page.locator("#username").fill("admin")
    anon_page.locator("#password").fill("admin123")
    anon_page.get_by_role("button", name="登录").click()
    expect(anon_page.locator("h1")).to_contain_text("任务")
    expect(anon_page.get_by_role("button", name="退出")).to_be_visible()


def test_logout_returns_to_login(authed_page):
    authed_page.get_by_role("button", name="退出").click()
    expect(authed_page).to_have_url(lambda u: "/api/auth/login" in u)


def test_me_json_after_login(authed_page):
    resp = authed_page.request.get("/api/auth/me")
    assert resp.ok
    body = resp.json()
    data = body.get("data") or {}
    assert data.get("username") == "admin"
