"""Playwright 端到端夹具。

环境变量：
- ROLLOUT_INSIGHT_BASE  默认 http://127.0.0.1:39998
- ROLLOUT_INSIGHT_E2E_USER / ROLLOUT_INSIGHT_E2E_PASSWORD  默认 admin / admin123
"""
from __future__ import annotations

import os

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Page, expect  # noqa: E402

BASE_URL = os.environ.get("ROLLOUT_INSIGHT_BASE", "http://127.0.0.1:39998").rstrip("/")
USERNAME = os.environ.get("ROLLOUT_INSIGHT_E2E_USER", "admin")
PASSWORD = os.environ.get("ROLLOUT_INSIGHT_E2E_PASSWORD", "admin123")


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session")
def browser_type_launch_args():
    return {"headless": os.environ.get("ROLLOUT_INSIGHT_E2E_HEADED", "") != "1"}


@pytest.fixture(scope="session")
def browser_context_args():
    return {"base_url": BASE_URL, "ignore_https_errors": True}


def login(page: Page, username: str = USERNAME, password: str = PASSWORD) -> None:
    page.goto("/api/auth/login")
    page.locator("#username").fill(username)
    page.locator("#password").fill(password)
    page.get_by_role("button", name="登录").click()
    expect(page).to_have_url(lambda url: "/api/auth/login" not in url, timeout=15000)


@pytest.fixture
def anon_page(page: Page) -> Page:
    page.context.clear_cookies()
    return page


@pytest.fixture
def authed_page(page: Page) -> Page:
    login(page)
    return page
