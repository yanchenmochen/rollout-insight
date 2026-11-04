"""模块：agent transcript 报告页。"""
from __future__ import annotations

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_agent_transcript_page_renders(authed_page):
    authed_page.goto("/api/agent-transcripts/page")
    expect(authed_page.locator("h1")).to_be_visible()
    body = authed_page.locator("body")
    text = body.inner_text()
    assert "transcript" in text.lower() or "token" in text.lower() or "报告" in text
