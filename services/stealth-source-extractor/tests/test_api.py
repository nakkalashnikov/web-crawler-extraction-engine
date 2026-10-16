"""Integration tests for FastAPI endpoints — no real network calls."""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import src.main as main_module
from src.models import SourceExtractResult, VideoEmbed


@pytest.fixture(scope="module")
def client():
    with TestClient(main_module.app) as c:
        yield c


class TestHealth:
    def test_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestExtractEndpoint:
    def test_social_direct_happy_path(self, client):
        fake = SourceExtractResult(
            url="https://youtu.be/abc123", kind="social_direct",
            platform="youtube", canonical_url="https://youtu.be/abc123",
        )
        with patch.object(main_module, "extract", new=AsyncMock(return_value=fake)):
            r = client.post("/extract", json={"url": "https://youtu.be/abc123"})
        assert r.status_code == 200
        body = r.json()
        assert body["kind"] == "social_direct"
        assert body["platform"] == "youtube"
        assert body["canonical_url"] == "https://youtu.be/abc123"

    def test_article_happy_path(self, client):
        fake = SourceExtractResult(
            url="https://example.com/a", kind="article", source="live",
            title="A title", content="hello", images=["https://example.com/photo.jpg"],
            video_embeds=[VideoEmbed(url="https://t.me/x/1", platform="telegram")],
        )
        with patch.object(main_module, "extract", new=AsyncMock(return_value=fake)):
            r = client.post("/extract", json={"url": "https://example.com/a"})
        assert r.status_code == 200
        body = r.json()
        assert body["kind"] == "article"
        assert body["source"] == "live"
        assert body["content"] == "hello"
        assert body["video_embeds"] == [{"url": "https://t.me/x/1", "platform": "telegram"}]

    def test_failure_result_still_returns_200_with_kind_failure(self, client):
        fake = SourceExtractResult(url="https://example.com/a", kind="failure", reason="upstream_unavailable")
        with patch.object(main_module, "extract", new=AsyncMock(return_value=fake)):
            r = client.post("/extract", json={"url": "https://example.com/a"})
        assert r.status_code == 200
        assert r.json()["kind"] == "failure"
        assert r.json()["reason"] == "upstream_unavailable"

    def test_url_passed_through_to_extract(self, client):
        fake = SourceExtractResult(url="https://example.com/a", kind="article")
        with patch.object(main_module, "extract", new=AsyncMock(return_value=fake)) as mock_extract:
            client.post("/extract", json={"url": "https://example.com/a"})
        assert mock_extract.call_args[0][3] == "https://example.com/a"
