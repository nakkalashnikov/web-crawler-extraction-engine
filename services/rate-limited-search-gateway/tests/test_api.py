"""Integration tests for FastAPI endpoints — no real network calls."""
import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import src.main as main_module
from src.config import settings
from src.models import ExtractResult, SearchResult
from src.search import TavilySearchFailedError


@pytest.fixture(scope="module")
def client():
    # The real lifespan runs on TestClient startup, which calls
    # build_key_pool() -> fetch_usage() -> a real GET /usage — mocked here
    # so test runs never depend on live network access.
    with respx.mock:
        respx.get("https://api.tavily.com/usage").mock(
            return_value=httpx.Response(200, json={"account": {"current_plan": "Researcher", "plan_limit": 1000, "plan_usage": 0}}),
        )
        with TestClient(main_module.app) as c:
            yield c


class TestHealth:
    def test_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestSearchEndpoint:
    def test_happy_path(self, client):
        fake_results = [SearchResult(
            url="https://militarnyi.com/uk/news/inguar-3", title="Inguar-3 news",
            content="text", score=0.68,
        )]
        with patch.object(main_module, "search", new=AsyncMock(return_value=fake_results)):
            r = client.post("/search", json={"query": "Inguar-3", "include_domains": ["militarnyi.com"]})
        assert r.status_code == 200
        body = r.json()
        assert body["results"][0]["url"] == "https://militarnyi.com/uk/news/inguar-3"
        assert body["results"][0]["score"] == 0.68

    def test_no_results_returns_empty_not_error(self, client):
        with patch.object(main_module, "search", new=AsyncMock(return_value=[])):
            r = client.post("/search", json={"query": "nonsense query xyz"})
        assert r.status_code == 200
        assert r.json()["results"] == []

    def test_default_topic_is_general(self, client):
        with patch.object(main_module, "search", new=AsyncMock(return_value=[])) as mock_search:
            client.post("/search", json={"query": "X"})
        assert mock_search.call_args[0][6] == "general"   # (..., max_results, search_depth, topic)

    def test_news_topic_is_forwarded_and_published_date_returned(self, client):
        fake = [SearchResult(url="https://militarnyi.com/en/news/a", title="A", content="t", score=0.7,
                             published_date="Sat, 05 Sep 2026 15:03:16 GMT")]
        with patch.object(main_module, "search", new=AsyncMock(return_value=fake)) as mock_search:
            r = client.post("/search", json={"query": "X", "topic": "news"})
        assert mock_search.call_args[0][6] == "news"
        assert r.json()["results"][0]["published_date"] == "Sat, 05 Sep 2026 15:03:16 GMT"

    def test_max_results_absent_is_forwarded_as_none(self, client):
        with patch.object(main_module, "search", new=AsyncMock(return_value=[])) as mock_search:
            client.post("/search", json={"query": "X"})
        assert mock_search.call_args[0][4] is None   # (http, key_pool, query, include_domains, max_results)

    def test_max_results_given_is_forwarded(self, client):
        with patch.object(main_module, "search", new=AsyncMock(return_value=[])) as mock_search:
            client.post("/search", json={"query": "X", "max_results": 20})
        assert mock_search.call_args[0][4] == 20

    def test_upstream_4xx_passes_through_status(self, client):
        err = httpx.HTTPStatusError(
            "boom", request=httpx.Request("POST", "https://api.tavily.com/search"),
            response=httpx.Response(401, text="invalid api key"),
        )
        with patch.object(main_module, "search", new=AsyncMock(side_effect=err)):
            r = client.post("/search", json={"query": "X"})
        assert r.status_code == 401

    def test_upstream_429_passes_through(self, client):
        err = httpx.HTTPStatusError(
            "boom", request=httpx.Request("POST", "https://api.tavily.com/search"),
            response=httpx.Response(429, text="quota exceeded"),
        )
        with patch.object(main_module, "search", new=AsyncMock(side_effect=err)):
            r = client.post("/search", json={"query": "X"})
        assert r.status_code == 429

    def test_network_error_returns_502(self, client):
        err = httpx.ConnectTimeout("timed out")
        with patch.object(main_module, "search", new=AsyncMock(side_effect=err)):
            r = client.post("/search", json={"query": "X"})
        assert r.status_code == 502

    def test_key_pool_exhausted_returns_502_with_real_error_body(self, client):
        real_error = {"detail": {"error": "This request exceeds your plan's set usage limit."}}
        err = TavilySearchFailedError(432, real_error)
        with patch.object(main_module, "search", new=AsyncMock(side_effect=err)):
            r = client.post("/search", json={"query": "X"})
        assert r.status_code == 502
        detail = r.json()["detail"]
        assert detail["status_code"] == 432
        assert detail["tavily_error"] == real_error


class TestExtractEndpoint:
    def test_happy_path(self, client):
        fake_results = [ExtractResult(
            url="https://militarnyi.com/uk/news/inguar-3", title="Inguar-3 news",
            raw_content="text", images=[],
        )]
        with patch.object(main_module, "extract", new=AsyncMock(return_value=(fake_results, []))):
            r = client.post("/extract", json={"url": "https://militarnyi.com/uk/news/inguar-3"})
        assert r.status_code == 200
        body = r.json()
        assert body["results"][0]["url"] == "https://militarnyi.com/uk/news/inguar-3"
        assert "score" not in body["results"][0]

    def test_upstream_4xx_passes_through_status(self, client):
        err = httpx.HTTPStatusError(
            "boom", request=httpx.Request("POST", "https://api.tavily.com/extract"),
            response=httpx.Response(401, text="invalid api key"),
        )
        with patch.object(main_module, "extract", new=AsyncMock(side_effect=err)):
            r = client.post("/extract", json={"url": "https://example.com"})
        assert r.status_code == 401

    def test_network_error_returns_502(self, client):
        err = httpx.ConnectTimeout("timed out")
        with patch.object(main_module, "extract", new=AsyncMock(side_effect=err)):
            r = client.post("/extract", json={"url": "https://example.com"})
        assert r.status_code == 502

    def test_key_pool_exhausted_returns_502_with_real_error_body(self, client):
        real_error = {"detail": {"error": "This request exceeds your plan's set usage limit."}}
        err = TavilySearchFailedError(432, real_error)
        with patch.object(main_module, "extract", new=AsyncMock(side_effect=err)):
            r = client.post("/extract", json={"url": "https://example.com"})
        assert r.status_code == 502
        detail = r.json()["detail"]
        assert detail["status_code"] == 432
        assert detail["tavily_error"] == real_error


class TestConcurrencyGate:
    """See .claude/specs/tavily-search-concurrency-gate.md — a real ceiling on
    concurrent Tavily calls, independent of any other service's concurrency."""

    def test_default_max_concurrent_is_10(self):
        """Lowered from 50 (2026-10-11): 50 in flight plus retries produced ~900 calls/min,
        all 429 (Tavily's development key limit is 100 RPM per key)."""
        assert settings.max_concurrent == 10

    def test_default_per_key_concurrency_is_1(self):
        assert settings.per_key_concurrency == 1

    async def test_semaphore_blocks_once_exhausted_then_admits_on_release(self, client):
        semaphore = main_module._semaphore
        assert semaphore is not None

        held = [await semaphore.acquire() for _ in range(settings.max_concurrent)]
        assert len(held) == settings.max_concurrent

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(semaphore.acquire(), timeout=0.05)

        semaphore.release()
        await asyncio.wait_for(semaphore.acquire(), timeout=0.05)  # now succeeds

        for _ in range(settings.max_concurrent):
            semaphore.release()
