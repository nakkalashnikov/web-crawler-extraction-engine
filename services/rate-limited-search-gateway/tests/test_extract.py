import asyncio
import json
from unittest.mock import patch

import httpx
import pytest
import respx

from src.config import settings
from src.extract import extract
from src.keys import KeyState, TavilyKeyPool
from src.search import TavilySearchFailedError

_OK_RESPONSE = {
    "results": [
        {
            "url":         "https://militarnyi.com/uk/news/inguar-3",
            "title":       "Inguar-3 news",
            "raw_content": "full article text",
            "images":      ["https://militarnyi.com/img1.jpg"],
        },
    ],
    "failed_results": [],
}


def _key_state(key: str, tier: str = "free") -> KeyState:
    return KeyState(key=key, tier=tier, semaphore=asyncio.Semaphore(settings.per_key_concurrency))


def _pool(*keys: str) -> TavilyKeyPool:
    return TavilyKeyPool([_key_state(k) for k in keys], [])


@respx.mock
async def test_extract_success_no_score_field():
    respx.post("https://api.tavily.com/extract").mock(return_value=httpx.Response(200, json=_OK_RESPONSE))
    async with httpx.AsyncClient() as http:
        results, failed_results = await extract(http, _pool("k1"), "https://militarnyi.com/uk/news/inguar-3")
    assert results[0].url == "https://militarnyi.com/uk/news/inguar-3"
    assert results[0].title == "Inguar-3 news"
    assert results[0].raw_content == "full article text"
    assert results[0].images == ["https://militarnyi.com/img1.jpg"]
    assert not hasattr(results[0], "score")
    assert failed_results == []


@respx.mock
async def test_extract_null_title_coerced_to_empty_string():
    """Tavily can send an explicit `"title": null` (e.g. PDF extraction)
    instead of omitting the key -- extract() must coerce it to "" rather
    than let a bare None reach ExtractResult's `title: str` field."""
    body = {
        "results": [
            {
                "url":         "https://example.com/doc.pdf",
                "title":       None,
                "raw_content": "pdf text",
                "images":      [],
            },
        ],
        "failed_results": [],
    }
    respx.post("https://api.tavily.com/extract").mock(return_value=httpx.Response(200, json=body))
    async with httpx.AsyncClient() as http:
        results, failed_results = await extract(http, _pool("k1"), "https://example.com/doc.pdf")
    assert results[0].title == ""
    assert failed_results == []


@respx.mock
async def test_extract_partial_failure_returns_failed_results_and_zero_cost():
    """1 url Tavily can't resolve — 200, empty results, failed_results has
    {url, error}, credit_cost computed as 0.0."""
    body = {"results": [], "failed_results": [{"url": "https://dead-link.example", "error": "could not resolve"}]}
    respx.post("https://api.tavily.com/extract").mock(return_value=httpx.Response(200, json=body))
    pool = _pool("k1")
    with patch.object(pool, "record_attempt") as mock_record:
        async with httpx.AsyncClient() as http:
            results, failed_results = await extract(http, pool, "https://dead-link.example")
    assert results == []
    assert failed_results[0].url == "https://dead-link.example"
    assert failed_results[0].error == "could not resolve"
    mock_record.assert_called_once_with("k1", 0.0)


@respx.mock
async def test_extract_full_failure_all_urls_unresolved():
    body = {"results": [], "failed_results": [{"url": "https://dead-link.example", "error": "timeout"}]}
    respx.post("https://api.tavily.com/extract").mock(return_value=httpx.Response(200, json=body))
    async with httpx.AsyncClient() as http:
        results, failed_results = await extract(http, _pool("k1"), "https://dead-link.example")
    assert results == []
    assert len(failed_results) == 1


@respx.mock
async def test_extract_depth_override_sent_and_credit_cost_uses_basic_rate():
    pool = _pool("k1")
    route = respx.post("https://api.tavily.com/extract").mock(return_value=httpx.Response(200, json=_OK_RESPONSE))
    with patch.object(pool, "record_attempt") as mock_record:
        async with httpx.AsyncClient() as http:
            await extract(http, pool, "https://militarnyi.com/uk/news/inguar-3", extract_depth="basic")
    sent = json.loads(route.calls.last.request.content)
    assert sent["extract_depth"] == "basic"
    assert sent["urls"] == ["https://militarnyi.com/uk/news/inguar-3"]
    mock_record.assert_called_once_with("k1", 0.2)   # 1 successful url * 0.2 (basic rate)


@respx.mock
async def test_extract_credit_cost_correctness_advanced_multi_result():
    """3 results at extract_depth="advanced" -> credit_cost = 3 * 0.4 = 1.2."""
    body = {
        "results": [
            {"url": "https://a.example", "title": "A", "raw_content": "a"},
            {"url": "https://b.example", "title": "B", "raw_content": "b"},
            {"url": "https://c.example", "title": "C", "raw_content": "c"},
        ],
        "failed_results": [{"url": "https://d.example", "error": "boom"}],
    }
    respx.post("https://api.tavily.com/extract").mock(return_value=httpx.Response(200, json=body))
    pool = _pool("k1")
    with patch.object(pool, "record_attempt") as mock_record:
        async with httpx.AsyncClient() as http:
            results, failed_results = await extract(http, pool, "https://a.example")
    assert len(results) == 3
    assert len(failed_results) == 1
    mock_record.assert_called_once_with("k1", pytest.approx(1.2))


@respx.mock
async def test_extract_record_attempt_never_called_on_429_or_432(monkeypatch):
    monkeypatch.setattr(settings, "per_key_backoff_base_s", 0.001)
    route = respx.post("https://api.tavily.com/extract")
    route.side_effect = [
        httpx.Response(429, json={"detail": {"error": "rate limited"}}),
        httpx.Response(432, json={"detail": {"error": "plan limit exceeded"}}),
        httpx.Response(200, json=_OK_RESPONSE),
    ]
    pool = _pool("k1", "k2")
    with patch.object(pool, "record_attempt") as mock_record:
        async with httpx.AsyncClient() as http:
            await extract(http, pool, "https://militarnyi.com/uk/news/inguar-3")
    mock_record.assert_called_once_with(mock_record.call_args[0][0], 0.4)


@respx.mock
async def test_extract_default_depth_is_advanced():
    route = respx.post("https://api.tavily.com/extract").mock(return_value=httpx.Response(200, json=_OK_RESPONSE))
    async with httpx.AsyncClient() as http:
        await extract(http, _pool("k1"), "https://militarnyi.com/uk/news/inguar-3")
    sent = json.loads(route.calls.last.request.content)
    assert sent["extract_depth"] == "advanced"


@respx.mock
async def test_extract_raises_typed_error_when_entire_pool_exhausted():
    real_error = {"detail": {"error": "This request exceeds your plan's set usage limit."}}
    respx.post("https://api.tavily.com/extract").mock(return_value=httpx.Response(432, json=real_error))
    pool = _pool("k1")
    async with httpx.AsyncClient() as http:
        with pytest.raises(TavilySearchFailedError) as exc_info:
            await extract(http, pool, "https://militarnyi.com/uk/news/inguar-3")
    assert exc_info.value.status_code == 432
