import pytest
import respx
import httpx
from src.limiter import RateLimiter
from src.search import search, TavilySearchFailedError
from src.extract import extract

@pytest.fixture
def limiter():
    return RateLimiter(max_per_minute=60)

@pytest.mark.asyncio
async def test_search_success(respx_mock, limiter):
    respx_mock.post("https://api.tavily.com/search").mock(
        return_value=httpx.Response(200, json={
            "results": [
                {"url": "https://example.com/a", "title": "A", "content": "Sample content", "score": 0.9}
            ]
        })
    )
    async with httpx.AsyncClient() as client:
        results = await search(client, limiter, query="test", include_domains=[], max_results=5)
    assert len(results) == 1
    assert results[0].url == "https://example.com/a"
    assert results[0].title == "A"

@pytest.mark.asyncio
async def test_search_429_penalizes_limiter(respx_mock, limiter):
    route = respx_mock.post("https://api.tavily.com/search")
    route.side_effect = [
        httpx.Response(429, headers={"retry-after": "5"}, text="Rate limit exceeded"),
        httpx.Response(200, json={"results": [{"url": "https://example.com/ok", "title": "OK", "content": "Text", "score": 1.0}]})
    ]
    async with httpx.AsyncClient() as client:
        results = await search(client, limiter, query="test", include_domains=[], max_results=5)
    assert len(results) == 1
    assert results[0].url == "https://example.com/ok"

@pytest.mark.asyncio
async def test_extract_success(respx_mock, limiter):
    respx_mock.post("https://api.tavily.com/extract").mock(
        return_value=httpx.Response(200, json={
            "results": [
                {"url": "https://example.com/article", "title": "Title", "raw_content": "Full article text", "images": ["https://example.com/img.jpg"]}
            ],
            "failed_results": []
        })
    )
    async with httpx.AsyncClient() as client:
        results, failed = await extract(client, limiter, url="https://example.com/article")
    assert len(results) == 1
    assert results[0].title == "Title"
    assert results[0].raw_content == "Full article text"
    assert len(failed) == 0
