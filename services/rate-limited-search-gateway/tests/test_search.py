import asyncio
import json
import time
from unittest.mock import patch

import httpx
import pytest
import respx

from src.config import settings
from src.keys import KeyState, TavilyKeyPool
from src.search import TavilySearchFailedError, search

_OK_RESPONSE = {
    "results": [
        {"url": "https://militarnyi.com/uk/news/inguar-3", "title": "Inguar-3 news", "content": "text", "score": 0.68},
    ],
}


def _key_state(key: str, tier: str = "free") -> KeyState:
    # Real pools (build_key_pool) always set a semaphore — tests construct
    # KeyState directly, so this mirrors that instead of leaving it None
    # (search() would crash on `async with None`).
    return KeyState(key=key, tier=tier, semaphore=asyncio.Semaphore(settings.per_key_concurrency))


def _pool(*keys: str) -> TavilyKeyPool:
    return TavilyKeyPool([_key_state(k) for k in keys], [])


@respx.mock
async def test_search_happy_path():
    respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(200, json=_OK_RESPONSE))
    async with httpx.AsyncClient() as http:
        results = await search(http, _pool("k1"), "Inguar-3", ["militarnyi.com"], 5)
    assert results[0].url == "https://militarnyi.com/uk/news/inguar-3"
    assert results[0].score == 0.68


@respx.mock
async def test_search_sends_bearer_auth_header():
    route = respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(200, json=_OK_RESPONSE))
    async with httpx.AsyncClient() as http:
        await search(http, _pool("my-test-key"), "Inguar-3", [], 5)
    assert route.calls.last.request.headers["authorization"] == "Bearer my-test-key"


@respx.mock
async def test_search_sends_advanced_depth():
    """"basic" was tried first but verified to miss relevant results entirely
    (not just coarser ranking) — see src/search.py's own docstring."""
    route = respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(200, json=_OK_RESPONSE))
    async with httpx.AsyncClient() as http:
        await search(http, _pool("k1"), "Inguar-3", ["militarnyi.com"], 5)
    sent = json.loads(route.calls.last.request.content)
    assert sent["search_depth"] == "advanced"
    assert sent["include_domains"] == ["militarnyi.com"]
    assert sent["max_results"] == 5


@respx.mock
async def test_search_omits_max_results_when_none():
    """No worker-side default: with max_results=None the field is not sent,
    so Tavily applies its own default."""
    route = respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(200, json=_OK_RESPONSE))
    async with httpx.AsyncClient() as http:
        await search(http, _pool("k1"), "Inguar-3", [], None)
    assert "max_results" not in json.loads(route.calls.last.request.content)


@respx.mock
async def test_search_basic_depth_override_sent_and_credit_cost_is_1():
    """Given caller passes search_depth="basic", the request to Tavily
    carries "basic" and record_attempt is charged 1.0, not the
    advanced-depth default of 2.0."""
    route = respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(200, json=_OK_RESPONSE))
    pool = _pool("k1")
    with patch.object(pool, "record_attempt") as mock_record:
        async with httpx.AsyncClient() as http:
            await search(http, pool, "Inguar-3", ["militarnyi.com"], 5, search_depth="basic")
    sent = json.loads(route.calls.last.request.content)
    assert sent["search_depth"] == "basic"
    mock_record.assert_called_once_with("k1", 1.0)


@respx.mock
async def test_search_record_attempt_never_called_on_429_or_432(monkeypatch):
    """record_attempt must fire only after a confirmed 2xx — never on a
    429/432 branch, even mid-retry-ladder."""
    monkeypatch.setattr(settings, "per_key_backoff_base_s", 0.001)   # keep the test fast
    route = respx.post("https://api.tavily.com/search")
    route.side_effect = [
        httpx.Response(429, json={"detail": {"error": "rate limited"}}),
        httpx.Response(432, json={"detail": {"error": "plan limit exceeded"}}),
        httpx.Response(200, json=_OK_RESPONSE),
    ]
    pool = _pool("k1", "k2")
    with patch.object(pool, "record_attempt") as mock_record:
        async with httpx.AsyncClient() as http:
            await search(http, pool, "X", [], 5)
    mock_record.assert_called_once()
    assert mock_record.call_args[0][1] == 2.0   # advanced-depth default cost, from the final successful call only


@respx.mock
async def test_search_empty_results():
    respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(200, json={"results": []}))
    async with httpx.AsyncClient() as http:
        results = await search(http, _pool("k1"), "nonsense query xyz", [], 5)
    assert results == []


@respx.mock
async def test_search_raises_on_upstream_4xx_unchanged():
    """A non-432 error is NOT a pool-retry signal — raises immediately,
    exactly as before the key pool ever existed."""
    respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(401, text="invalid api key"))
    async with httpx.AsyncClient() as http:
        with pytest.raises(httpx.HTTPStatusError):
            await search(http, _pool("k1"), "X", [], 5)


@respx.mock
async def test_search_rotates_to_next_key_on_432():
    route = respx.post("https://api.tavily.com/search")
    route.side_effect = [
        httpx.Response(432, json={"detail": {"error": "plan limit exceeded"}}),
        httpx.Response(200, json=_OK_RESPONSE),
    ]
    pool = _pool("exhausted-key", "fresh-key")
    async with httpx.AsyncClient() as http:
        results = await search(http, pool, "X", [], 5)
    assert results[0].url == "https://militarnyi.com/uk/news/inguar-3"


@respx.mock
async def test_search_raises_typed_error_when_entire_pool_exhausted():
    real_error = {"detail": {"error": "This request exceeds your plan's set usage limit."}}
    respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(432, json=real_error))
    pool = _pool("k1", "k2")
    async with httpx.AsyncClient() as http:
        with pytest.raises(TavilySearchFailedError) as exc_info:
            await search(http, pool, "X", [], 5)
    assert exc_info.value.status_code == 432
    assert exc_info.value.error_body == real_error


@respx.mock
async def test_search_raises_immediately_when_pool_already_fully_exhausted():
    """No route mocked at all — respx itself would fail the test if search()
    attempted any HTTP call, confirming it never does."""
    pool = _pool("k1")
    pool.mark_exhausted("k1")
    async with httpx.AsyncClient() as http:
        with pytest.raises(TavilySearchFailedError):
            await search(http, pool, "X", [], 5)


# ── per-key semaphore + held-through-backoff 429 retry ──────────────────────
# See .claude/specs/tavily-search-per-key-semaphore.md.

@respx.mock
async def test_search_retries_same_key_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(settings, "per_key_backoff_base_s", 0.001)   # keep the test fast
    route = respx.post("https://api.tavily.com/search")
    route.side_effect = [
        httpx.Response(429, json={"detail": {"error": "rate limited"}}),
        httpx.Response(200, json=_OK_RESPONSE),
    ]
    pool = _pool("k1")
    async with httpx.AsyncClient() as http:
        results = await search(http, pool, "X", [], 5)
    assert results[0].url == "https://militarnyi.com/uk/news/inguar-3"
    # both attempts hit the SAME key — never rotated away on a 429
    assert route.call_count == 2


@respx.mock
async def test_search_retries_same_key_on_raw_transport_error_then_succeeds(monkeypatch):
    """Live-confirmed bug: a raw httpx connection/timeout error (not an HTTP
    response at all) used to propagate straight out of _call_with_key_pool,
    skipping the entire 429/432 retry-rotation ladder — a single-shot
    failure even though the same key/url often succeeds moments later."""
    monkeypatch.setattr(settings, "per_key_backoff_base_s", 0.001)   # keep the test fast
    route = respx.post("https://api.tavily.com/search")
    route.side_effect = [
        httpx.ConnectTimeout("connection timed out"),
        httpx.Response(200, json=_OK_RESPONSE),
    ]
    pool = _pool("k1")
    async with httpx.AsyncClient() as http:
        results = await search(http, pool, "X", [], 5)
    assert results[0].url == "https://militarnyi.com/uk/news/inguar-3"
    assert route.call_count == 2


@respx.mock
async def test_search_raises_typed_error_when_every_attempt_is_a_transport_error(monkeypatch):
    monkeypatch.setattr(settings, "per_key_backoff_base_s", 0.001)
    monkeypatch.setattr(settings, "per_key_rate_limit_cooldown_s", 0.01)
    monkeypatch.setattr(settings, "retry_wall_clock_cap_seconds", 0.05)
    respx.post("https://api.tavily.com/search").mock(side_effect=httpx.ReadTimeout("read timed out"))
    pool = _pool("k1")
    async with httpx.AsyncClient() as http:
        with pytest.raises(TavilySearchFailedError) as exc_info:
            await search(http, pool, "X", [], 5)
    assert exc_info.value.status_code == 599


@respx.mock
async def test_search_semaphore_held_across_the_backoff_sleep(monkeypatch):
    """The whole point of holding through backoff: a second concurrent
    caller must NOT be able to acquire this key's semaphore while the first
    caller is asleep between its 429 and its retry."""
    monkeypatch.setattr(settings, "per_key_backoff_base_s", 0.05)
    route = respx.post("https://api.tavily.com/search")
    route.side_effect = [
        httpx.Response(429, json={"detail": {"error": "rate limited"}}),
        httpx.Response(200, json=_OK_RESPONSE),
    ]
    pool = _pool("k1")
    key_state = pool.pick()

    async def _run_search():
        async with httpx.AsyncClient() as http:
            return await search(http, pool, "X", [], 5)

    task = asyncio.create_task(_run_search())
    await asyncio.sleep(0.01)   # let it hit the 429 and enter the backoff sleep
    # Semaphore capacity is per_key_concurrency (1 by default) — acquire
    # every remaining slot to prove none are free while task 1 backs off.
    held = [await key_state.semaphore.acquire() for _ in range(settings.per_key_concurrency - 1)]
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(key_state.semaphore.acquire(), timeout=0.02)
    for _ in held:
        key_state.semaphore.release()
    results = await task
    assert results[0].url == "https://militarnyi.com/uk/news/inguar-3"


@respx.mock
async def test_search_moves_to_different_key_after_exhausting_per_key_retries(monkeypatch):
    """Both k1/k2 start with identical weight, so pick() may legitimately
    try either one first — assert on the STRUCTURE (2 calls to one key,
    then 1 to the other, then success), not on which key goes first."""
    monkeypatch.setattr(settings, "per_key_backoff_base_s", 0.001)
    monkeypatch.setattr(settings, "per_key_max_retries", 2)
    route = respx.post("https://api.tavily.com/search")
    route.side_effect = [
        httpx.Response(429, json={"detail": {"error": "rate limited"}}),   # first key, attempt 1
        httpx.Response(429, json={"detail": {"error": "rate limited"}}),   # first key, attempt 2 — retries exhausted
        httpx.Response(200, json=_OK_RESPONSE),                            # the other key
    ]
    pool = _pool("k1", "k2")
    async with httpx.AsyncClient() as http:
        results = await search(http, pool, "X", [], 5)
    assert results[0].url == "https://militarnyi.com/uk/news/inguar-3"
    assert route.call_count == 3
    used_keys = [c.request.headers["authorization"] for c in route.calls]
    first_key = used_keys[0]
    assert used_keys[1] == first_key           # both 429s hit the SAME key
    assert used_keys[2] != first_key           # success came from the OTHER key
    assert {used_keys[0], used_keys[2]} == {"Bearer k1", "Bearer k2"}


@respx.mock
async def test_search_432_mid_retry_sequence_switches_immediately_no_extra_backoff(monkeypatch):
    monkeypatch.setattr(settings, "per_key_backoff_base_s", 0.001)
    route = respx.post("https://api.tavily.com/search")
    route.side_effect = [
        httpx.Response(429, json={"detail": {"error": "rate limited"}}),           # k1 attempt 1
        httpx.Response(432, json={"detail": {"error": "plan limit exceeded"}}),    # k1 attempt 2 — exhausted
        httpx.Response(200, json=_OK_RESPONSE),                                    # k2
    ]
    pool = _pool("k1", "k2")
    async with httpx.AsyncClient() as http:
        results = await search(http, pool, "X", [], 5)
    assert results[0].url == "https://militarnyi.com/uk/news/inguar-3"
    assert route.call_count == 3


_NEWS_RESPONSE = {
    "results": [
        {"url": "https://militarnyi.com/en/news/a", "title": "A", "content": "text", "score": 0.7,
         "published_date": "Sat, 05 Sep 2026 15:03:16 GMT"},
        {"url": "https://defence-ua.com/b", "title": "B", "content": "text", "score": 0.5},
    ],
}


@respx.mock
async def test_search_default_topic_is_not_sent_and_dates_are_none():
    """topic="general" is Tavily's own default — never sent, so existing
    callers' request bodies stay byte-identical to before this field existed."""
    route = respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(200, json=_OK_RESPONSE))
    async with httpx.AsyncClient() as http:
        results = await search(http, _pool("k1"), "X", [], 5)
    assert "topic" not in json.loads(route.calls.last.request.content)
    assert results[0].published_date is None


@respx.mock
async def test_search_news_topic_is_sent_and_published_date_passed_through_verbatim():
    route = respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(200, json=_NEWS_RESPONSE))
    async with httpx.AsyncClient() as http:
        results = await search(http, _pool("k1"), "X", [], 5, topic="news")
    assert json.loads(route.calls.last.request.content)["topic"] == "news"
    assert results[0].published_date == "Sat, 05 Sep 2026 15:03:16 GMT"   # not parsed, passed through
    assert results[1].published_date is None                              # an undated news result stays None


# ── global rate limiter (src/limiter.py) ─────────────────────────────────────

class _FakeClock:
    def __init__(self) -> None:
        self.now = time.monotonic()   # search() takes its deadline from the real monotonic clock
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += s


@respx.mock
async def test_a_429_pauses_every_call_for_the_retry_after_the_api_sent(monkeypatch):
    from src.limiter import RateLimiter
    monkeypatch.setattr(settings, "per_key_backoff_base_s", 0.0)
    clock = _FakeClock()
    pool = TavilyKeyPool([_key_state("k1")], [], limiter=RateLimiter(1000, clock=clock, sleep=clock.sleep))
    route = respx.post("https://api.tavily.com/search")
    route.side_effect = [httpx.Response(429, headers={"retry-after": "17"}, json={"detail": "slow down"}), httpx.Response(200, json=_OK_RESPONSE)]
    async with httpx.AsyncClient() as http:
        await search(http, pool, "X", [], 5)
    assert 17.0 in clock.slept          # the retry waited for the pause the API asked for


@respx.mock
async def test_without_retry_after_the_default_pause_is_used(monkeypatch):
    from src.limiter import RateLimiter
    monkeypatch.setattr(settings, "per_key_backoff_base_s", 0.0)
    monkeypatch.setattr(settings, "rate_limit_pause_seconds", 30.0)
    clock = _FakeClock()
    pool = TavilyKeyPool([_key_state("k1")], [], limiter=RateLimiter(1000, clock=clock, sleep=clock.sleep))
    route = respx.post("https://api.tavily.com/search")
    route.side_effect = [httpx.Response(429, json={}), httpx.Response(200, json=_OK_RESPONSE)]
    async with httpx.AsyncClient() as http:
        await search(http, pool, "X", [], 5)
    assert 30.0 in clock.slept


@respx.mock
async def test_every_attempt_takes_a_limiter_slot(monkeypatch):
    from src.limiter import RateLimiter
    clock = _FakeClock()
    lim = RateLimiter(2, clock=clock, sleep=clock.sleep)
    pool = TavilyKeyPool([_key_state("k1")], [], limiter=lim)
    respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(200, json=_OK_RESPONSE))
    async with httpx.AsyncClient() as http:
        for _ in range(3):
            await search(http, pool, "X", [], 5)
    assert clock.slept == [60.0]        # the third call waited for the window to free a slot
