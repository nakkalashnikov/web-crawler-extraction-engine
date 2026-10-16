from __future__ import annotations

import asyncio
import random
import time
from typing import Awaitable, Callable

import httpx
import structlog

from .config import settings
from .limiter import RateLimiter, RateLimitWaitExceeded
from .models import Depth, SearchResult, Topic

log = structlog.get_logger()


class TavilySearchFailedError(RuntimeError):
    def __init__(self, status_code: int, error_body: dict | str):
        self.status_code = status_code
        self.error_body = error_body
        super().__init__(f"Tavily request failed: {status_code} — {error_body}")


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    try:
        value = float(resp.headers.get("retry-after", ""))
    except ValueError:
        return None
    return value if value > 0 else None


def _rate_limit_headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower() == "retry-after" or "ratelimit" in k.lower() or "rate-limit" in k.lower()}


def _safe_body(resp: httpx.Response) -> dict | str:
    try:
        return resp.json()
    except ValueError:
        return resp.text


async def execute_request(
    http: httpx.AsyncClient,
    limiter: RateLimiter,
    attempt_fn: Callable[[], Awaitable[httpx.Response]],
    deadline: float,
) -> httpx.Response:
    """Resilient execution loop:
    - Enforces sliding-window rate limit via RateLimiter before each call.
    - Handles 429 by setting a global pause/penalty on the limiter and backing off.
    - Handles transient network TransportErrors with backoff and jitter.
    - Raises TavilySearchFailedError upon deadline or retries exhausted.
    """
    last_status = 500
    last_body: dict | str = "request timed out or failed"

    for attempt in range(settings.max_retries):
        if time.monotonic() >= deadline:
            break

        try:
            await limiter.acquire(deadline)
        except RateLimitWaitExceeded:
            raise TavilySearchFailedError(429, "Rate limit wait exceeded deadline")

        try:
            resp = await attempt_fn()
        except httpx.TransportError as exc:
            last_status, last_body = 599, str(exc)
            if attempt < settings.max_retries - 1:
                delay = settings.backoff_base_s * (2 ** attempt) + random.uniform(0, 1)
                log.warning("Transport error, retrying", attempt=attempt + 1, delay_s=round(delay, 2), error=str(exc))
                await asyncio.sleep(delay)
            continue

        if resp.status_code == 429:
            last_status, last_body = resp.status_code, _safe_body(resp)
            pause = _retry_after_seconds(resp) or settings.rate_limit_pause_seconds
            limiter.penalize(pause)
            log.warning("429 rate limited — paused all callers", pause_s=pause, headers=_rate_limit_headers(resp))
            if attempt < settings.max_retries - 1:
                delay = settings.backoff_base_s * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
            continue

        resp.raise_for_status()
        return resp

    raise TavilySearchFailedError(last_status, last_body)


async def search(
    http: httpx.AsyncClient,
    limiter: RateLimiter,
    query: str,
    include_domains: list[str],
    max_results: int | None,
    search_depth: Depth = "advanced",
    topic: Topic = "general",
) -> list[SearchResult]:
    deadline = time.monotonic() + settings.retry_wall_clock_cap_seconds

    payload = {
        "query": query,
        "include_domains": include_domains,
        "search_depth": search_depth,
    }
    if max_results is not None:
        payload["max_results"] = max_results
    if topic != "general":
        payload["topic"] = topic

    async def attempt_fn() -> httpx.Response:
        return await http.post(
            f"{settings.tavily_base_url}/search",
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            json=payload,
            timeout=settings.http_timeout_seconds,
        )

    resp = await execute_request(http, limiter, attempt_fn, deadline)
    data = resp.json()
    return [
        SearchResult(
            url=r["url"],
            title=r["title"],
            content=r["content"],
            score=r["score"],
            published_date=r.get("published_date"),
        )
        for r in data.get("results", [])
    ]
