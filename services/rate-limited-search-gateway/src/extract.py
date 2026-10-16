from __future__ import annotations

import time

import httpx
import structlog

from .config import settings
from .keys import KeyState, TavilyKeyPool, extract_credit_cost
from .models import Depth, ExtractResult, FailedResult
from .search import TavilySearchFailedError, _call_with_key_pool, _key_from_response, _safe_body

log = structlog.get_logger()


async def extract(
    http: httpx.AsyncClient, key_pool: TavilyKeyPool, url: str, extract_depth: Depth = "advanced",
) -> tuple[list[ExtractResult], list[FailedResult]]:
    """POST /extract — one `url` per call (no batching in this endpoint's
    contract). Shares search()'s retry/rotation ladder via
    _call_with_key_pool: same per-key semaphore, same 429 backoff-retry-
    same-key, same 432 mark-exhausted-and-rotate, same wall-clock deadline.

    `extract_depth` defaults to "advanced" for the same empirical reason
    documented on search()'s `search_depth` — an unverified but consistent
    assumption that Tavily's "basic" depth risks missing content, carried
    over defensively rather than re-verified specifically for /extract.

    `key_pool.record_attempt` fires exactly once, after a confirmed 2xx,
    with `extract_credit_cost(extract_depth, len(results))` — a url that
    fully fails to resolve (empty `results`, non-empty `failed_results`)
    still costs 0.0, never a flat per-attempt charge. Tavily's /extract
    response has no `score` field — verified live this session — so
    ExtractResult never carries one (unlike SearchResult)."""
    deadline = time.monotonic() + settings.retry_wall_clock_cap_seconds

    async def attempt_fn(key_state: KeyState) -> httpx.Response:
        return await http.post(
            f"{settings.tavily_base_url}/extract",
            headers = {"Authorization": f"Bearer {key_state.key}"},
            json = {
                "urls":           [url],
                "extract_depth":  extract_depth,
                "include_images": True,
            },
            timeout = settings.http_timeout_seconds,
        )

    resp = await _call_with_key_pool(http, key_pool, attempt_fn, deadline)
    try:
        data = resp.json()
    except ValueError:
        # Malformed JSON on a confirmed 2xx — Tavily may have already
        # billed this call, but we can't parse a credit cost out of it, so
        # raise before any record_attempt happens (mirrors search.py's
        # _safe_body helper pattern).
        raise TavilySearchFailedError(resp.status_code, _safe_body(resp))

    results = [
        ExtractResult(
            # `or default`, not `.get(key, default)`, on all four fields --
            # confirmed live (PDF extraction, see
            # tests/live/test_live_extract.py) that Tavily sends a field
            # back as an explicit `null`, not an absent key, when a source
            # (e.g. a PDF) carries no value for it. `.get(key, default)`
            # only covers the absent-key case, so a bare `None` was
            # reaching ExtractResult's typed fields and raising a pydantic
            # ValidationError. Only `title` has been live-observed as
            # `null` so far, but the same gap applies to any of these
            # fields, so all four are guarded uniformly.
            url         = r.get("url") or "",
            title       = r.get("title") or "",
            raw_content = r.get("raw_content") or "",
            images      = r.get("images") or [],
        )
        for r in data.get("results", [])
    ]
    failed_results = [
        FailedResult(url=r.get("url", ""), error=r.get("error", ""))
        for r in data.get("failed_results", [])
    ]

    log.info(
        "tavily extract succeeded", url=url, n_results=len(results), n_failed=len(failed_results),
    )
    key_pool.record_attempt(_key_from_response(resp), extract_credit_cost(extract_depth, len(results)))
    return results, failed_results
