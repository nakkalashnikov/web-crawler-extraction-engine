from __future__ import annotations

import time

import httpx
import structlog

from .config import settings
from .limiter import RateLimiter
from .models import Depth, ExtractResult, FailedResult
from .search import TavilySearchFailedError, _safe_body, execute_request

log = structlog.get_logger()


async def extract(
    http: httpx.AsyncClient,
    limiter: RateLimiter,
    url: str,
    extract_depth: Depth = "advanced",
) -> tuple[list[ExtractResult], list[FailedResult]]:
    deadline = time.monotonic() + settings.retry_wall_clock_cap_seconds

    async def attempt_fn() -> httpx.Response:
        return await http.post(
            f"{settings.tavily_base_url}/extract",
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            json={
                "urls": [url],
                "extract_depth": extract_depth,
                "include_images": True,
            },
            timeout=settings.http_timeout_seconds,
        )

    resp = await execute_request(http, limiter, attempt_fn, deadline)
    try:
        data = resp.json()
    except ValueError:
        raise TavilySearchFailedError(resp.status_code, _safe_body(resp))

    results = [
        ExtractResult(
            url=r.get("url") or "",
            title=r.get("title") or "",
            raw_content=r.get("raw_content") or "",
            images=r.get("images") or [],
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
    return results, failed_results
