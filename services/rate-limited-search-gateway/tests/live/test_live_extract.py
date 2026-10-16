"""Live contract tests for src/extract.py's extract() -- real
api.tavily.com, no mocking. Each fixed URL/assertion pair was independently
confirmed via a real curl call to api.tavily.com during spec-writing (see
spec-2-1-1's Design Notes), not written from assumption.

These are the permanent regression guard for the include_images fix landed
this session (extract.py sends "include_images": True) -- a respx-mocked
test can prove the request body is shaped right, but only a live call can
prove Tavily actually returns non-empty images for it.

Run: `pytest -m live workers/tavily-search/tests/live/test_live_extract.py`
(skips cleanly with no real TAVILY_API_KEY_* configured).
"""
from __future__ import annotations

import httpx
import pytest

from src.config import settings
from src.extract import extract
from src.keys import build_key_pool
from tests.live.conftest import require_real_tavily_key, run

_RADA5_URL = "https://rada5.com/video/15386-opublikovan-rolik-s-novoj-ukrainskoj-raketoj-grom-2-video/"
_UNRESOLVABLE_URL = "https://this-domain-absolutely-does-not-exist-mv9k2x.invalid/some-page"
_PDF_URL = "https://www.gulflink.osd.mil/17SuspectCW_BWStorageSites.pdf"


@pytest.mark.live
def test_extract_returns_images_for_rada5_url():
    """Confirmed live: 8 real image URLs on this article. Regression guard
    for the include_images request-param fix."""
    keys = require_real_tavily_key()

    async def _run():
        async with httpx.AsyncClient() as http:
            pool = await build_key_pool(http, keys)
            return await extract(http, pool, _RADA5_URL)

    results, failed_results = run(_run())
    assert results, f"expected at least one result, got none (failed_results={failed_results})"
    assert results[0].images, "expected a non-empty images list for rada5.com"


@pytest.mark.live
def test_extract_raw_response_has_no_score_field():
    """Confirmed live: Tavily's /extract response never carries a "score"
    key anywhere in a result dict (unlike /search) -- inspects the raw JSON
    directly rather than going through extract()'s own ExtractResult model,
    since the model simply wouldn't have a score attribute to check either
    way; this asserts the absence in Tavily's actual wire response."""
    keys = require_real_tavily_key()

    async def _run() -> dict:
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http:
            # Try each configured key directly against /extract until one
            # isn't exhausted (432), rather than going through
            # build_key_pool/pick() -- this deliberately does NOT depend on
            # GET /usage succeeding (it has its own separate, stricter
            # rate limit -- confirmed live while implementing spec-2-1-1),
            # and keys[0] alone (a fixed .env ordering) isn't guaranteed to
            # have any remaining quota.
            last_resp = None
            for key in keys:
                resp = await http.post(
                    f"{settings.tavily_base_url}/extract",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"urls": [_RADA5_URL], "extract_depth": "advanced", "include_images": True},
                )
                if resp.status_code in (429, 432):
                    # 429 (per-key rate limited) and 432 (plan limit
                    # exhausted) both just mean "try the next key" here --
                    # this is a one-shot test loop, not production's
                    # backoff-retry-same-key ladder, so no sleep needed.
                    last_resp = resp
                    continue
                resp.raise_for_status()
                return resp.json()
            body = last_resp.text if last_resp is not None else "no keys configured"
            raise AssertionError(f"every configured TAVILY_API_KEY_* is rate limited or exhausted (429/432): {body}")

    data = run(_run())
    assert data["results"], "expected at least one raw result"
    assert "score" not in data["results"][0]


@pytest.mark.live
def test_extract_unresolvable_domain_returns_failed_results_not_exception():
    """Confirmed live: a domain that can't resolve at all is a normal 200
    with empty results and a failed_results entry -- not an exception."""
    keys = require_real_tavily_key()

    async def _run():
        async with httpx.AsyncClient() as http:
            pool = await build_key_pool(http, keys)
            return await extract(http, pool, _UNRESOLVABLE_URL)

    results, failed_results = run(_run())
    assert results == []
    assert len(failed_results) == 1
    assert failed_results[0].url == _UNRESOLVABLE_URL
    assert failed_results[0].error, "expected a non-empty error message for an unresolvable domain"


@pytest.mark.live
def test_extract_pdf_returns_long_raw_content():
    """Confirmed live: raw_content was 22715 characters of real, readable
    text for this PDF. title came back None -- PDFs apparently don't carry
    one through Tavily's extraction -- so this deliberately does NOT assert
    anything about title, per spec-2-1-1's I/O matrix."""
    keys = require_real_tavily_key()

    async def _run():
        async with httpx.AsyncClient() as http:
            pool = await build_key_pool(http, keys)
            return await extract(http, pool, _PDF_URL)

    results, failed_results = run(_run())
    assert results, f"expected at least one result, got none (failed_results={failed_results})"
    assert len(results[0].raw_content) > 5000
