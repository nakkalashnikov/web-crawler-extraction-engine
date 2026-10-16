"""Live contract test for src/search.py's search() -- real api.tavily.com,
no mocking. See spec-2-1-1's Design Notes: this assertion was independently
confirmed via a live curl call during spec-writing (all 3 sampled results
carried a numeric score, e.g. 0.83742094), not written from assumption.

Run: `pytest -m live workers/tavily-search/tests/live/test_live_search.py`
(skips cleanly with no real TAVILY_API_KEY_* configured).
"""
from __future__ import annotations

import httpx
import pytest

from src.keys import build_key_pool
from src.search import search
from tests.live.conftest import require_real_tavily_key, run


@pytest.mark.live
def test_search_results_have_numeric_score():
    keys = require_real_tavily_key()

    async def _run():
        async with httpx.AsyncClient() as http:
            pool = await build_key_pool(http, keys)
            return await search(http, pool, "HIMARS Ukraine", [], 5)

    results = run(_run())
    assert results, "expected at least one result"
    for r in results:
        assert isinstance(r.score, float)
