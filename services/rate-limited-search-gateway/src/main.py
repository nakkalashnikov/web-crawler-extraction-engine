import asyncio
import contextlib
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI, HTTPException

from .config import settings
from .extract import extract
from .keys import TavilyKeyPool, build_key_pool, recheck_exhausted_keys
from .models import ExtractRequest, ExtractResponse, SearchRequest, SearchResponse
from .search import TavilySearchFailedError, search

log = structlog.get_logger()

_http:         httpx.AsyncClient    | None = None
_semaphore:    asyncio.Semaphore    | None = None
_key_pool:     TavilyKeyPool        | None = None
_recheck_task: asyncio.Task         | None = None


async def _recheck_loop(http: httpx.AsyncClient, key_pool: TavilyKeyPool) -> None:
    """Tavily gives no reset timestamp — this is the only way an exhausted
    key ever gets a chance to be picked again within a long-running
    process. See keys.py's module docstring and recheck_exhausted_keys."""
    while True:
        await asyncio.sleep(settings.recheck_interval_seconds)
        try:
            await recheck_exhausted_keys(http, key_pool)
        except Exception as e:
            log.warning("periodic key recheck failed — will retry next interval", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http, _semaphore, _key_pool, _recheck_task
    _http = httpx.AsyncClient()
    _semaphore = asyncio.Semaphore(settings.max_concurrent)
    _key_pool = await build_key_pool(_http, settings.key_pool)
    _recheck_task = asyncio.create_task(_recheck_loop(_http, _key_pool))
    log.info(
        "tavily-search ready", max_concurrent=settings.max_concurrent,
        n_free_keys=_key_pool.n_free, n_paygo_keys=_key_pool.n_paygo,
        recheck_interval_seconds=settings.recheck_interval_seconds,
    )
    yield
    _recheck_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await _recheck_task
    await _http.aclose()


app = FastAPI(title="tavily-search", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    async with _semaphore:  # held for the whole request, including the key-pool retry loop inside search()
        try:
            results = await search(
                _http, _key_pool, req.query, req.include_domains, req.max_results, req.search_depth, req.topic,
            )
        except TavilySearchFailedError as e:
            log.warning("tavily search failed after exhausting key pool", status=e.status_code, query=req.query)
            raise HTTPException(502, {"message": str(e), "status_code": e.status_code, "tavily_error": e.error_body})
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            log.warning("tavily upstream error", status=status, query=req.query)
            raise HTTPException(status, f"tavily upstream error {status}: {e.response.text[:200]}")
        except httpx.HTTPError as e:
            log.error("tavily request failed", query=req.query, error=str(e))
            raise HTTPException(502, f"tavily request failed: {e}")
    return SearchResponse(results=results)


@app.post("/extract", response_model=ExtractResponse)
async def extract_endpoint(req: ExtractRequest):
    async with _semaphore:  # held for the whole request, including the key-pool retry loop inside extract()
        try:
            results, failed_results = await extract(_http, _key_pool, req.url, req.extract_depth)
        except TavilySearchFailedError as e:
            log.warning("tavily extract failed after exhausting key pool", status=e.status_code, url=req.url)
            raise HTTPException(502, {"message": str(e), "status_code": e.status_code, "tavily_error": e.error_body})
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            log.warning("tavily upstream error", status=status, url=req.url)
            raise HTTPException(status, f"tavily upstream error {status}: {e.response.text[:200]}")
        except httpx.HTTPError as e:
            log.error("tavily request failed", url=req.url, error=str(e))
            raise HTTPException(502, f"tavily request failed: {e}")
    return ExtractResponse(results=results, failed_results=failed_results)
