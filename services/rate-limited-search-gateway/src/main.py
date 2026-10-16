import asyncio
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI, HTTPException

from .config import settings
from .extract import extract
from .limiter import RateLimiter
from .models import ExtractRequest, ExtractResponse, SearchRequest, SearchResponse
from .search import TavilySearchFailedError, search

log = structlog.get_logger()

_http: httpx.AsyncClient | None = None
_semaphore: asyncio.Semaphore | None = None
_limiter: RateLimiter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http, _semaphore, _limiter
    _http = httpx.AsyncClient()
    _semaphore = asyncio.Semaphore(settings.max_concurrent)
    _limiter = RateLimiter(max_per_minute=settings.max_requests_per_minute)
    log.info(
        "search-gateway ready",
        max_concurrent=settings.max_concurrent,
        max_rpm=settings.max_requests_per_minute,
    )
    yield
    await _http.aclose()


app = FastAPI(title="rate-limited-search-gateway", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    async with _semaphore:
        try:
            results = await search(
                _http, _limiter, req.query, req.include_domains, req.max_results, req.search_depth, req.topic,
            )
        except TavilySearchFailedError as e:
            log.warning("tavily search failed", status=e.status_code, query=req.query)
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
    async with _semaphore:
        try:
            results, failed_results = await extract(_http, _limiter, req.url, req.extract_depth)
        except TavilySearchFailedError as e:
            log.warning("tavily extract failed", status=e.status_code, url=req.url)
            raise HTTPException(502, {"message": str(e), "status_code": e.status_code, "tavily_error": e.error_body})
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            log.warning("tavily upstream error", status=status, url=req.url)
            raise HTTPException(status, f"tavily upstream error {status}: {e.response.text[:200]}")
        except httpx.HTTPError as e:
            log.error("tavily request failed", url=req.url, error=str(e))
            raise HTTPException(502, f"tavily request failed: {e}")
    return ExtractResponse(results=results, failed_results=failed_results)
