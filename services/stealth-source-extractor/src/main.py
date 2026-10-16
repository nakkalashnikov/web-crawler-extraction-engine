from contextlib import asynccontextmanager

import httpx
import structlog
from curl_cffi.requests import AsyncSession as StealthSession
from fastapi import FastAPI

from .config import settings
from .extract import extract
from .models import SourceExtractRequest, SourceExtractResult

log = structlog.get_logger()

_http:    httpx.AsyncClient | None = None
# TLS-fingerprint-impersonating client for arbitrary third-party sites
# (extract.py's own docstrings explain why plain httpx isn't enough) —
# separate from _http, which stays plain httpx for every call to one of
# our OWN workers (tavily-search, yt-dlp).
_stealth: StealthSession | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http, _stealth
    # Not 30s: tavily-search queues bursts (see config.tavily_timeout_seconds). `pool` matters as much as `read`
    # — with more concurrent requests than httpx's connection limit, the surplus waits for a free connection.
    _http = httpx.AsyncClient(timeout=httpx.Timeout(settings.tavily_timeout_seconds, connect=10.0))
    _stealth = StealthSession()
    log.info("source-extract ready")
    yield
    await _http.aclose()
    await _stealth.close()


app = FastAPI(title="source-extract", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/extract", response_model=SourceExtractResult)
async def extract_endpoint(req: SourceExtractRequest):
    return await extract(_http, _stealth, settings.tavily_search_url, req.url)
