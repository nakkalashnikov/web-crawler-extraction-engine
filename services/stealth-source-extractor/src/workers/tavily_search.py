import httpx
from pydantic import BaseModel


class SearchResult(BaseModel):
    url:     str
    title:   str
    content: str
    score:   float
    # RFC-1123 string as Tavily returns it; only populated when search()
    # is called with topic="news" (and not even then for every result).
    published_date: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult] = []


class ExtractResult(BaseModel):
    # No `score` field — tavily-search's /extract never returns one
    # (unlike /search).
    url:         str
    title:       str       = ""
    raw_content: str       = ""
    images:      list[str] = []


class FailedResult(BaseModel):
    url:   str
    error: str


class ExtractResponse(BaseModel):
    results:        list[ExtractResult] = []
    failed_results: list[FailedResult]  = []


async def search(
    http: httpx.AsyncClient, url: str, query: str,
    include_domains: list[str] | None = None, max_results: int | None = None, topic: str = "general",
) -> SearchResponse:
    body = {
        "query":           query,
        "include_domains": include_domains or [],
    }
    # Omitted when None: the worker then does not send it to Tavily either,
    # and Tavily's own default applies.
    if max_results is not None:
        body["max_results"] = max_results
    if topic != "general":
        body["topic"] = topic
    resp = await http.post(f"{url}/search", json=body)
    resp.raise_for_status()
    return SearchResponse(**resp.json())


async def extract(
    http: httpx.AsyncClient, url: str, article_url: str, extract_depth: str = "advanced",
) -> ExtractResponse:
    resp = await http.post(f"{url}/extract", json={
        "url":           article_url,
        "extract_depth": extract_depth,
    })
    resp.raise_for_status()
    return ExtractResponse(**resp.json())
