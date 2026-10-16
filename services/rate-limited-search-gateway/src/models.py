from typing import Literal

from pydantic import BaseModel

Depth = Literal["basic", "advanced"]
Topic = Literal["general", "news"]


class SearchRequest(BaseModel):
    query:           str
    include_domains: list[str] = []
    # None = not sent to Tavily at all, so Tavily's own default applies
    # (10, verified live 2026-10-08). No worker-side default on purpose: a
    # hidden cap here silently cut every caller that omitted the field.
    max_results:     int | None = None
    # Defaults to "advanced" — see src/search.py's docstring for the
    # empirical reason "basic" isn't the default. Omitted by existing
    # callers (resolve_materiel, resolve_quote, resolve_proof's Navigator)
    # → unchanged behavior.
    search_depth:    Depth     = "advanced"
    # "general" (default) is Tavily's own default and is never sent
    # explicitly. "news" restricts results to news sources and is the ONLY
    # mode in which Tavily returns published_date (verified live
    # 2026-10-07: 0/15 dated results on "general", 12/15 on "news").
    topic:           Topic     = "general"


class SearchResult(BaseModel):
    url:     str
    title:   str
    content: str
    score:   float
    # Passed through exactly as Tavily returns it (an RFC-1123 string, e.g.
    # "Sat, 05 Sep 2026 15:03:16 GMT") — no parsing here, a worker does one
    # computation. None on topic="general" and for undated news results.
    published_date: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult] = []


class ExtractRequest(BaseModel):
    url:            str
    # Defaults to "advanced" for the same empirical reason as
    # SearchRequest.search_depth (see src/search.py).
    extract_depth:  Depth = "advanced"


class ExtractResult(BaseModel):
    # No `score` field — verified live this session that Tavily's
    # /extract never returns one (unlike /search).
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
