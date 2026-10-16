import httpx
import pytest
import respx
from curl_cffi.requests import RequestsError as StealthError

from src.extract import extract
from src.social_meta import fetch_social_metadata as _fetch_social_metadata

_TAVILY_URL = "http://tavily-search:8000"
_YT_DLP_URL = "http://yt-dlp:8000"


class _FakeStealthResponse:
    """Just enough of curl_cffi's response shape (.status_code, .text,
    .json(), .raise_for_status()) for extract.py's stealth.get() call
    sites — respx can't intercept curl_cffi (it patches httpx's own
    transport only), so the stealth session itself is swapped for this
    fake rather than mocked at the HTTP layer."""
    def __init__(self, status_code: int = 200, text: str = "", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json = json_data

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise StealthError(f"HTTP {self.status_code}")


class _FakeStealth:
    """url -> _FakeStealthResponse map. A url with no entry raises
    StealthError — every real call site in extract.py already catches
    that broadly and degrades gracefully, same as an unmocked respx route
    would for the httpx side. A url mapped to a LIST is consumed one
    response per call (repeating the last entry once exhausted) — for
    simulating a retry that gets a different answer on a later attempt."""
    def __init__(self, responses: dict[str, "_FakeStealthResponse"] | None = None):
        self._responses = responses or {}
        self._call_counts: dict[str, int] = {}

    async def get(self, url: str, **kwargs):
        if url not in self._responses:
            raise StealthError(f"no fake response configured for {url}")
        entry = self._responses[url]
        if isinstance(entry, list):
            i = self._call_counts.get(url, 0)
            self._call_counts[url] = i + 1
            return entry[min(i, len(entry) - 1)]
        return entry

_HTML_WITH_YOUTUBE_EMBED = (
    '<html><body><article><p>Some article text.</p>'
    '<iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>'
    "</article></body></html>"
)
_HTML_NO_EMBEDS = "<html><body><article><p>Some article text, no embeds.</p></article></body></html>"

_CDX_HIT = [["urlkey", "timestamp"], ["com,example)/a", "20240101000000"]]
_CDX_MISS = [["urlkey", "timestamp"]]


def _tavily_extract_response(url: str, title: str = "A title", raw_content: str = "x" * 2000, images=None):
    return {
        "results": [{"url": url, "title": title, "raw_content": raw_content, "images": images or []}],
        "failed_results": [],
    }


def _tavily_failed_response(url: str, error: str = "not found"):
    return {"results": [], "failed_results": [{"url": url, "error": error}]}


class TestSocialDirectClassification:
    @pytest.mark.parametrize("url,expected_platform", [
        ("https://www.youtube.com/watch?v=abc123", "youtube"),
        ("https://youtu.be/abc123", "youtube"),
        ("https://x.com/someone/status/123", "twitter"),
        ("https://twitter.com/someone/status/123", "twitter"),
        ("https://t.me/somechannel/42", "telegram"),
        ("https://www.instagram.com/p/abc123/", "instagram"),
        ("https://m.facebook.com/somepage/posts/123", "facebook"),
        ("https://www.tiktok.com/@someone/video/123", "tiktok"),
    ])
    @respx.mock
    async def test_direct_social_link_classified_with_zero_fetches(self, url, expected_platform):
        async with httpx.AsyncClient() as http:
            res = await extract(http, _FakeStealth(), _TAVILY_URL, url)
        assert res.kind == "social_direct"
        assert res.platform == expected_platform
        assert res.canonical_url == url

    @respx.mock
    async def test_social_metadata_retries_once_then_succeeds(self):
        route = respx.post(f"{_YT_DLP_URL}/metadata")
        route.side_effect = [
            httpx.Response(502, json={"error": "upstream overloaded"}),
            httpx.Response(200, json={"title": "T", "description": "D", "uploader": "U"}),
        ]
        async with httpx.AsyncClient() as http:
            result = await _fetch_social_metadata(http, "https://www.instagram.com/reels/abc")
        assert route.call_count == 2
        assert result["title"] == "T"
        assert "error" not in result

    @respx.mock
    async def test_social_metadata_exhausts_retries_sets_error_not_silently_empty(self):
        """Live-confirmed real-run bug: a transient yt-dlp /metadata failure
        used to come back as the exact same {"title": "", "content": "",
        "images": []} as "this post genuinely has no caption" — a caption-
        bearing Instagram reel lost its text with zero trace of why. Now
        the caller can tell the two apart via the "error" key."""
        respx.post(f"{_YT_DLP_URL}/metadata").mock(
            side_effect=httpx.ConnectTimeout("connection timed out")
        )
        async with httpx.AsyncClient() as http:
            result = await _fetch_social_metadata(http, "https://www.instagram.com/reels/abc")
        assert result["title"] == ""
        assert result["content"] == ""
        assert "connection timed out" in result["error"]

    @respx.mock
    async def test_instagram_url_meta_error_surfaces_on_extract_result(self):
        respx.post(f"{_YT_DLP_URL}/metadata").mock(
            side_effect=httpx.ConnectTimeout("connection timed out")
        )
        async with httpx.AsyncClient() as http:
            res = await extract(http, _FakeStealth(), _TAVILY_URL, "https://www.instagram.com/reels/abc")
        assert res.title == "" and res.content == ""
        assert res.meta_error is not None and "connection timed out" in res.meta_error

    @respx.mock
    async def test_twitter_oembed_fallback_clears_meta_error_when_it_recovers_content(self):
        """Twitter's own oembed fallback can still salvage real content after
        _fetch_social_metadata's own call failed — meta_error must not leak
        through once something else recovered usable text."""
        respx.post(f"{_YT_DLP_URL}/metadata").mock(side_effect=httpx.ConnectTimeout("timed out"))
        stealth = _FakeStealth({
            "https://publish.twitter.com/oembed": _FakeStealthResponse(
                200, json_data={"author_name": "Someone", "html": "<p>real tweet text</p>"},
            ),
        })
        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://x.com/someone/status/123")
        assert "real tweet text" in res.content
        assert res.meta_error is None

    @respx.mock
    async def test_twitter_meta_error_surfaces_when_oembed_also_fails(self):
        respx.post(f"{_YT_DLP_URL}/metadata").mock(side_effect=httpx.ConnectTimeout("metadata timed out"))
        stealth = _FakeStealth({
            "https://publish.twitter.com/oembed": [
                _FakeStealthResponse(502),
                _FakeStealthResponse(502),
            ],
        })
        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://x.com/someone/status/123")
        assert res.content == ""
        # oembed's OWN failure (not metadata's stale one) is what surfaces,
        # since oembed ran last and still didn't recover anything
        assert res.meta_error is not None and "502" in res.meta_error

    @respx.mock
    async def test_telegram_listing_url_not_classified_as_social_direct(self):
        """Live-confirmed bug: a channel preview/listing url (t.me/s/<channel>
        ?before=<id>, which shows a scroll of messages, not one post) used to
        be accepted as platform="telegram" with canonical_url set to the
        listing url itself — telegram_fetch.fetch() then always failed trying
        to resolve it as a single message. Only a /<channel>/<digits>
        single-post permalink may classify as telegram; anything else falls
        through to ordinary article extraction."""
        url = "https://t.me/s/somechannel?before=14087"
        respx.post(f"{_TAVILY_URL}/extract").mock(
            return_value=httpx.Response(200, json=_tavily_extract_response(url))
        )
        stealth = _FakeStealth({url: _FakeStealthResponse(200, text=_HTML_NO_EMBEDS)})
        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, url)
        assert res.kind == "article"
        assert res.platform is None

    @respx.mock
    async def test_plain_article_url_not_classified_as_social(self):
        respx.post(f"{_TAVILY_URL}/extract").mock(
            return_value=httpx.Response(200, json=_tavily_extract_response("https://example.com/a"))
        )
        stealth = _FakeStealth({"https://example.com/a": _FakeStealthResponse(200, text=_HTML_NO_EMBEDS)})
        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://example.com/a")
        assert res.kind == "article"


class TestArticleExtraction:
    @respx.mock
    async def test_tavily_success_merges_call_a_and_call_b(self):
        respx.post(f"{_TAVILY_URL}/extract").mock(
            return_value=httpx.Response(
                200,
                json=_tavily_extract_response(
                    "https://example.com/a", title="Real title", images=["https://example.com/photo.jpg"],
                ),
            )
        )
        stealth = _FakeStealth({"https://example.com/a": _FakeStealthResponse(200, text=_HTML_WITH_YOUTUBE_EMBED)})
        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://example.com/a")
        assert res.kind == "article"
        assert res.source == "live"
        assert res.title == "Real title"
        assert res.images == ["https://example.com/photo.jpg"]
        assert len(res.video_embeds) == 1
        assert res.video_embeds[0].platform == "youtube"

    @respx.mock
    async def test_page_meta_from_the_own_fetch_lands_on_the_result(self):
        respx.post(f"{_TAVILY_URL}/extract").mock(
            return_value=httpx.Response(200, json=_tavily_extract_response("https://example.com/a", title="T", images=[]))
        )
        html = (
            '<meta property="article:published_time" content="2025-05-14T09:30:00Z">'
            '<meta property="og:image" content="https://example.com/cover.jpg">'
            '<figure><img src="https://example.com/p.jpg"><figcaption>Caption</figcaption></figure>'
        )
        stealth = _FakeStealth({"https://example.com/a": _FakeStealthResponse(200, text=html)})
        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://example.com/a")
        assert res.published_at == "2025-05-14T09:30:00Z"
        assert res.og_image == "https://example.com/cover.jpg"
        assert res.image_captions == {"https://example.com/p.jpg": "Caption"}

    @respx.mock
    async def test_blocked_own_fetch_leaves_the_page_meta_empty(self):
        respx.post(f"{_TAVILY_URL}/extract").mock(
            return_value=httpx.Response(200, json=_tavily_extract_response("https://example.com/a", title="T", images=[]))
        )
        stealth = _FakeStealth({"https://example.com/a": _FakeStealthResponse(403)})
        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://example.com/a")
        assert res.kind == "article" and res.published_at is None and res.og_image is None and res.image_captions == {}

    @respx.mock
    async def test_images_filtered_through_filter_images_on_live_path(self):
        """Proves the wiring (filter_images actually gets called on the
        primary live path), not just the pure function in isolation."""
        respx.post(f"{_TAVILY_URL}/extract").mock(
            return_value=httpx.Response(
                200,
                json=_tavily_extract_response(
                    "https://example.com/a",
                    images=[
                        "https://example.com/img/logo.svg",
                        "https://example.com/media/photo.jpg",
                    ],
                ),
            )
        )
        stealth = _FakeStealth({"https://example.com/a": _FakeStealthResponse(200, text=_HTML_NO_EMBEDS)})
        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://example.com/a")
        assert res.images == ["https://example.com/media/photo.jpg"]

    @respx.mock
    async def test_own_fetch_failure_never_causes_overall_failure(self):
        respx.post(f"{_TAVILY_URL}/extract").mock(
            return_value=httpx.Response(200, json=_tavily_extract_response("https://example.com/a"))
        )
        stealth = _FakeStealth({"https://example.com/a": _FakeStealthResponse(403)})
        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://example.com/a")
        assert res.kind == "article"
        assert res.video_embeds == []

    @respx.mock
    async def test_tavily_unreachable_returns_upstream_unavailable(self):
        respx.post(f"{_TAVILY_URL}/extract").mock(side_effect=httpx.ConnectError("connection refused"))
        stealth = _FakeStealth({"https://example.com/a": _FakeStealthResponse(200, text=_HTML_NO_EMBEDS)})
        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://example.com/a")
        assert res.kind == "failure"
        assert res.reason == "upstream_unavailable"


class TestWaybackFallback:
    @respx.mock
    async def test_tavily_failed_result_falls_back_to_wayback_via_tavily(self):
        wb_url = "https://web.archive.org/web/20240101000000/https://example.com/a"

        def extract_side_effect(request: httpx.Request) -> httpx.Response:
            body = request.content
            if b"web.archive.org" in body:
                return httpx.Response(200, json=_tavily_extract_response(wb_url))
            return httpx.Response(200, json=_tavily_failed_response("https://example.com/a"))

        respx.post(f"{_TAVILY_URL}/extract").mock(side_effect=extract_side_effect)
        stealth = _FakeStealth({
            "https://web.archive.org/cdx/search/cdx": _FakeStealthResponse(200, json_data=_CDX_HIT),
            "https://example.com/a": _FakeStealthResponse(200, text=_HTML_NO_EMBEDS),
        })

        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://example.com/a")
        assert res.kind == "article"
        assert res.source == "wayback"
        assert res.wayback_timestamp == "20240101000000"

    @respx.mock
    async def test_empty_results_and_failed_results_falls_back_to_wayback(self):
        """Neither `results` nor `failed_results` mentions the requested
        URL — not the same as a reported failure, but `if resp.results and
        not resp.failed_results` is falsy either way, so this must still
        fall through to the Wayback path rather than being treated as a
        (wrongly) successful empty article."""
        wb_url = "https://web.archive.org/web/20240101000000/https://example.com/a"

        def extract_side_effect(request: httpx.Request) -> httpx.Response:
            body = request.content
            if b"web.archive.org" in body:
                return httpx.Response(200, json=_tavily_extract_response(wb_url))
            return httpx.Response(200, json={"results": [], "failed_results": []})

        respx.post(f"{_TAVILY_URL}/extract").mock(side_effect=extract_side_effect)
        stealth = _FakeStealth({
            "https://web.archive.org/cdx/search/cdx": _FakeStealthResponse(200, json_data=_CDX_HIT),
            "https://example.com/a": _FakeStealthResponse(200, text=_HTML_NO_EMBEDS),
        })

        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://example.com/a")
        assert res.kind == "article"
        assert res.source == "wayback"
        assert res.wayback_timestamp == "20240101000000"

    @respx.mock
    async def test_images_filtered_through_filter_images_on_wayback_path(self):
        """Proves both `images=r.images` construction sites (live and
        Wayback) go through filter_images, not just one of the two."""
        wb_url = "https://web.archive.org/web/20240101000000/https://example.com/a"

        def extract_side_effect(request: httpx.Request) -> httpx.Response:
            body = request.content
            if b"web.archive.org" in body:
                return httpx.Response(
                    200,
                    json=_tavily_extract_response(
                        wb_url,
                        images=[
                            "https://example.com/img/logo.svg",
                            "https://example.com/media/photo.jpg",
                        ],
                    ),
                )
            return httpx.Response(200, json=_tavily_failed_response("https://example.com/a"))

        respx.post(f"{_TAVILY_URL}/extract").mock(side_effect=extract_side_effect)
        stealth = _FakeStealth({
            "https://web.archive.org/cdx/search/cdx": _FakeStealthResponse(200, json_data=_CDX_HIT),
            "https://example.com/a": _FakeStealthResponse(200, text=_HTML_NO_EMBEDS),
        })

        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://example.com/a")
        assert res.source == "wayback"
        assert res.images == ["https://example.com/media/photo.jpg"]

    @respx.mock
    async def test_no_wayback_snapshot_returns_failure(self):
        respx.post(f"{_TAVILY_URL}/extract").mock(
            return_value=httpx.Response(200, json=_tavily_failed_response("https://example.com/a"))
        )
        stealth = _FakeStealth({
            "https://web.archive.org/cdx/search/cdx": _FakeStealthResponse(200, json_data=_CDX_MISS),
            "https://example.com/a": _FakeStealthResponse(200, text=_HTML_NO_EMBEDS),
        })
        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://example.com/a")
        assert res.kind == "failure"
        assert res.reason == "no_snapshot"

    @respx.mock
    async def test_wayback_cdx_empty_result_retried_and_recovers_a_later_hit(self):
        """Live-confirmed bug: CDX's own 200-with-zero-rows response used to
        be trusted immediately, no retry — but querying the exact same url
        again (manually, minutes later) turned up 3 real snapshots. CDX is
        flaky/eventually-consistent, not authoritative on one call."""
        wb_url = "https://web.archive.org/web/20240101000000/https://example.com/a"

        def extract_side_effect(request: httpx.Request) -> httpx.Response:
            body = request.content
            if b"web.archive.org" in body:
                return httpx.Response(200, json=_tavily_extract_response(wb_url, images=[]))
            return httpx.Response(200, json=_tavily_failed_response("https://example.com/a"))

        respx.post(f"{_TAVILY_URL}/extract").mock(side_effect=extract_side_effect)
        stealth = _FakeStealth({
            "https://web.archive.org/cdx/search/cdx": [
                _FakeStealthResponse(200, json_data=_CDX_MISS),
                _FakeStealthResponse(200, json_data=_CDX_HIT),
            ],
            "https://example.com/a": _FakeStealthResponse(200, text=_HTML_NO_EMBEDS),
        })
        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://example.com/a")
        assert res.source == "wayback"

    @respx.mock
    async def test_wayback_outage_reported_as_wayback_unavailable(self):
        respx.post(f"{_TAVILY_URL}/extract").mock(
            return_value=httpx.Response(200, json=_tavily_failed_response("https://example.com/a"))
        )
        stealth = _FakeStealth({
            "https://web.archive.org/cdx/search/cdx": _FakeStealthResponse(503),
            "https://example.com/a": _FakeStealthResponse(200, text=_HTML_NO_EMBEDS),
        })
        async with httpx.AsyncClient() as http:
            res = await extract(http, stealth, _TAVILY_URL, "https://example.com/a")
        assert res.kind == "failure"
        assert res.reason == "wayback_unavailable"
