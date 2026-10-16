from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog
from curl_cffi.requests import AsyncSession as StealthSession
from curl_cffi.requests import RequestsError as StealthError
from pydantic import ValidationError

from workers import tavily_search

from .config import settings
from .embed_parser import extract_video_embeds
from .image_filter import filter_images
from .models import Reason, SourceExtractResult, VideoEmbed
from .page_meta import PageMeta, parse_page_meta
from .social_meta import classify_social_domain, fetch_social_metadata, fetch_twitter_oembed_text

log = structlog.get_logger()

async def _fetch_own_page(stealth: StealthSession, url: str) -> tuple[list[VideoEmbed], PageMeta]:
    """Call B: this worker's own raw-HTML fetch — regex-scans it for video/social embeds tavily-search's /extract doesn't
    surface, and reads the page metadata (publication date, og:image, figure captions). Never raises — any failure here
    (403, timeout, malformed HTML) degrades to empty results and never becomes an overall failure.

    Uses the TLS-impersonating stealth session, not the plain httpx client every internal worker-to-worker call in this
    file still uses — this is an arbitrary THIRD-PARTY site, and several real ones (Wikimedia, Newsweek, live-confirmed
    2026-10-05/06) block httpx's/requests' own TLS ClientHello fingerprint outright regardless of headers sent, while
    the exact same request succeeds through curl_cffi's impersonate='chrome'."""
    try:
        r = await stealth.get(
            url, headers=settings.browser_headers,
            timeout=settings.http_timeout_seconds, allow_redirects=True,
            impersonate="chrome",
        )
        r.raise_for_status()
    except StealthError as e:
        log.warning("own embed-regex fetch failed", url=url, error=str(e))
        return [], PageMeta()
    try:
        embeds = extract_video_embeds(r.text)
    except Exception as e:
        log.warning("embed regex scan failed", url=url, error=str(e))
        embeds = []
    return embeds, parse_page_meta(r.text, url)


async def _wayback_snapshot(stealth: StealthSession, url: str) -> tuple[str | None, bool]:
    """Returns (timestamp, was_unavailable). was_unavailable=True means the
    CDX API itself could not be reached after retries (e.g. Internet
    Archive outage) — distinct from a clean 200 response that simply has no
    snapshot for this URL."""
    for attempt in range(settings.wayback_max_retries):
        try:
            r = await stealth.get(
                "https://web.archive.org/cdx/search/cdx",
                params={"url": url, "output": "json", "filter": "statuscode:200", "limit": 1},
                timeout=settings.wayback_timeout_seconds, impersonate="chrome",
            )
            if r.status_code != 200:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            data = r.json()
            if len(data) > 1:
                return data[1][1], False  # timestamp
            # A 200 response with zero rows used to be trusted immediately,
            # on the theory that CDX wouldn't lie about a clean "no
            # snapshot" answer. Live-confirmed wrong: querying the SAME url
            # again minutes later turned up 3 real snapshots — CDX is
            # eventually-consistent/flaky, not authoritative on a single
            # call. Retry it exactly like a transient failure, within the
            # same wayback_max_retries budget, instead of giving up on the
            # first empty answer.
            if attempt < settings.wayback_max_retries - 1:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            return None, False  # exhausted retries, still empty — genuinely no snapshot
        except (StealthError, ValueError) as e:
            log.warning("wayback CDX lookup failed", url=url, attempt=attempt, error=str(e))
            await asyncio.sleep(1.5 * (attempt + 1))
    return None, True


async def _try_wayback(
    http: httpx.AsyncClient, stealth: StealthSession, tavily_search_url: str, url: str,
) -> tuple[SourceExtractResult | None, Reason | None]:
    """Re-delegates to tavily-search's /extract on the archived snapshot
    URL instead of hand-rolled extraction (see spec Design Notes)."""
    ts, unavailable = await _wayback_snapshot(stealth, url)
    if unavailable:
        return None, "wayback_unavailable"
    if not ts:
        return None, "no_snapshot"

    wb_url = f"https://web.archive.org/web/{ts}/{url}"
    try:
        resp = await tavily_search.extract(http, tavily_search_url, wb_url)
    except (httpx.HTTPError, ValueError, ValidationError) as e:
        log.warning("tavily-search unreachable on wayback snapshot", url=url, wb_url=wb_url, error=str(e))
        return None, "no_snapshot"

    if not resp.results:
        return None, "no_snapshot"
    r = resp.results[0]
    if len(r.raw_content) < settings.min_text_chars:
        return None, "empty_content"

    return SourceExtractResult(
        url=url, kind="article", source="wayback", wayback_timestamp=ts,
        title=r.title, content=r.raw_content, images=filter_images(r.images),
    ), None


async def _try_apnews_video_mirror(http: httpx.AsyncClient, url: str) -> SourceExtractResult | None:
    """If an AP News video page is blocked by Cloudflare, resolves the official
    video mirror uploaded by Associated Press / AP Archive on YouTube."""
    if "apnews.com" not in url or "/video" not in url:
        return None
    try:
        slug = url.split("/video/")[-1].split("?")[0].rstrip("/")
        parts = slug.rsplit("-", 1)
        if len(parts) > 1 and len(parts[1]) >= 16:
            slug = parts[0]
        words = slug.replace("-", " ")

        query = f"ytsearch1:{words} AP Archive"
        resp = await http.post(
            f"{settings.yt_dlp_url}/metadata",
            json={"url": query},
            timeout=settings.http_timeout_seconds,
        )
        if resp.status_code == 200:
            data = resp.json()
            vid_url = data.get("webpage_url")
            if vid_url and "youtube.com" in vid_url:
                title = data.get("title") or words.title()
                desc = data.get("description") or f"Associated Press report: {title}"
                images = data.get("images") or ([data["thumbnail_url"]] if data.get("thumbnail_url") else [])
                return SourceExtractResult(
                    url=url, kind="article", source="live",
                    title=title, content=desc,
                    video_embeds=[VideoEmbed(url=vid_url, platform="youtube")],
                    images=images,
                )
    except Exception as e:
        log.warning("apnews youtube mirror lookup failed", url=url, error=str(e))
    return None


async def extract(
    http: httpx.AsyncClient, stealth: StealthSession, tavily_search_url: str, url: str,
) -> SourceExtractResult:
    """Domain classification first (zero I/O). For an ordinary article, Call
    A (tavily-search's /extract) and Call B (this worker's own embed-regex
    fetch) run in parallel. Only the Call A(+Wayback) path failing produces
    an overall kind="failure" — Call B failing/finding nothing never does.

    Two HTTP clients, not one: `http` (plain httpx) for every call to one
    of OUR OWN workers (tavily-search, yt-dlp) — no anti-bot measure to
    evade there, just unnecessary TLS-impersonation overhead. `stealth`
    (curl_cffi, TLS-fingerprint-impersonating) for every fetch of an
    arbitrary THIRD-PARTY site (the article page itself, web.archive.org,
    publish.twitter.com) — see _fetch_own_page's own docstring for why
    plain httpx isn't enough there."""
    platform = classify_social_domain(url)
    if platform is not None:
        meta: dict[str, Any] = {}
        if platform == "twitter":
            meta = await fetch_social_metadata(http, url)
            if not meta.get("content"):
                oembed = await fetch_twitter_oembed_text(stealth, url)
                meta["title"] = meta.get("title") or oembed.get("title", "")
                meta["content"] = meta.get("content") or oembed.get("content", "")
                # oembed's own failure matters only if it didn't salvage
                # content either — otherwise the first call's error is now
                # moot too (real text was found).
                if not meta["content"] and oembed.get("error"):
                    meta["error"] = oembed["error"]
        elif platform != "telegram":
            meta = await fetch_social_metadata(http, url)

        embeds: list[VideoEmbed] = []
        if platform in ("youtube", "tiktok", "twitter"):
            embeds.append(VideoEmbed(url=url, platform=platform))
        elif platform == "facebook" and any(x in url for x in ("/videos/", "/watch", "fb.watch")):
            embeds.append(VideoEmbed(url=url, platform="facebook"))

        final_title, final_content = meta.get("title", ""), meta.get("content", "")
        return SourceExtractResult(
            url=url, kind="social_direct", platform=platform, canonical_url=url,
            title=final_title, content=final_content,
            video_embeds=embeds, images=meta.get("images", []),
            # Twitter's oembed fallback (above) can still recover real
            # content after fetch_social_metadata's own error — only
            # surface meta_error when the fetch genuinely failed AND
            # nothing else salvaged it.
            meta_error=meta.get("error") if not final_title and not final_content else None,
        )

    try:
        resp, (embeds, page) = await asyncio.gather(
            tavily_search.extract(http, tavily_search_url, url),
            _fetch_own_page(stealth, url),
        )
    except (httpx.HTTPError, ValueError, ValidationError) as e:
        log.warning("tavily-search unreachable", url=url, error=str(e))
        return SourceExtractResult(url=url, kind="failure", reason="upstream_unavailable")

    if resp.results and not resp.failed_results:
        r = resp.results[0]
        text_embeds = extract_video_embeds(r.raw_content)
        seen_urls = {e.url for e in embeds}
        all_embeds = list(embeds)
        for e in text_embeds:
            if e.url not in seen_urls:
                seen_urls.add(e.url)
                all_embeds.append(e)

        if not all_embeds and "/video" in url:
            ap_mirror = await _try_apnews_video_mirror(http, url)
            if ap_mirror is not None and ap_mirror.video_embeds:
                for ve in ap_mirror.video_embeds:
                    if ve.url not in seen_urls:
                        seen_urls.add(ve.url)
                        all_embeds.append(ve)

        return SourceExtractResult(
            url=url, kind="article", source="live",
            title=r.title, content=r.raw_content, images=filter_images(r.images),
            video_embeds=all_embeds,
            published_at=page.published_at, og_image=page.og_image,
            image_captions={**page.captions, **({page.og_image: page.og_image_alt} if page.og_image and page.og_image_alt else {})},
        )

    wb, reason = await _try_wayback(http, stealth, tavily_search_url, url)
    if wb is not None:
        text_embeds = extract_video_embeds(wb.content)
        seen_urls = {e.url for e in embeds}
        all_embeds = list(embeds)
        for e in text_embeds:
            if e.url not in seen_urls:
                seen_urls.add(e.url)
                all_embeds.append(e)
        wb.video_embeds = all_embeds
        return wb

    ap_mirror = await _try_apnews_video_mirror(http, url)
    if ap_mirror is not None:
        return ap_mirror

    return SourceExtractResult(url=url, kind="failure", reason=reason)
