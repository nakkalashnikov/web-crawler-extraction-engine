"""source-extract — social-platform domain classification + metadata/caption
fetch (the `social_direct` path's own Call A: yt-dlp's /metadata, plus
Twitter's oEmbed fallback). Split out of extract.py purely to keep that
file under CLAUDE.md's 350-line limit."""
from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog
from curl_cffi.requests import AsyncSession as StealthSession

from .config import settings
from .models import Platform

log = structlog.get_logger()

# Fixed social/video domain set — matching one of these means Tavily would
# fail on the URL directly, so it is classified with zero fetches instead
# of ever being handed to Call A/Call B. Netloc is compared after stripping
# a leading www./m./mobile. prefix.
_SOCIAL_DOMAINS: dict[str, Platform] = {
    "youtube.com":   "youtube",
    "youtu.be":      "youtube",
    "twitter.com":   "twitter",
    "x.com":         "twitter",
    "t.me":          "telegram",
    "telegram.me":   "telegram",
    "instagram.com": "instagram",
    "facebook.com":  "facebook",
    "fb.watch":      "facebook",
    "tiktok.com":    "tiktok",
}

_STRIPPABLE_NETLOC_PREFIXES = ("www.", "m.", "mobile.")

# A telegram URL is only a single-post permalink in the shape /<channel>/<id>
# — same requirement embed_parser.py's own _TELEGRAM_LINK_RE already enforces
# for embeds found inside an article. This module's domain classifier used to
# skip that check entirely and accept ANY t.me path (live-confirmed real-run
# bug: a channel PREVIEW/LISTING url, t.me/s/<channel>?before=<id>, sailed
# through as platform="telegram", canonical_url=<the listing url itself> —
# telegram_fetch.fetch() then tried to resolve it as one message and failed
# every time, even though the channel/listing page was reachable and public).
_TELEGRAM_POST_PATH_RE = re.compile(r"^/[a-zA-Z0-9_]+/\d+/?$")


def classify_social_domain(url: str) -> Platform | None:
    netloc = urlparse(url).netloc.lower().split(":")[0]
    for prefix in _STRIPPABLE_NETLOC_PREFIXES:
        if netloc.startswith(prefix):
            netloc = netloc[len(prefix):]
            break
    platform = _SOCIAL_DOMAINS.get(netloc)
    if platform == "telegram" and not _TELEGRAM_POST_PATH_RE.match(urlparse(url).path):
        # Not a single-post permalink (e.g. a /s/<channel> listing page, or
        # the bare channel root) — fall through to ordinary article
        # extraction instead, where tavily-search's /extract can still pull
        # real text/images off the rendered page.
        return None
    return platform


_SOCIAL_METADATA_MAX_ATTEMPTS = 3   # matches media_gather.py's _MAX_IMAGE_ATTEMPTS convention


async def fetch_social_metadata(http: httpx.AsyncClient, url: str) -> dict[str, Any]:
    """Queries yt-dlp /metadata to extract title, uploader, upload_date, description, and images.
    Returns title, formatted content, and images, or empty values if yt-dlp fails.

    Live-confirmed real-run bug: a transient failure here (timeout, a
    momentary non-200 from yt-dlp) used to be indistinguishable in the
    final result from "this post genuinely has no caption" — both produced
    the identical {"title": "", "content": "", "images": []}. A post with a
    real caption (confirmed by manually viewing it) came back completely
    empty this way, and the vision-verification step downstream then judged
    10 frames with zero textual context, on a post that had explicit
    confirming text available. Retried now (same convention as
    media_gather.py's _ingest_photo's _MAX_IMAGE_ATTEMPTS), and on final
    failure an `error` key is set so the caller can tell "empty because
    nothing was ever here" apart from "empty because the fetch failed" —
    see activities/extract/activities.py's extract_source, which persists
    this to a diagnostic artifact (DB) rather than only a container log
    that rotates away."""
    last_err: str | None = None
    for attempt in range(1, _SOCIAL_METADATA_MAX_ATTEMPTS + 1):
        try:
            resp = await http.post(
                f"{settings.yt_dlp_url}/metadata",
                json={"url": url},
                timeout=settings.http_timeout_seconds,
            )
            if resp.status_code == 200:
                data = resp.json()
                title = data.get("title") or ""
                uploader = data.get("uploader") or ""
                date = data.get("upload_date") or ""
                desc = data.get("description") or ""
                images = data.get("images") or []
                if not images and data.get("thumbnail_url"):
                    images = [data["thumbnail_url"]]

                lines = []
                if title:
                    lines.append(f"Title: {title}")
                if uploader:
                    lines.append(f"Author / Uploader: {uploader}")
                if date:
                    lines.append(f"Date: {date}")
                if desc:
                    lines.append(f"Description:\n{desc}")
                content = "\n".join(lines).strip()
                return {"title": title, "content": content, "images": images}
            last_err = f"yt-dlp /metadata returned HTTP {resp.status_code}"
        except Exception as e:
            last_err = str(e)
        if attempt < _SOCIAL_METADATA_MAX_ATTEMPTS:
            await asyncio.sleep(1.0 * attempt)

    log.warning(
        "social metadata extraction failed after retries", url=url,
        attempts=_SOCIAL_METADATA_MAX_ATTEMPTS, error=last_err,
    )
    return {"title": "", "content": "", "images": [], "error": last_err}


_TWITTER_OEMBED_MAX_ATTEMPTS = 3   # same convention


async def fetch_twitter_oembed_text(stealth: StealthSession, url: str) -> dict[str, str]:
    """Fetches text and author from public unauthenticated Twitter/X oEmbed
    endpoint — the fallback fetch_social_metadata's own caller reaches for
    when THAT call came back with no content. Same retry + distinguishable-
    error shape as fetch_social_metadata (same class of bug: an oembed
    fetch that merely failed must not look identical to "there's genuinely
    no text here")."""
    last_err: str | None = None
    for attempt in range(1, _TWITTER_OEMBED_MAX_ATTEMPTS + 1):
        try:
            r = await stealth.get(
                "https://publish.twitter.com/oembed",
                params={"url": url, "omit_script": "true"},
                headers=settings.browser_headers,
                timeout=settings.http_timeout_seconds, impersonate="chrome",
            )
            if r.status_code == 200:
                data = r.json()
                author = data.get("author_name") or ""
                html_snippet = data.get("html") or ""
                plain_text = re.sub(r"<[^>]+>", " ", html_snippet).strip()
                title = f"Post by {author}" if author else "X / Twitter Post"
                return {"title": title, "content": plain_text}
            last_err = f"twitter oembed returned HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        if attempt < _TWITTER_OEMBED_MAX_ATTEMPTS:
            await asyncio.sleep(1.0 * attempt)

    log.warning("twitter oembed fetch failed after retries", url=url, error=last_err)
    return {"title": "", "content": "", "error": last_err}
