"""schema.org VideoObject/NewsArticle.video JSON-LD, OpenGraph/Twitter
Player Card meta tags, and a blunt HLS/DASH-manifest catch-all — the
universal, player-agnostic video signals embed_parser.py's named-platform
patterns can't see (AP/Yahoo/Townnews-CMS-style syndication, or any page
using standard SEO video markup regardless of underlying player vendor).
Split out of embed_parser.py (2026-09-30) purely to stay under CLAUDE.md's
350-line cap — see .claude/scratch/embed-detection-gaps-plan.md.

Live-confirmed 2026-09-30 on a real HIMARS resolve_proof run: this exact gap
silently dropped the actual video on 3 of 10 extracted sources in one run
(all syndicated copies of the same AP story, whose only video signal was a
VideoObject JSON-LD block) — see that plan's Audit section. Every URL this
module surfaces is a raw candidate string, not yet classified into a
Platform — embed_parser.py's own _add_classified() re-runs each one through
its named-platform patterns before tagging it, so e.g. a VideoObject.
contentUrl that happens to be a YouTube link dedupes correctly against a
plain-text YouTube link elsewhere on the same page.
"""
from __future__ import annotations

import json
import re

_JSONLD_SCRIPT_RE = re.compile(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.I | re.S)

# Any <script type="...json...">...</script> block (ld+json, plain
# application/json client-hydration state, etc.) — used to exclude these
# from the blunt manifest scan below. Broader than _JSONLD_SCRIPT_RE on
# purpose: a page commonly embeds the SAME video metadata twice — once as
# ld+json for crawlers, once as a framework's own hydration JSON blob
# (e.g. Next.js __NEXT_DATA__) for the client — and JSON serializers
# routinely escape "&" as the literal backslash-u-0026 sequence inside
# ANY such block, not just ld+json specifically. Live-confirmed 2026-09-30:
# a real Yahoo page carries the identical contentUrl in both an
# application/ld+json block AND a separate application/json one.
_ANY_JSON_SCRIPT_RE = re.compile(r'<script[^>]*type=["\'][^"\']*json[^"\']*["\'][^>]*>.*?</script>', re.I | re.S)

# Attribute order on a <meta> tag isn't guaranteed, so match the whole tag
# first, then pull property/name and content from within it — same
# two-step technique embed_parser.py uses for Facebook's XFBML tag.
_META_TAG_RE     = re.compile(r"<meta\b[^>]*>", re.I)
_META_ATTR_RE    = re.compile(r'(?:property|name)\s*=\s*["\']([^"\']+)["\']', re.I)
_META_CONTENT_RE = re.compile(r'content\s*=\s*["\']([^"\']*)["\']', re.I)
_OG_VIDEO_META_KEYS = {
    "og:video", "og:video:url", "og:video:secure_url",
    "twitter:player", "twitter:player:stream",
}

# Deliberately .m3u8/.mpd ONLY, never .mp4: a bare .mp4 extension is far too
# common as an unrelated asset (ad creative, unrelated background clip) to
# blind-regex across a whole page, whereas a manifest URL sitting loose in
# an inline player config (with no JSON-LD/OG signal at all) is a much
# stronger, low-false-positive tell that THIS is the page's actual video.
# The character class also excludes "\\" — defense in depth alongside
# gather_manifest_urls' own JSON-script stripping below: live-confirmed
# 2026-09-30 that some pages carry a THIRD, doubly-JSON-escaped copy of the
# same video metadata in a script with no identifiable "json" type at all
# (e.g. a framework's own inline state var), which script-stripping alone
# can't find and exclude. Without this, the class would swallow a JSON
# string's literal `&` escape sequence as if it were normal URL text,
# producing a garbled duplicate instead of stopping cleanly at the
# backslash (a truncated-but-valid URL is an acceptable, non-corrupting
# fallback for a case this rare).
_DIRECT_MANIFEST_RE = re.compile(r"https?://[^\s\"'<>\\]+\.(?:m3u8|mpd)(?:\?[^\s\"'<>\\]*)?", re.I)


def _iter_jsonld_nodes(html: str):
    """Yields every top-level JSON-LD node on the page — unwrapping a
    "@graph" array or a bare top-level array, since both are legal shapes.
    A single malformed <script> block (common in the wild — hand-authored
    or template-generated JSON-LD frequently has trailing commas or stray
    control characters) is skipped, never allowed to kill detection on the
    rest of the page — same per-unit failure isolation this codebase uses
    everywhere else (see NFR13)."""
    for m in _JSONLD_SCRIPT_RE.finditer(html):
        raw = m.group(1).strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("@graph"), list):
            yield from data["@graph"]
        elif isinstance(data, list):
            yield from data
        else:
            yield data


def _find_video_objects(node) -> list[dict]:
    """Recursively finds every dict whose "@type" is (or includes)
    "VideoObject" anywhere inside a JSON-LD node — covers both a
    stand-alone VideoObject node and one nested under a NewsArticle/
    Article's own "video"/"subjectOf" property."""
    found: list[dict] = []
    if isinstance(node, dict):
        types = node.get("@type")
        types = types if isinstance(types, list) else [types]
        if "VideoObject" in types:
            found.append(node)
        for value in node.values():
            found.extend(_find_video_objects(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_find_video_objects(item))
    return found


def gather_jsonld_video_urls(html: str) -> list[str]:
    urls: list[str] = []
    for node in _iter_jsonld_nodes(html):
        for video_obj in _find_video_objects(node):
            for key in ("contentUrl", "embedUrl"):
                url = video_obj.get(key)
                if isinstance(url, str) and url:
                    urls.append(url)
    return urls


def gather_meta_video_urls(html: str) -> list[str]:
    urls: list[str] = []
    for tag_m in _META_TAG_RE.finditer(html):
        tag = tag_m.group(0)
        key_m = _META_ATTR_RE.search(tag)
        if not key_m or key_m.group(1).lower() not in _OG_VIDEO_META_KEYS:
            continue
        content_m = _META_CONTENT_RE.search(tag)
        if content_m and content_m.group(1):
            urls.append(content_m.group(1))
    return urls


def gather_manifest_urls(html: str) -> list[str]:
    """Scans everywhere EXCEPT inside any JSON-typed script block. A JSON-LD
    VideoObject's contentUrl containing a manifest URL is already extracted
    correctly (JSON-decoded) by gather_jsonld_video_urls above — re-scanning
    the SAME script body (or a duplicate copy in a different JSON block —
    see _ANY_JSON_SCRIPT_RE) as raw text hits JSON's own string-escaping
    (ampersands there are literal `\\u0026` backslash-u sequences, not real
    `&` characters) and produces a mangled duplicate. Genuinely inline
    (non-JSON) player configs — this pattern's actual target — never have
    this escaping mismatch."""
    stripped = _ANY_JSON_SCRIPT_RE.sub("", html)
    return [m.group(0) for m in _DIRECT_MANIFEST_RE.finditer(stripped)]


def gather_indirect_video_urls(html: str) -> list[str]:
    """Combined, in a stable order — JSON-LD first (most structured/
    reliable), then OG/twitter meta, then the blunt manifest catch-all.
    Every entry is a raw candidate URL string, not yet deduped/classified —
    that's embed_parser.py's own job."""
    return [
        *gather_jsonld_video_urls(html),
        *gather_meta_video_urls(html),
        *gather_manifest_urls(html),
    ]
