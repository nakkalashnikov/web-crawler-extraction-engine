"""Pure, cheap ("Tier 1") filter for `SourceExtractResult.images` — the URLs
Tavily's /extract `images` field returns — that drops URLs reliably
identifiable as site chrome (navbar/logo icons, avatar-shaped thumbnails,
tracking pixels, UI templates) from URL structure alone, no HTTP fetch per candidate.

Self-contained like embed_parser.py: module-level compiled regex
constants, pure functions, no external I/O, no config.py dependency.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

# Threshold raised to 500px: real content photos are larger than 500px,
# while thumbnails and related-article crops are <=500px.
_MAX_CHROME_DIMENSION_PX = 500

# defence24.pl's CDN convention: a `/{W}x{H}px/` path segment.
_DIMENSION_PATH_SEGMENT_RE = re.compile(r"/(\d+)x(\d+)px/")
# WordPress's convention, seen on rada5.com: a `-{W}x{H}.` filename suffix.
_DIMENSION_FILENAME_SUFFIX_RE = re.compile(r"-(\d+)x(\d+)\.")
# defence-ua.com's CDN convention: a `/resize_{W}x{H}/` path segment.
_DIMENSION_RESIZE_SEGMENT_RE = re.compile(r"/resize_(\d+)x(\d+)/")
# General / Bitrix CDN convention: e.g. `/{W}_{H}_/` or `/{W}x{H}/`
_DIMENSION_SLASH_RE = re.compile(r"/(\d+)[x_](\d+)(?:_\d+)?/")

_DIMENSION_PATTERNS = (
    _DIMENSION_PATH_SEGMENT_RE,
    _DIMENSION_FILENAME_SUFFIX_RE,
    _DIMENSION_RESIZE_SEGMENT_RE,
    _DIMENSION_SLASH_RE,
)

# Vector and icon file extensions (always dropped)
_CHROME_EXTENSIONS = (".svg", ".ico")

# Valid raster image extensions for article photos
_VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif")

# Trackers, counters, analytics keywords in host or path
_TRACKER_KEYWORDS = (
    "informer.yandex",
    "mc.yandex",
    "google-analytics",
    "top100.rambler",
    "counter",
    "pixel",
    "pageviews",
    "beacon",
    "hit.gif",
)

# UI chrome / template patterns in URL path
_UI_PATH_SEGMENTS = (
    "/templates/",
    "/themes/",
    "/assets/",
    "/icons/",
    "/icon/",
    "/userpic/",
    "/avatars/",
    "/avatar/",
    "/emojis/",
    "/smileys/",
)

# UI file name prefixes or tokens
_UI_FILENAME_PATTERNS = re.compile(
    r"(?:^|[-_./])(search|logo|favicon|arrow|loading|spinner|btn|button)(?:[-_.]|$)",
    re.IGNORECASE,
)


def _is_invalid_format_or_tracker(url: str) -> bool:
    """Drop rule 1: URL is an analytics counter/pixel or does not have a valid raster image extension."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False

    full_url_lower = url.lower()
    for tracker in _TRACKER_KEYWORDS:
        if tracker in full_url_lower:
            return True

    path = parsed.path.lower()
    if path.endswith(_CHROME_EXTENSIONS):
        return True

    # If the URL path contains a file extension (has a dot in the last segment),
    # ensure it is a recognized raster image extension.
    # RESTful CDN image routes without an extension (e.g. Meduza /imgly/share/...) are preserved.
    last_segment = path.rsplit("/", 1)[-1]
    if "." in last_segment:
        if not any(last_segment.endswith(ext) for ext in _VALID_IMAGE_EXTENSIONS):
            return True

    return False


def _is_ui_element(url: str) -> bool:
    """Drop rule 2: URL belongs to site UI chrome (templates, icons, avatars, search icons, logos)."""
    parsed = urlparse(url)
    path = parsed.path.lower()

    for seg in _UI_PATH_SEGMENTS:
        if seg in path:
            return True

    filename = path.rsplit("/", 1)[-1]
    if _UI_FILENAME_PATTERNS.search(filename):
        return True

    return False


def _is_avatar_or_icon_sized(url: str) -> bool:
    """Drop rule 3: URL encodes width/height via a recognized pattern, and
    BOTH dimensions are <= _MAX_CHROME_DIMENSION_PX (500px)."""
    path = urlparse(url).path.lower()
    for pattern in _DIMENSION_PATTERNS:
        m = pattern.search(path)
        if m:
            width, height = int(m.group(1)), int(m.group(2))
            if width <= _MAX_CHROME_DIMENSION_PX and height <= _MAX_CHROME_DIMENSION_PX:
                return True
    return False


def filter_images(urls: list[str]) -> list[str]:
    """Drops navbar/icon/avatar chrome, trackers, invalid formats, and UI elements
    out of a `SourceExtractResult.images` candidate list, and deduplicates identical files.
    Order-preserving."""
    result: list[str] = []
    seen_urls: set[str] = set()
    seen_filenames: set[str] = set()

    for url in urls:
        if url in seen_urls:
            continue

        if not url:
            result.append(url)
            seen_urls.add(url)
            continue

        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            result.append(url)
            seen_urls.add(url)
            continue

        if _is_invalid_format_or_tracker(url):
            continue
        if _is_ui_element(url):
            continue
        if _is_avatar_or_icon_sized(url):
            continue

        path = parsed.path
        filename = path.rsplit("/", 1)[-1].lower()
        if filename:
            if filename in seen_filenames:
                continue
            seen_filenames.add(filename)

        seen_urls.add(url)
        result.append(url)

    return result
