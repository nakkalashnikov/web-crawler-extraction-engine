from typing import Literal

from pydantic import BaseModel

# "social_direct" — the request URL itself is a direct link into a fixed
# social/video domain set (Tavily fails on these; classified with zero
# fetches). "article" — an ordinary page, handled via tavily-search's
# /extract (+ this worker's own embed regex pass). "failure" — the
# Tavily+Wayback path never produced usable content.
Kind = Literal["social_direct", "article", "failure"]

# The fixed domain-classification set (extract.py's _SOCIAL_DOMAINS) covers
# the first six; embed_parser.py's regex pass detects all of these plus
# "vimeo"/"dailymotion"/"jwplayer"/"brightcove"/"html5" inside an article's
# own HTML (no direct-link domain classification exists for those five —
# they're only ever embed-detected, never a source's own top-level
# platform). "direct" is used when an indirect signal (JSON-LD VideoObject,
# OG/twitter meta tag, or a bare HLS/DASH manifest URL) resolves to a URL
# that doesn't match any named platform — still handed to yt-dlp unchanged
# downstream (media_gather.py branches on platform for the social_direct/
# telegram path only; the article video_embeds path is already
# platform-agnostic). See .claude/scratch/embed-detection-gaps-plan.md.
Platform = Literal[
    "youtube", "twitter", "telegram", "instagram", "facebook", "tiktok",
    "vimeo", "dailymotion", "jwplayer", "brightcove", "html5", "direct",
]

Reason = Literal[
    "empty_content", "wayback_unavailable", "no_snapshot",
    # New in this story — the network-level failure calling tavily-search
    # itself, distinct from a per-URL failure Tavily reports in its own
    # `failed_results` (that one routes to the Wayback fallback instead).
    "upstream_unavailable",
]

# Which pass produced `content`/`images` for a kind="article" result —
# "live" is tavily-search's /extract on the original URL, "wayback" is the
# same call re-run against an archived snapshot URL.
Source = Literal["live", "wayback"]


class SourceExtractRequest(BaseModel):
    url: str


class VideoEmbed(BaseModel):
    url:      str
    platform: Platform


class SourceExtractResult(BaseModel):
    url:                str
    kind:               Kind
    reason:             Reason | None = None
    # Set only for kind="social_direct".
    platform:           Platform | None = None
    canonical_url:      str | None = None
    # Set only for kind="article" (from tavily-search's /extract).
    title:              str = ""
    content:            str = ""
    # Tier-1-filtered via image_filter.filter_images() — not the raw,
    # unfiltered list tavily-search's /extract returned.
    images:             list[str] = []
    # This worker's own regex pass — always populated (possibly empty) for
    # kind="article"; never the cause of an overall failure on its own.
    video_embeds:       list[VideoEmbed] = []
    source:             Source | None = None
    wayback_timestamp:  str | None = None
    # Read from the article's own HTML (page_meta.py) — empty when that fetch was blocked or the page has none; a
    # Wayback snapshot never carries them. published_at is the page's own string (usually ISO 8601), uninterpreted.
    published_at:       str | None = None
    og_image:           str | None = None
    image_captions:     dict[str, str] = {}   # absolute image URL -> its <figcaption>; og:image's alt is stored under og_image
    # Set only for kind="social_direct" when _fetch_social_metadata
    # exhausted its retries without a 200 — distinguishes "title/content
    # are empty because the post genuinely has none" from "empty because
    # the metadata fetch itself failed" (live-confirmed real-run bug: these
    # were indistinguishable before, and a caption-bearing Instagram reel
    # came back completely empty with zero trace of why). None on success.
    meta_error:         str | None = None
