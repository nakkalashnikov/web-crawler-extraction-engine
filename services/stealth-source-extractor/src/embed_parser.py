"""Regex-scans an already-fetched page's raw HTML for video/social embeds
that tavily-search's /extract doesn't surface (it extracts article text/
images, not embedded widgets). YouTube/Twitter/Telegram patterns are
generalized from workers/militarnyi/src/embed_parser.py's
extract_video_embeds (verified live there 2026-07-03/04); Instagram and
Facebook are new for this worker.

Deliberately duplicated rather than imported from militarnyi/defence-express
— these are independent worker units (CLAUDE.md invariant #4: duplication
between independent units is preferred over a shared abstraction).
"""
from __future__ import annotations

import html as html_lib
import re
from urllib.parse import parse_qs, unquote, urlparse

from .indirect_video_signals import gather_indirect_video_urls
from .models import Platform, VideoEmbed

_YOUTUBE_EMBED_RE = re.compile(r"youtube\.com/embed/([a-zA-Z0-9_-]{6,})")

# Plain-text links in article body copy (not an iframe/widget at all) — e.g.
# a bare "https://www.youtube.com/watch?v=XXXX" pasted into a paragraph.
# Confirmed live this session: militarnyi-style sites link this way, not
# just embed it.
_YOUTUBE_WATCH_RE = re.compile(r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{6,})")
_YOUTUBE_SHORT_RE = re.compile(r"youtu\.be/([a-zA-Z0-9_-]{6,})")

_TWITTER_IFRAME_SRC_RE = re.compile(
    r'<iframe[^>]+src="([^"]*platform\.twitter\.com/embed/Tweet\.html[^"]*)"'
)
# Plain-text status link: twitter.com/<handle>/status/<id> or x.com/....
# Anchored on /status/ specifically so it never matches a bare profile link
# or the site's own share-intent buttons (twitter.com/intent/tweet?...).
_TWITTER_STATUS_RE = re.compile(
    r"(?:twitter\.com|x\.com)/[A-Za-z0-9_]+/status/(\d+)"
)

# Telegram posts appear either as a widget script (data-telegram-post=
# "channel/123") or as a plain link/embed to t.me/channel/123 — both point
# at the same public post.
_TELEGRAM_WIDGET_RE = re.compile(r'data-telegram-post="([a-zA-Z0-9_]+)/(\d+)"')
_TELEGRAM_LINK_RE   = re.compile(r"t\.me/([a-zA-Z0-9_]+)/(\d+)")

# Instagram's oEmbed-style blockquote widget: <blockquote class=
# "instagram-media" data-instgrm-permalink="https://www.instagram.com/p/
# XXXX/"> — some CMSes serve this URL-encoded in the page source, so the
# caller runs this pattern against both the raw HTML and its
# urllib.parse.unquote()'d form, merging/deduping by URL.
_INSTAGRAM_PERMALINK_RE = re.compile(r'data-instgrm-permalink="([^"]+)"')
# Plain-text post/reel/tv link, e.g. a bare "instagram.com/p/XXXX/" pasted
# into body copy rather than wrapped in the oEmbed blockquote widget.
_INSTAGRAM_PLAIN_RE = re.compile(
    r"instagram\.com/(p|reel|tv)/([A-Za-z0-9_-]+)"
)

# Facebook's XFBML widget form: <div class="fb-post" data-href="...">
# (or "fb-video"); attribute order on the tag is not guaranteed, so this
# matches the whole opening tag first and then searches within it for
# data-href, rather than anchoring class and data-href to a fixed order.
_FB_XFBML_TAG_RE  = re.compile(r'<[^>]+class="[^"]*\bfb-(?:post|video)\b[^"]*"[^>]*>')
_FB_DATA_HREF_RE  = re.compile(r'data-href="([^"]+)"')
# Facebook's iframe-plugin form: src="https://www.facebook.com/plugins/
# post.php?href=<url-encoded-url>&..." — href may be the first query param
# (directly after the literal "?") or a later one (after some "&other=...").
_FB_PLUGIN_HREF_RE = re.compile(r'plugins/(?:post|video)\.php\?(?:[^"\'<>]*&)?href=([^"\'&<>]+)')
# Plain-text video link only — deliberately narrower than a bare
# "facebook.com" substring match, which this story's own live testing
# showed is dominated by footer noise (sharer.php share buttons, a site's
# own facebook.com/PageName link). /watch/?v=<id> and .../videos/<id> are
# specific enough to a real posted video that they don't false-positive on
# that noise.
_FB_WATCH_RE  = re.compile(r"facebook\.com/watch/?\?v=(\d+)")
_FB_VIDEOS_RE = re.compile(r"facebook\.com/([^/\s\"'<>]+)/videos/(\d+)")

# Vimeo: the player iframe form (player.vimeo.com/video/<id>) and a bare
# numeric-id-only plain link — deliberately excludes a bare
# "vimeo.com/<username>" profile link (usernames aren't purely numeric),
# same false-positive guard already used above for Twitter's /status/
# anchor and Facebook's /watch·/videos narrowing.
_VIMEO_PLAYER_RE = re.compile(r"player\.vimeo\.com/video/(\d+)")
_VIMEO_RE        = re.compile(r"vimeo\.com/(\d{6,})")

# Dailymotion: iframe embed, plain watch link, and dai.ly short link.
_DAILYMOTION_EMBED_RE = re.compile(r"dailymotion\.com/embed/video/([a-zA-Z0-9]+)")
_DAILYMOTION_WATCH_RE = re.compile(r"dailymotion\.com/video/([a-zA-Z0-9]+)")
_DAILYMOTION_SHORT_RE = re.compile(r"dai\.ly/([a-zA-Z0-9]+)")

# TikTok — Platform already reserved this value (models.py) but no pattern
# existed yet. oEmbed-style blockquote widget (cite="<full URL>", mirrors
# Instagram's data-instgrm-permalink handling above) plus a plain-text
# profile/video link.
_TIKTOK_BLOCKQUOTE_RE = re.compile(r'<blockquote[^>]+class="[^"]*\btiktok-embed\b[^"]*"[^>]*\bcite="([^"]+)"')
_TIKTOK_PLAIN_RE = re.compile(r"tiktok\.com/(@[\w.-]+)/video/(\d+)")

# CMS-hosted iframe players — lower-confidence, best-effort only (embed-code
# format varies more than the named social platforms above). See
# .claude/scratch/embed-detection-gaps-plan.md's Increment 3 for why these
# are scoped to the iframe-src form only: JW Player's other common
# integration (inline `jwplayer("id").setup({file: "....m3u8"})`) is already
# covered by _DIRECT_MANIFEST_RE below, not duplicated here.
_JWPLAYER_RE   = re.compile(r"(?:cdn\.jwplayer\.com|content\.jwplatform\.com)/players/([a-zA-Z0-9]+-[a-zA-Z0-9]+\.html)")
_BRIGHTCOVE_RE = re.compile(r"players\.brightcove\.net/(\d+)/([a-zA-Z0-9_-]+)_default/index\.html\?videoId=(\d+)")

# iz.ru / Известия internal embedded video player (e.g. /video/embed/1359955#inside)
_IZ_VIDEO_EMBED_RE = re.compile(r'src=["\'](/video/embed/\d+[^"\']*)["\']', re.I)

# Native, fully self-hosted HTML5 video — no third-party platform at all.
# Two shapes: src directly on the <video> tag, or a nested <source src=...>
# child. The block regex is bounded by the first </video> it finds
# (non-greedy), so it can't run on into an unrelated later <video> on the
# same page.
_HTML5_VIDEO_TAG_RE   = re.compile(r"<video\b[^>]*>", re.I)
_HTML5_VIDEO_BLOCK_RE = re.compile(r"<video\b[^>]*>(.*?)</video>", re.I | re.S)
_HTML5_SRC_ATTR_RE    = re.compile(r'\bsrc="([^"]+)"', re.I)
_HTML5_SOURCE_SRC_RE  = re.compile(r'<source\b[^>]*\bsrc="([^"]+)"', re.I)

# JSON-LD VideoObject, OG/twitter meta tags, and the blunt manifest
# catch-all live in indirect_video_signals.py (split out to stay under
# CLAUDE.md's 350-line cap) — gather_indirect_video_urls() returns raw
# candidate URL strings; _add_classified() below re-classifies each one
# through the SAME plain-link patterns used for direct HTML scanning, so
# e.g. a VideoObject.contentUrl that happens to be a YouTube link (live-
# confirmed 2026-09-30 on an AP-mirror site) dedupes correctly against a
# plain-text YouTube link elsewhere on the page instead of being
# double-counted as a second, separately-tagged "direct" entry for the
# same video.
_PLAIN_LINK_CLASSIFIERS: list[tuple[re.Pattern, Platform, "callable"]] = [
    (_YOUTUBE_EMBED_RE, "youtube", lambda m: f"https://www.youtube.com/watch?v={m.group(1)}"),
    (_YOUTUBE_WATCH_RE, "youtube", lambda m: f"https://www.youtube.com/watch?v={m.group(1)}"),
    (_YOUTUBE_SHORT_RE, "youtube", lambda m: f"https://www.youtube.com/watch?v={m.group(1)}"),
    (_TWITTER_STATUS_RE, "twitter", lambda m: f"https://x.com/i/status/{m.group(1)}"),
    (_TELEGRAM_LINK_RE, "telegram", lambda m: f"https://t.me/{m.group(1)}/{m.group(2)}"),
    (_INSTAGRAM_PLAIN_RE, "instagram", lambda m: f"https://www.instagram.com/{m.group(1)}/{m.group(2)}/"),
    (_FB_WATCH_RE, "facebook", lambda m: f"https://www.facebook.com/watch/?v={m.group(1)}"),
    (_FB_VIDEOS_RE, "facebook", lambda m: f"https://www.facebook.com/{m.group(1)}/videos/{m.group(2)}"),
    (_VIMEO_PLAYER_RE, "vimeo", lambda m: f"https://vimeo.com/{m.group(1)}"),
    (_VIMEO_RE, "vimeo", lambda m: f"https://vimeo.com/{m.group(1)}"),
    (_DAILYMOTION_EMBED_RE, "dailymotion", lambda m: f"https://www.dailymotion.com/video/{m.group(1)}"),
    (_DAILYMOTION_WATCH_RE, "dailymotion", lambda m: f"https://www.dailymotion.com/video/{m.group(1)}"),
    (_DAILYMOTION_SHORT_RE, "dailymotion", lambda m: f"https://www.dailymotion.com/video/{m.group(1)}"),
    (_TIKTOK_PLAIN_RE, "tiktok", lambda m: f"https://www.tiktok.com/{m.group(1)}/video/{m.group(2)}"),
]


# html.unescape() is deliberately NOT used for arbitrary third-party URL
# text (only for the narrow, controlled attribute-value extractions
# elsewhere in this file, e.g. the Twitter iframe src): it matches HTML4
# legacy named entities WITHOUT a trailing semicolon, which silently
# corrupts a real-world query string — "...site=news&region=CA..." becomes
# "...site=news®ion=CA..." because "&reg" alone is a valid legacy entity
# for "®". Live-confirmed 2026-09-30: exactly this mangled a real Yahoo
# CDN's .m3u8 URL once JSON-LD/OG detection started passing arbitrary
# third-party query strings through here for the first time (no earlier
# platform pattern ever did — they all build a clean canonical URL from a
# captured id, never re-emit a raw third-party query string). Only `&amp;`
# (the one entity that legitimately appears HTML-attribute-encoded inside a
# real URL) is unescaped here.
_AMP_ENTITY_RE = re.compile(r"&(?:amp|#38|#x26);?", re.I)


def _unescape_amp(url: str) -> str:
    return _AMP_ENTITY_RE.sub("&", url)


def _add(embeds: list[VideoEmbed], seen: set[str], url: str, platform: Platform) -> None:
    url = _unescape_amp(url)
    if url and url not in seen:
        seen.add(url)
        embeds.append(VideoEmbed(url=url, platform=platform))


def _add_classified(embeds: list[VideoEmbed], seen: set[str], url: str) -> None:
    """Adds a URL discovered INDIRECTLY (JSON-LD, OG/twitter meta, or a bare
    manifest URL) — tries each plain-link classifier first so it normalizes/
    dedupes against a same-page direct link to the identical video; falls
    back to platform="direct" when nothing matches (still handed to yt-dlp
    unchanged downstream, see media_gather.py)."""
    url = _unescape_amp(url)
    if not url:
        return
    for pattern, platform, canonicalize in _PLAIN_LINK_CLASSIFIERS:
        m = pattern.search(url)
        if m:
            _add(embeds, seen, canonicalize(m), platform)
            return
    _add(embeds, seen, url, "direct")


def extract_video_embeds(html: str) -> list[VideoEmbed]:
    """Returns order-preserving, de-duplicated video/social embeds found in
    an article page's raw HTML."""
    seen: set[str] = set()
    embeds: list[VideoEmbed] = []

    for m in _YOUTUBE_EMBED_RE.finditer(html):
        _add(embeds, seen, f"https://www.youtube.com/watch?v={m.group(1)}", "youtube")
    for m in _YOUTUBE_WATCH_RE.finditer(html):
        _add(embeds, seen, f"https://www.youtube.com/watch?v={m.group(1)}", "youtube")
    for m in _YOUTUBE_SHORT_RE.finditer(html):
        _add(embeds, seen, f"https://www.youtube.com/watch?v={m.group(1)}", "youtube")

    for m in _TWITTER_IFRAME_SRC_RE.finditer(html):
        src = html_lib.unescape(m.group(1))
        tweet_id = parse_qs(urlparse(src).query).get("id", [None])[0]
        if tweet_id:
            _add(embeds, seen, f"https://x.com/i/status/{tweet_id}", "twitter")
    for m in _TWITTER_STATUS_RE.finditer(html):
        _add(embeds, seen, f"https://x.com/i/status/{m.group(1)}", "twitter")

    for pattern in (_TELEGRAM_WIDGET_RE, _TELEGRAM_LINK_RE):
        for m in pattern.finditer(html):
            channel, post_id = m.groups()
            _add(embeds, seen, f"https://t.me/{channel}/{post_id}", "telegram")

    for variant in (html, unquote(html)):
        for m in _INSTAGRAM_PERMALINK_RE.finditer(variant):
            # unquote() the captured value too (not just the surrounding
            # HTML) — covers both a whole-page-encoded embed AND the rarer
            # case where only the permalink attribute's own value is
            # percent-encoded; either way the two passes converge on the
            # same decoded URL and dedupe correctly.
            _add(embeds, seen, unquote(m.group(1)), "instagram")
    for m in _INSTAGRAM_PLAIN_RE.finditer(html):
        kind, shortcode = m.groups()
        _add(embeds, seen, f"https://www.instagram.com/{kind}/{shortcode}/", "instagram")

    for m in _FB_XFBML_TAG_RE.finditer(html):
        href_m = _FB_DATA_HREF_RE.search(m.group(0))
        if href_m:
            _add(embeds, seen, href_m.group(1), "facebook")

    for m in _FB_PLUGIN_HREF_RE.finditer(html):
        _add(embeds, seen, unquote(m.group(1)), "facebook")

    for m in _FB_WATCH_RE.finditer(html):
        _add(embeds, seen, f"https://www.facebook.com/watch/?v={m.group(1)}", "facebook")
    for m in _FB_VIDEOS_RE.finditer(html):
        page, video_id = m.groups()
        _add(embeds, seen, f"https://www.facebook.com/{page}/videos/{video_id}", "facebook")

    for pattern in (_VIMEO_PLAYER_RE, _VIMEO_RE):
        for m in pattern.finditer(html):
            _add(embeds, seen, f"https://vimeo.com/{m.group(1)}", "vimeo")

    for pattern in (_DAILYMOTION_EMBED_RE, _DAILYMOTION_WATCH_RE, _DAILYMOTION_SHORT_RE):
        for m in pattern.finditer(html):
            _add(embeds, seen, f"https://www.dailymotion.com/video/{m.group(1)}", "dailymotion")

    for m in _TIKTOK_BLOCKQUOTE_RE.finditer(html):
        _add(embeds, seen, m.group(1), "tiktok")
    for m in _TIKTOK_PLAIN_RE.finditer(html):
        _add(embeds, seen, f"https://www.tiktok.com/{m.group(1)}/video/{m.group(2)}", "tiktok")

    for m in _JWPLAYER_RE.finditer(html):
        _add(embeds, seen, f"https://{m.group(0)}", "jwplayer")
    for m in _BRIGHTCOVE_RE.finditer(html):
        _add(embeds, seen, f"https://{m.group(0)}", "brightcove")
    for m in _IZ_VIDEO_EMBED_RE.finditer(html):
        _add(embeds, seen, f"https://iz.ru{m.group(1)}", "direct")

    for tag_m in _HTML5_VIDEO_TAG_RE.finditer(html):
        src_m = _HTML5_SRC_ATTR_RE.search(tag_m.group(0))
        if src_m:
            _add(embeds, seen, src_m.group(1), "html5")
    for block_m in _HTML5_VIDEO_BLOCK_RE.finditer(html):
        src_m = _HTML5_SOURCE_SRC_RE.search(block_m.group(1))
        if src_m:
            _add(embeds, seen, src_m.group(1), "html5")

    # Indirect, player-agnostic signals last (JSON-LD/OG meta/blunt manifest
    # — see indirect_video_signals.py) — every candidate URL they surface
    # goes through _add_classified so it dedupes against whatever the direct
    # scans above already found for the same underlying video.
    for url in gather_indirect_video_urls(html):
        _add_classified(embeds, seen, url)

    return embeds
