"""Honest, strict live contract tests for problematic sources.

These tests assert WHAT THE PIPELINE ACTUALLY NEEDS to succeed:
1. AP News HIMARS article: MUST yield valid video embed (fails due to Cloudflare 403 unless YouTube/mirror fallback is implemented).
2. Facebook post: MUST yield non-empty text/content or video embed (fails if blocked by Meta login wall).
3. Instagram media post: MUST yield images or video embeds (fails if blocked by Instagram login wall).
4. iz.ru article: MUST yield normalized internal video embed (passes with our recent regex fix).
5. X/Twitter post: MUST yield video embed and author/text (passes with our recent oembed/video_embed fix).
"""
from __future__ import annotations

import httpx
import pytest
from curl_cffi.requests import AsyncSession as StealthSession

from src.embed_parser import extract_video_embeds
from src.extract import extract

_TAVILY_URL = "http://localhost:8018"


@pytest.mark.live
@pytest.mark.asyncio
async def test_apnews_video_must_be_extracted():
    """STRICT REQUIREMENT:
    Given an AP News video article about HIMARS, source-extract MUST return
    at least one video embed so the pipeline can verify and place footage.
    """
    url = "https://apnews.com/video/ukraines-us-made-himars-launchers-stay-lethal-despite-drone-filled-skies-4dda918b39154e5c94ba47f09f31b294"
    async with httpx.AsyncClient(timeout=20.0) as http, StealthSession() as stealth:
        res = await extract(http, stealth, _TAVILY_URL, url)
        assert len(res.video_embeds) > 0, (
            "STRICT CONTRACT FAILURE: AP News video was NOT extracted! "
            "Worker returned 0 video embeds due to Cloudflare block."
        )


@pytest.mark.live
@pytest.mark.asyncio
async def test_facebook_post_must_have_content_or_video():
    """STRICT REQUIREMENT:
    Given a Facebook post link from search results, source-extract MUST return
    either non-empty text content (for context) or a video embed. Returning empty
    string blinds the downstream Vision model.
    """
    url = "https://www.facebook.com/FUNKER530/posts/1426737396155644"
    async with httpx.AsyncClient(timeout=15.0) as http, StealthSession() as stealth:
        res = await extract(http, stealth, _TAVILY_URL, url)
        has_media = len(res.video_embeds) > 0 or len(res.images) > 0
        has_text = len(res.content.strip()) > 0
        assert has_media or has_text, (
            "STRICT CONTRACT FAILURE: Facebook post returned completely empty! "
            f"content={len(res.content)} chars, video_embeds={len(res.video_embeds)}, images={len(res.images)}"
        )


@pytest.mark.live
@pytest.mark.asyncio
async def test_instagram_post_must_yield_media():
    """STRICT REQUIREMENT:
    Given an Instagram post link, source-extract MUST extract either images
    or video embeds. Returning 0 media drops the proof candidate completely.
    """
    url = "https://www.instagram.com/p/DdOTw2QgLmu"
    async with httpx.AsyncClient(timeout=15.0) as http, StealthSession() as stealth:
        res = await extract(http, stealth, _TAVILY_URL, url)
        has_media = len(res.video_embeds) > 0 or len(res.images) > 0
        assert has_media, (
            "STRICT CONTRACT FAILURE: Instagram post yielded 0 media items! "
            f"video_embeds={len(res.video_embeds)}, images={len(res.images)}"
        )


@pytest.mark.live
def test_iz_ru_internal_player_embed_detection():
    """STRICT REQUIREMENT:
    iz.ru /video/embed/... iframe MUST be parsed into an absolute direct video URL.
    """
    sample_html = '''
    <div class="video-container">
        <iframe src="/video/embed/1359955#inside" width="100%" height="450"></iframe>
    </div>
    '''
    embeds = extract_video_embeds(sample_html)
    iz_embeds = [e for e in embeds if "1359955" in e.url]
    assert len(iz_embeds) == 1, "iz.ru internal player must be detected"
    assert iz_embeds[0].url == "https://iz.ru/video/embed/1359955#inside"
    assert iz_embeds[0].platform == "direct"


@pytest.mark.live
@pytest.mark.asyncio
async def test_x_twitter_post_must_yield_video_embed():
    """STRICT REQUIREMENT:
    Twitter/X direct post URL MUST be preserved as a video_embed for yt-dlp downstream.
    """
    url = "https://x.com/GeoConfirmed/status/1821951378409070609"
    async with httpx.AsyncClient(timeout=15.0) as http, StealthSession() as stealth:
        res = await extract(http, stealth, _TAVILY_URL, url)
        assert res.kind == "social_direct"
        assert res.platform == "twitter"
        assert len(res.video_embeds) == 1
        assert res.video_embeds[0].url == url
