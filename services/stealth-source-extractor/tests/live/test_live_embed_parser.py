"""Live regression tests for embed_parser.py's JSON-LD VideoObject detection
— real pages, no mocks, no docker stack required (plain httpx GET, this
module has no other dependency). Run: pytest -m live tests/live/

Both URLs are real, already-confirmed-live pages (checked manually this
session, not hypothetical) — see .claude/scratch/embed-detection-gaps-plan.md's
Audit section for the full story: these are sources S6 and S8 from a real
resolve_proof run on project 69d07fe7-7b3b-5aa6-a212-4adcec03a513, both
syndicated copies of the same AP video that embed_parser.py silently missed
before this change (fell back to unrelated page images instead).

A live test SKIPS rather than fails on a network error (site down, blocked,
markup changed) — these assert OUR detection logic against real-world HTML,
not the third party's uptime."""
from __future__ import annotations

import httpx
import pytest

from src.embed_parser import extract_video_embeds

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"


def _fetch(url: str) -> str:
    try:
        r = httpx.get(url, headers={"User-Agent": _UA}, timeout=20.0, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError as e:
        pytest.skip(f"live fetch failed (network/site issue, not our logic): {e}")
    return r.text


@pytest.mark.live
def test_yahoo_ap_syndicated_article_videoobject_detected():
    """Source S6 from the real run — a Yahoo/Canadian-Press page whose only
    video signal is a schema.org VideoObject with a direct .m3u8 contentUrl,
    no named-platform iframe/link anywhere on the page. Before this change,
    video_embeds came back empty and extraction fell back to 3 unrelated
    images from Yahoo's own "recommended articles" carousel."""
    html = _fetch("https://ca.news.yahoo.com/ukraines-us-made-himars-launchers-153925726.html")
    embeds = extract_video_embeds(html)
    m3u8_embeds = [e for e in embeds if e.url.endswith((".m3u8",)) or ".m3u8?" in e.url]
    assert m3u8_embeds, f"expected an .m3u8 VideoObject embed, got: {embeds}"
    assert m3u8_embeds[0].platform == "direct"


@pytest.mark.live
def test_ap_mirror_videoobject_content_url_reclassified_as_youtube():
    """Source S8 from the real run — a Townnews-CMS AP mirror whose
    VideoObject.contentUrl is itself a plain YouTube link. Must come back
    tagged platform="youtube" (proves the re-classification path actually
    works against real-world markup, not just the synthetic unit-test
    fixture)."""
    html = _fetch(
        "https://lufkindailynews.com/ap_video/us-made-himars-launchers-changed-the-course-of-the-ukraine-war-serviceman-says/video_beaf1791-e5d2-51b9-8282-9838891def9c.html"
    )
    embeds = extract_video_embeds(html)
    youtube_embeds = [e for e in embeds if e.platform == "youtube"]
    assert youtube_embeds, f"expected a YouTube-classified embed, got: {embeds}"
    assert youtube_embeds[0].url == "https://www.youtube.com/watch?v=NRpv05WcW7Q"
