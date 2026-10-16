from src.embed_parser import extract_video_embeds
from src.models import VideoEmbed


def test_youtube_embed_detected():
    html = '<iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ", platform="youtube")]


def test_twitter_iframe_embed_detected():
    html = (
        '<iframe src="https://platform.twitter.com/embed/Tweet.html'
        '?id=1234567890&amp;theme=light"></iframe>'
    )
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://x.com/i/status/1234567890", platform="twitter")]


def test_youtube_plain_watch_link_detected():
    """Bare 'https://www.youtube.com/watch?v=XXXX' pasted into article body
    copy (not an iframe) — confirmed live on a real militarnyi-style article
    this session."""
    html = "<p>Дивіться відео: https://www.youtube.com/watch?v=eon23heDf5k</p>"
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.youtube.com/watch?v=eon23heDf5k", platform="youtube")]


def test_youtube_plain_short_link_detected():
    html = "<p>See https://youtu.be/dQw4w9WgXcQ for details</p>"
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ", platform="youtube")]


def test_youtube_embed_and_plain_link_dedupe_to_one():
    html = (
        '<iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>'
        '<p>https://www.youtube.com/watch?v=dQw4w9WgXcQ</p>'
    )
    embeds = extract_video_embeds(html)
    assert len(embeds) == 1


def test_twitter_plain_status_link_detected():
    html = '<p>see this: https://x.com/someuser/status/1234567890</p>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://x.com/i/status/1234567890", platform="twitter")]


def test_twitter_intent_tweet_link_not_falsely_matched():
    """The share-intent button link (twitter.com/intent/tweet?text=...)
    must not false-positive as a status embed — it has no /status/ segment."""
    html = '<a href="https://twitter.com/intent/tweet?text=hello">Tweet</a>'
    assert extract_video_embeds(html) == []


def test_instagram_plain_post_link_detected():
    html = '<p>https://www.instagram.com/p/ABC123/</p>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.instagram.com/p/ABC123/", platform="instagram")]


def test_instagram_plain_reel_link_detected():
    html = '<p>https://www.instagram.com/reel/XYZ789/</p>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.instagram.com/reel/XYZ789/", platform="instagram")]


def test_facebook_plain_watch_link_detected():
    html = '<p>https://www.facebook.com/watch/?v=999888</p>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.facebook.com/watch/?v=999888", platform="facebook")]


def test_facebook_plain_videos_link_detected():
    html = '<p>https://www.facebook.com/somepage/videos/456123</p>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.facebook.com/somepage/videos/456123", platform="facebook")]


def test_facebook_sharer_link_not_falsely_matched():
    """Footer share-button noise — confirmed live this session (0/20 real
    articles had a genuine embed, every 'facebook.com' hit was noise like
    this) — must not be picked up by the plain-link patterns."""
    html = '<a href="https://www.facebook.com/sharer.php?u=https%3A%2F%2Fexample.com%2Farticle">Share</a>'
    assert extract_video_embeds(html) == []


def test_facebook_plain_page_link_not_falsely_matched():
    html = '<a href="https://www.facebook.com/SomeNewsOutlet">Follow us</a>'
    assert extract_video_embeds(html) == []


def test_telegram_widget_embed_detected():
    html = '<script data-telegram-post="somechannel/42" data-width="100%"></script>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://t.me/somechannel/42", platform="telegram")]


def test_telegram_plain_link_detected():
    html = '<a href="https://t.me/somechannel/42">see post</a>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://t.me/somechannel/42", platform="telegram")]


def test_instagram_permalink_detected_raw():
    html = (
        '<blockquote class="instagram-media" '
        'data-instgrm-permalink="https://www.instagram.com/p/ABC123/"></blockquote>'
    )
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.instagram.com/p/ABC123/", platform="instagram")]


def test_instagram_permalink_found_via_unquote_pass():
    """abcnews.com-style markup: the permalink attribute value itself is
    URL-encoded in the page source — the raw-HTML pass alone would not
    match `data-instgrm-permalink="..."` against the encoded text, but the
    urllib.parse.unquote pass must find it."""
    encoded_permalink = "https%3A%2F%2Fwww.instagram.com%2Fp%2FXYZ789%2F"
    html = f'<blockquote class="instagram-media" data-instgrm-permalink="{encoded_permalink}"></blockquote>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.instagram.com/p/XYZ789/", platform="instagram")]


def test_facebook_xfbml_post_detected():
    html = '<div class="fb-post" data-href="https://www.facebook.com/permalink.php?story_fbid=1&id=2"></div>'
    embeds = extract_video_embeds(html)
    assert embeds == [
        VideoEmbed(url="https://www.facebook.com/permalink.php?story_fbid=1&id=2", platform="facebook"),
    ]


def test_facebook_xfbml_video_detected():
    html = '<div data-href="https://www.facebook.com/watch/?v=999" class="fb-video"></div>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.facebook.com/watch/?v=999", platform="facebook")]


def test_facebook_plugin_iframe_detected():
    html = (
        '<iframe src="https://www.facebook.com/plugins/post.php?'
        'href=https%3A%2F%2Fwww.facebook.com%2Fsomepage%2Fposts%2F123&amp;width=500"></iframe>'
    )
    embeds = extract_video_embeds(html)
    assert embeds == [
        VideoEmbed(url="https://www.facebook.com/somepage/posts/123", platform="facebook"),
    ]


def test_facebook_plugin_video_iframe_detected():
    html = (
        '<iframe src="https://www.facebook.com/plugins/video.php?'
        'href=https%3A%2F%2Fwww.facebook.com%2Fsomepage%2Fvideos%2F456&amp;width=500"></iframe>'
    )
    embeds = extract_video_embeds(html)
    assert embeds == [
        VideoEmbed(url="https://www.facebook.com/somepage/videos/456", platform="facebook"),
    ]


def test_no_embeds_returns_empty_list():
    html = "<html><body><article><p>Just a plain article, no embeds here.</p></article></body></html>"
    assert extract_video_embeds(html) == []


def test_duplicate_embeds_deduplicated():
    html = (
        '<iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>'
        '<iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>'
    )
    embeds = extract_video_embeds(html)
    assert len(embeds) == 1


# ── Vimeo / Dailymotion / TikTok — new platforms, no pattern existed before ──

def test_vimeo_iframe_embed_detected():
    html = '<iframe src="https://player.vimeo.com/video/123456789"></iframe>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://vimeo.com/123456789", platform="vimeo")]


def test_vimeo_plain_link_not_confused_with_username():
    """A bare vimeo.com/<username> profile link must not false-positive —
    usernames aren't purely numeric, same guard class as the existing
    Twitter/Facebook plain-link narrowing above."""
    html = '<a href="https://vimeo.com/someusername">Follow</a>'
    assert extract_video_embeds(html) == []


def test_dailymotion_embed_iframe_detected():
    html = '<iframe src="https://www.dailymotion.com/embed/video/x7abcde"></iframe>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.dailymotion.com/video/x7abcde", platform="dailymotion")]


def test_dailymotion_short_link_detected():
    html = '<p>https://dai.ly/x7abcde</p>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.dailymotion.com/video/x7abcde", platform="dailymotion")]


def test_tiktok_blockquote_embed_detected():
    html = (
        '<blockquote class="tiktok-embed" cite="https://www.tiktok.com/@someuser/video/7123456789">'
        '</blockquote>'
    )
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.tiktok.com/@someuser/video/7123456789", platform="tiktok")]


def test_tiktok_plain_link_detected():
    html = '<p>https://www.tiktok.com/@someuser/video/7123456789</p>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.tiktok.com/@someuser/video/7123456789", platform="tiktok")]


# ── JW Player / Brightcove — CMS-hosted iframe players (best-effort) ────────

def test_jwplayer_cdn_iframe_detected():
    html = '<iframe src="https://cdn.jwplayer.com/players/AbCdEfGh-IjKlMnOp.html"></iframe>'
    embeds = extract_video_embeds(html)
    assert embeds == [
        VideoEmbed(url="https://cdn.jwplayer.com/players/AbCdEfGh-IjKlMnOp.html", platform="jwplayer"),
    ]


def test_jwplayer_jwplatform_iframe_detected():
    html = '<iframe src="https://content.jwplatform.com/players/AbCdEfGh-IjKlMnOp.html"></iframe>'
    embeds = extract_video_embeds(html)
    assert embeds == [
        VideoEmbed(url="https://content.jwplatform.com/players/AbCdEfGh-IjKlMnOp.html", platform="jwplayer"),
    ]


def test_brightcove_iframe_detected():
    html = '<iframe src="https://players.brightcove.net/1234567890/default_default/index.html?videoId=6111222333"></iframe>'
    embeds = extract_video_embeds(html)
    assert embeds == [
        VideoEmbed(
            url="https://players.brightcove.net/1234567890/default_default/index.html?videoId=6111222333",
            platform="brightcove",
        ),
    ]


# ── Native HTML5 <video>/<source> — fully self-hosted, no platform at all ───

def test_html5_video_src_attribute_detected():
    html = '<video src="https://cdn.example.com/clips/event.mp4" controls></video>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://cdn.example.com/clips/event.mp4", platform="html5")]


def test_html5_video_nested_source_detected():
    html = '<video controls><source src="https://cdn.example.com/clips/event.mp4" type="video/mp4"></video>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://cdn.example.com/clips/event.mp4", platform="html5")]


def test_html5_video_block_does_not_run_into_next_video_tag():
    """The block regex must stay bounded to its OWN </video> — a second,
    unrelated <video> later on the page must not leak its <source> into the
    first block's match."""
    html = (
        '<video></video>'
        '<p>unrelated</p>'
        '<video><source src="https://cdn.example.com/second.mp4"></video>'
    )
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://cdn.example.com/second.mp4", platform="html5")]


# ── schema.org VideoObject JSON-LD — universal, player-agnostic signal ──────

def test_jsonld_videoobject_content_url_detected():
    """Live-confirmed 2026-09-30: this exact shape (a direct HLS manifest as
    contentUrl) is what a real Yahoo/AP-syndicated article used — the page
    this gap was originally found on."""
    html = (
        '<script type="application/ld+json">'
        '{"@context":"https://schema.org","@type":"VideoObject",'
        '"name":"Example","contentUrl":"https://video.example.com/stream/abc123.m3u8"}'
        '</script>'
    )
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://video.example.com/stream/abc123.m3u8", platform="direct")]


def test_jsonld_videoobject_nested_under_graph_detected():
    html = (
        '<script type="application/ld+json">'
        '{"@context":"https://schema.org","@graph":['
        '{"@type":"NewsArticle","headline":"Example"},'
        '{"@type":"VideoObject","embedUrl":"https://video.example.com/embed/xyz"}'
        ']}'
        '</script>'
    )
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://video.example.com/embed/xyz", platform="direct")]


def test_jsonld_videoobject_content_url_reclassified_as_youtube():
    """Live-confirmed 2026-09-30 on a real AP-mirror site: a VideoObject's
    contentUrl was itself a plain youtube.com/watch link — this must come
    back tagged platform="youtube" (via the same plain-link classifiers),
    not a separately-tagged "direct" entry for the same video."""
    html = (
        '<script type="application/ld+json">'
        '{"@type":"VideoObject","contentUrl":"https://www.youtube.com/watch?v=NRpv05WcW7Q"}'
        '</script>'
    )
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.youtube.com/watch?v=NRpv05WcW7Q", platform="youtube")]


def test_jsonld_videoobject_dedupes_against_plain_youtube_link_on_same_page():
    html = (
        '<script type="application/ld+json">'
        '{"@type":"VideoObject","contentUrl":"https://www.youtube.com/watch?v=NRpv05WcW7Q"}'
        '</script>'
        '<p>https://www.youtube.com/watch?v=NRpv05WcW7Q</p>'
    )
    embeds = extract_video_embeds(html)
    assert len(embeds) == 1


def test_jsonld_malformed_block_does_not_crash_or_block_the_rest_of_the_page():
    html = (
        '<script type="application/ld+json">{not valid json,,,</script>'
        '<iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>'
    )
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ", platform="youtube")]


def test_jsonld_non_video_object_ignored():
    html = (
        '<script type="application/ld+json">'
        '{"@type":"NewsArticle","headline":"No video here"}'
        '</script>'
    )
    assert extract_video_embeds(html) == []


# ── OpenGraph / Twitter Player Card meta tags ───────────────────────────────

def test_og_video_meta_tag_detected():
    html = '<meta property="og:video" content="https://video.example.com/clip.mp4">'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://video.example.com/clip.mp4", platform="direct")]


def test_og_video_secure_url_meta_tag_detected():
    html = '<meta property="og:video:secure_url" content="https://video.example.com/clip.mp4">'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://video.example.com/clip.mp4", platform="direct")]


def test_twitter_player_stream_meta_tag_detected():
    html = '<meta name="twitter:player:stream" content="https://video.example.com/stream.m3u8">'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://video.example.com/stream.m3u8", platform="direct")]


def test_unrelated_meta_tags_ignored():
    html = (
        '<meta property="og:image" content="https://example.com/thumb.jpg">'
        '<meta name="description" content="An article about something">'
    )
    assert extract_video_embeds(html) == []


# ── Blunt HLS/DASH manifest catch-all ────────────────────────────────────────

def test_bare_m3u8_manifest_url_detected():
    html = '<script>var config = {file: "https://video.example.com/stream/xyz.m3u8?token=abc"};</script>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://video.example.com/stream/xyz.m3u8?token=abc", platform="direct")]


def test_bare_mpd_manifest_url_detected():
    html = '<script>var manifestUrl = "https://video.example.com/dash/xyz.mpd";</script>'
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://video.example.com/dash/xyz.mpd", platform="direct")]


def test_jsonld_content_url_with_ampersand_query_params_not_corrupted():
    """Regression: live-confirmed 2026-09-30 on the real Yahoo page this gap
    was found on — a real CDN query string containing "&region=CA" was
    getting corrupted to "®ion=CA" by html.unescape() treating "&reg"
    (no trailing semicolon) as the legacy HTML4 entity for "®". Only &amp;
    may be unescaped here, nothing else."""
    html = (
        '<script type="application/ld+json">'
        '{"@type":"VideoObject","contentUrl":'
        '"https://video.example.com/stream.m3u8?site=news&region=CA&lang=en"}'
        '</script>'
    )
    embeds = extract_video_embeds(html)
    assert embeds == [
        VideoEmbed(url="https://video.example.com/stream.m3u8?site=news&region=CA&lang=en", platform="direct"),
    ]


def test_manifest_catch_all_does_not_rescan_inside_jsonld_and_mangle_escaped_url():
    """Regression: the blunt manifest catch-all must not re-scan a JSON-LD
    script's raw text — JSON string-escapes an embedded "&" as the literal
    backslash-u-0026 sequence, and scanning that as plain text (rather than
    through json.loads(), which gather_jsonld_video_urls already does
    correctly) produces a second, mangled duplicate of the same URL."""
    html = (
        '<script type="application/ld+json">'
        '{"@type":"VideoObject","contentUrl":'
        '"https://video.example.com/stream.m3u8?a=1\\u0026b=2"}'
        '</script>'
    )
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://video.example.com/stream.m3u8?a=1&b=2", platform="direct")]


def test_manifest_catch_all_ignores_duplicate_in_plain_json_hydration_block():
    """Regression: live-confirmed 2026-09-30 — a real page carried the
    IDENTICAL contentUrl in both an application/ld+json block AND a
    separate application/json client-hydration blob (e.g. a framework's own
    state dump). The manifest catch-all must skip JSON-typed script blocks
    generally, not just ld+json specifically, or it re-mangles the same
    already-correctly-extracted URL a second time."""
    html = (
        '<script type="application/ld+json">'
        '{"@type":"VideoObject","contentUrl":"https://video.example.com/s.m3u8?a=1\\u0026b=2"}'
        '</script>'
        '<script type="application/json" id="__NEXT_DATA__">'
        '{"props":{"contentUrl":"https://video.example.com/s.m3u8?a=1\\u0026b=2"}}'
        '</script>'
    )
    embeds = extract_video_embeds(html)
    assert embeds == [VideoEmbed(url="https://video.example.com/s.m3u8?a=1&b=2", platform="direct")]


def test_bare_mp4_url_not_matched_by_manifest_catch_all():
    """Deliberate: .mp4 is excluded from the blunt catch-all (too noisy as a
    blind whole-page regex — ad creatives, unrelated asset files). A bare
    .mp4 with no JSON-LD/OG signal at all must NOT be picked up here."""
    html = '<script>var adCreative = "https://ads.example.com/banner.mp4";</script>'
    assert extract_video_embeds(html) == []


def test_multiple_platforms_order_preserved():
    html = (
        '<iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>'
        '<a href="https://t.me/somechannel/42">post</a>'
    )
    embeds = extract_video_embeds(html)
    assert [e.platform for e in embeds] == ["youtube", "telegram"]
