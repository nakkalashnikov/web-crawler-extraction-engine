from src.image_filter import filter_images


def test_svg_logo_dropped():
    urls = ["https://site.com/img/logo.abc123.svg"]
    assert filter_images(urls) == []


def test_ico_favicon_dropped():
    urls = ["https://site.com/favicon.ico"]
    assert filter_images(urls) == []


def test_small_square_avatar_dropped_path_segment_pattern():
    """defence24.pl's CDN convention: a `/{W}x{H}px/` path segment."""
    urls = ["https://cdn.site.com/2025/11/05/100x100px/hash.webp"]
    assert filter_images(urls) == []


def test_small_thumbnail_dropped_filename_suffix_pattern():
    """WordPress's convention, seen on rada5.com: a `-{W}x{H}.` filename
    suffix."""
    urls = ["https://site.com/media/photo-120x120.jpg"]
    assert filter_images(urls) == []


def test_real_content_photo_survives():
    urls = ["https://cdn.site.com/2026/06/17/800x450px/hash.webp"]
    assert filter_images(urls) == urls


def test_related_article_thumbnail_dropped_at_500px():
    """Thumbnails with both dimensions <= 500px are dropped."""
    urls = ["https://cdn.site.com/2026/09/28/320x180px/hash.webp"]
    assert filter_images(urls) == []


def test_large_content_photo_survives():
    urls = ["https://cdn.site.com/2026/09/28/800x600px/hash.webp"]
    assert filter_images(urls) == urls


def test_no_dimension_encoded_non_svg_survives():
    urls = ["https://site.com/media/2018/01/photo.jpg"]
    assert filter_images(urls) == urls


def test_order_preserved_with_dedup():
    urls = [
        "https://site.com/media/2018/01/photo.jpg",
        "https://site.com/img/logo.svg",
        "https://site.com/media/2018/01/photo.jpg",
    ]
    assert filter_images(urls) == [
        "https://site.com/media/2018/01/photo.jpg",
    ]


def test_asymmetric_dimensions_survive_proving_and_not_or():
    """100x600: one dim <=500, the other isn't. The rule is BOTH dims
    <=500 (AND), not either (OR) — a regression to OR would wrongly drop
    this."""
    urls = ["https://site.com/media/photo-100x600.jpg"]
    assert filter_images(urls) == urls


def test_boundary_exactly_at_threshold_dropped():
    urls = ["https://site.com/media/photo-500x500.jpg"]
    assert filter_images(urls) == []


def test_boundary_just_over_threshold_survives():
    urls = ["https://site.com/media/photo-501x501.jpg"]
    assert filter_images(urls) == urls


def test_resize_segment_pattern_dropped():
    """defence-ua.com's CDN convention: a `/resize_{W}x{H}/` path segment
    — a third dimension encoding, added after live-testing a real
    defence-ua.com article this session."""
    urls = ["https://en.defence-ua.com/resize_276x205/media/illustration/articles/ab36ab7b49f411f9.jpg"]
    assert filter_images(urls) == []


def test_uppercase_extension_dropped_case_insensitive():
    urls = [
        "https://site.com/img/LOGO.SVG",
        "https://site.com/favicon.ICO",
    ]
    assert filter_images(urls) == []


def test_malformed_input_passes_through_without_crashing():
    urls = ["", "not-a-url-at-all"]
    assert filter_images(urls) == urls


# The 20 real URLs gathered live this session for
# https://defence24.pl/przemysl/himars-do-wszystkiego-i-produkcja-na-bezprecedensowa-skale
# — 16 navbar/site-logo .svg's, 1 avatar-sized (100x100px) thumbnail, 2 (320x180px)
# thumbnails (dropped at 500px), and the 1 real content image (800x450px) that survives.
_DEFENCE24_20_URLS = [
    "https://defence24.pl/img/cyber-defence-24.1573c48b.svg",
    "https://defence24.pl/img/cyber-defence-24.d509c00f.svg",
    "https://defence24.pl/img/energetyka-24.fb936884.svg",
    "https://defence24.pl/img/energetyka-24.b3d13655.svg",
    "https://defence24.pl/img/info-security-24.7e01e050.svg",
    "https://defence24.pl/img/info-security-24.c5765fd8.svg",
    "https://defence24.pl/img/space-24.bfd4266c.svg",
    "https://defence24.pl/img/space-24.be20b0c9.svg",
    "https://defence24.pl/img/defence-24-com.c72254e4.svg",
    "https://defence24.pl/img/defence-24-com.ba30348c.svg",
    "https://defence24.pl/img/defence-academy-24.7b266b04.svg",
    "https://defence24.pl/img/defence-academy-24.2c7aa615.svg",
    "https://cdn.defence24.pl/logos/defence24-light.svg",
    "https://defence24.pl/img/defence-24.f27228dc.svg",
    "https://cdn.defence24.pl/2025/11/05/100x100px/8WXfaWkS6sdrYypNAkyXmqoZY2EX7D1iifgRZLF0.6biq.webp",
    "https://cdn.defence24.pl/2026/06/17/800x450px/gWIgHWuCCgHOBA9320FyVosOiCOfFtRTVjIhRlxd.ymjt.webp",
    "https://cdn.defence24.pl/2026/09/28/320x180px/0ZbAPqQHfbAdwHrJOWUqRVeTdFw5jOEKoTJnG1xA.b1lb.webp",
    "https://cdn.defence24.pl/2026/09/28/320x180px/3tU8kIO3iJXt5aSQSr1LhoVeF92jKFM2JenLhMVu.ml1t.webp",
    "https://defence24.pl/img/karta-mundurowa.e6ee2f25.svg",
    "https://defence24.pl/img/karta-mundurowa.eb11b87d.svg",
]

_DEFENCE24_SURVIVORS = [
    "https://cdn.defence24.pl/2026/06/17/800x450px/gWIgHWuCCgHOBA9320FyVosOiCOfFtRTVjIhRlxd.ymjt.webp",
]


def test_full_defence24_regression_list_real_content_survives():
    assert len(_DEFENCE24_20_URLS) == 20
    result = filter_images(_DEFENCE24_20_URLS)
    assert result == _DEFENCE24_SURVIVORS
    assert len(result) == 1


# The 8 real URLs gathered live this session for a different rada5.com
# article's `images` list.
_RADA5_8_URLS = [
    "https://rada5.com/media/2018/01/Grom-Grim.jpg",
    "https://rada5.com/media/2026/09/photo_2026-09-11_13-39-23-214x140.jpg",
    "https://rada5.com/media/2026/09/Bezymyannyj-214x140.jpg",
    "https://rada5.com/media/2026/07/20260704-03-01tn-214x140.jpeg",
    "https://rada5.com/media/2017/04/logo_1line.png",
    "https://rada5.com/wp-content/plugins/wpfront-scroll-top/images/icons/11.png",
    "https://rada5.com/media/2018/01/Grom-Grim-702x336.jpg",
    "https://rada5.com/wp-content/plugins/wp-polls/images/loading.gif",
]

_RADA5_SURVIVORS = [
    "https://rada5.com/media/2018/01/Grom-Grim.jpg",
    "https://rada5.com/media/2018/01/Grom-Grim-702x336.jpg",
]


def test_full_rada5_regression_list_real_photos_survive():
    """At 500px threshold and with UI element filtering, logos, icons, and loading gif
    are properly dropped, leaving only the real content photos."""
    assert len(_RADA5_8_URLS) == 8
    result = filter_images(_RADA5_8_URLS)
    assert result == _RADA5_SURVIVORS
    assert len(result) == 2


# The 14 real URLs gathered live this session for a defence-ua.com article's
# `images` list — introduces the `/resize_{W}x{H}/` CDN convention.
_DEFENCE_UA_14_URLS = [
    "https://en.defence-ua.com/counter/?id=4049&type=articles",
    "https://en.defence-ua.com/media/userpic/7e45e445211b9159.jpg",
    "https://en.defence-ua.com/media/illustration/articles/974630cee4341a31.jpg",
    "https://en.defence-ua.com/media/contentimages/840630cef2c06a71.jpg",
    "https://en.defence-ua.com/resize_276x205/media/illustration/articles/ab36ab7b49f411f9.jpg",
    "https://en.defence-ua.com/resize_276x205/media/illustration/articles/be56ab7015fef99b.jpg",
    "https://en.defence-ua.com/resize_276x205/media/illustration/articles/3df6ab8b4daae2ec.jpg",
    "https://en.defence-ua.com/resize_292x164/media/illustration/articles/e496aba3df0ea5ff.jpg",
    "https://en.defence-ua.com/resize_292x164/media/illustration/articles/0536aba3acd29fae.jpg",
    "https://en.defence-ua.com/resize_292x164/media/illustration/articles/0bc6ab9ff45c3491.jpg",
    "https://en.defence-ua.com/resize_292x164/media/illustration/articles/c7a6aae47e03976b.jpg",
    "https://en.defence-ua.com/resize_292x164/media/illustration/articles/af26ab7e4d29b67d.jpg",
    "https://en.defence-ua.com/resize_292x164/media/illustration/articles/a846aaee5734e358.jpg",
    "https://en.defence-ua.com/media/contentimages/9cd692e9eeb814ed.jpg",
]

_DEFENCE_UA_SURVIVORS = [
    "https://en.defence-ua.com/media/illustration/articles/974630cee4341a31.jpg",
    "https://en.defence-ua.com/media/contentimages/840630cef2c06a71.jpg",
    "https://en.defence-ua.com/media/contentimages/9cd692e9eeb814ed.jpg",
]


def test_full_defence_ua_regression_list_real_photos_survive():
    """Tracking pixel /counter/ and userpic avatar are dropped, 9 thumbnails dropped,
    3 real content photos survive."""
    assert len(_DEFENCE_UA_14_URLS) == 14
    result = filter_images(_DEFENCE_UA_14_URLS)
    assert result == _DEFENCE_UA_SURVIVORS
    assert len(result) == 3


def test_trackers_and_invalid_formats_dropped():
    urls = [
        "https://informer.yandex.ru/informer/12773563/3_1_FFFFFFFF_EFEFEFFF_0_pageviews",
        "https://mc.yandex.ru/watch/123456",
        "https://site.com/counter/?id=123",
        "https://site.com/pixel.gif",
        "https://site.com/real_photo.jpg",
    ]
    assert filter_images(urls) == ["https://site.com/real_photo.jpg"]


def test_ui_elements_dropped():
    urls = [
        "https://site.com/local/templates/theme/img/search.webp",
        "https://site.com/assets/icons/arrow.png",
        "https://site.com/media/avatars/user1.png",
        "https://site.com/real_photo.jpg",
    ]
    assert filter_images(urls) == ["https://site.com/real_photo.jpg"]


def test_duplicate_filenames_deduplicated():
    urls = [
        "https://cdn.site.com/cache/iblock/123/hash123456789.webp",
        "https://cdn.site.com/resize/580_390/hash123456789.webp",
        "https://cdn.site.com/content/another_photo.jpg",
    ]
    assert filter_images(urls) == [
        "https://cdn.site.com/cache/iblock/123/hash123456789.webp",
        "https://cdn.site.com/content/another_photo.jpg",
    ]

