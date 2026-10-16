from src.page_meta import parse_page_meta

_BASE = "https://news.example.com/2025/a-story"


def test_date_from_meta_beats_json_ld_and_time_tag():
    html = """<head>
      <meta property="article:published_time" content="2025-05-14T09:30:00+02:00">
      <script type="application/ld+json">{"@type":"NewsArticle","datePublished":"2024-01-01"}</script>
    </head><body><time datetime="2023-03-03">x</time></body>"""
    assert parse_page_meta(html, _BASE).published_at == "2025-05-14T09:30:00+02:00"


def test_date_from_nested_json_ld_graph():
    html = '<script type="application/ld+json">{"@graph":[{"@type":"WebSite"},{"@type":"NewsArticle","datePublished":"2025-06-02T10:00:00Z"}]}</script>'
    assert parse_page_meta(html, _BASE).published_at == "2025-06-02T10:00:00Z"


def test_date_falls_back_to_the_time_tag_and_broken_json_ld_is_skipped():
    html = '<script type="application/ld+json">{not json</script><time datetime="2025-07-07">7 July</time>'
    assert parse_page_meta(html, _BASE).published_at == "2025-07-07"


def test_no_date_anywhere_is_none():
    assert parse_page_meta("<html><body><p>hello</p></body></html>", _BASE).published_at is None


def test_og_image_and_alt_are_resolved_against_the_page_url():
    html = '<meta property="og:image" content="/img/cover.jpg"><meta property="og:image:alt" content="The cover">'
    meta = parse_page_meta(html, _BASE)
    assert meta.og_image == "https://news.example.com/img/cover.jpg" and meta.og_image_alt == "The cover"


def test_figcaption_is_mapped_to_the_figures_images():
    html = """<article>
      <figure><img src="/img/a.jpg"><figcaption> A HIMARS   fires
        in Donbas </figcaption></figure>
      <figure><img data-src="https://cdn.example.com/b.jpg"></figure>
    </article>"""
    meta = parse_page_meta(html, _BASE)
    assert meta.captions == {"https://news.example.com/img/a.jpg": "A HIMARS fires in Donbas"}


def test_malformed_html_never_raises():
    assert parse_page_meta("<figure><img src=x><figcaption>cut off", _BASE).published_at is None
