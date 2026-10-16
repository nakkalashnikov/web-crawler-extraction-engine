"""Page metadata read from an article's own HTML — publication date, og:image and the captions of the page's figures.
Pure (stdlib html.parser, no I/O, no new dependency). The HTML is the one extract.py already fetches for the embed scan;
a blocked site (403) simply yields nothing, which is never an error."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin

# meta tags that carry the publication date, most reliable first
_DATE_META_KEYS = (
    "article:published_time", "og:article:published_time", "datepublished", "pubdate", "publish-date",
    "parsely-pub-date", "sailthru.date", "dc.date.issued", "dc.date", "date",
)
_MAX_CAPTION_CHARS = 400


@dataclass
class PageMeta:
    published_at: str | None = None
    og_image:     str | None = None
    og_image_alt: str | None = None
    captions:     dict[str, str] = field(default_factory=dict)   # absolute image URL -> its <figcaption> text


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.json_ld: list[str] = []
        self.time_datetime: str | None = None
        self.figures: list[tuple[list[str], str]] = []   # (image srcs, caption)
        self._in_ld = False
        self._ld_buf: list[str] = []
        self._fig_srcs: list[str] | None = None
        self._in_caption = False
        self._caption: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key and a.get("content") and key not in self.meta:
                self.meta[key] = a["content"].strip()
        elif tag == "script" and "ld+json" in a.get("type", "").lower():
            self._in_ld, self._ld_buf = True, []
        elif tag == "time" and a.get("datetime") and self.time_datetime is None:
            self.time_datetime = a["datetime"].strip()
        elif tag == "figure":
            self._fig_srcs = []
        elif tag in ("img", "source") and self._fig_srcs is not None:
            src = a.get("src") or a.get("data-src") or (a.get("srcset") or "").split(" ")[0]
            if src:
                self._fig_srcs.append(src)
        elif tag == "figcaption" and self._fig_srcs is not None:
            self._in_caption, self._caption = True, []

    def handle_endtag(self, tag):
        if tag == "script" and self._in_ld:
            self._in_ld = False
            self.json_ld.append("".join(self._ld_buf))
        elif tag == "figcaption":
            self._in_caption = False
        elif tag == "figure" and self._fig_srcs is not None:
            caption = " ".join("".join(self._caption).split())
            if self._fig_srcs and caption:
                self.figures.append((self._fig_srcs, caption[:_MAX_CAPTION_CHARS]))
            self._fig_srcs, self._caption = None, []

    def handle_data(self, data):
        if self._in_ld:
            self._ld_buf.append(data)
        elif self._in_caption:
            self._caption.append(data)


def _find_key(node, key: str):
    """First value of `key` anywhere in a parsed JSON-LD tree (dicts and lists, depth first)."""
    if isinstance(node, dict):
        if isinstance(node.get(key), str) and node[key].strip():
            return node[key].strip()
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = _find_key(child, key)
        if found:
            return found
    return None


def _ld_date(blobs: list[str]) -> str | None:
    for blob in blobs:
        try:
            found = _find_key(json.loads(blob), "datePublished")
        except ValueError:
            continue
        if found:
            return found
    return None


def parse_page_meta(html: str, base_url: str) -> PageMeta:
    """Never raises: a page this cannot make sense of gives an empty PageMeta."""
    parser = _Parser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:   # malformed markup — keep whatever was read before it
        pass
    published = next((parser.meta[k] for k in _DATE_META_KEYS if k in parser.meta), None) \
        or _ld_date(parser.json_ld) or parser.time_datetime
    og = parser.meta.get("og:image") or parser.meta.get("og:image:url")
    alt = parser.meta.get("og:image:alt") or parser.meta.get("twitter:image:alt")
    captions: dict[str, str] = {}
    for srcs, caption in parser.figures:
        for src in srcs:
            captions.setdefault(urljoin(base_url, src), caption)
    return PageMeta(
        published_at=published, og_image=urljoin(base_url, og) if og else None,
        og_image_alt=alt, captions=captions,
    )
