from __future__ import annotations

import re
import sys
from html.parser import HTMLParser

import requests

from ingest.common import Record, write_records

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nitin-search-bot/1.0)"}


class _HTMLToText(HTMLParser):
    BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                  "li", "blockquote", "tr", "br", "hr"}

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        elif tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        elif tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [line.strip() for line in raw.split("\n")]
        lines = [line for line in lines if line]
        return "\n\n".join(lines)


def html_to_text(html: str) -> str:
    parser = _HTMLToText()
    parser.feed(html)
    return parser.text()


def extract_date(html: str) -> str:
    match = re.search(
        r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
        html, re.IGNORECASE,
    )
    if match:
        return match.group(1)
    match = re.search(r'<time[^>]+datetime=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def discover_post_urls(sitemap_url: str) -> list[str]:
    resp = requests.get(sitemap_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return re.findall(r"<loc>(.*?)</loc>", resp.text)


def fetch_post(url: str) -> Record | None:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return None
    html = resp.text
    title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else url
    return Record(
        source="blog",
        title=title,
        reference=url,
        date=extract_date(html),
        text=html_to_text(html),
    )


def ingest_blog(sitemap_url: str, out_path: str) -> int:
    urls = discover_post_urls(sitemap_url)
    records = [r for r in (fetch_post(url) for url in urls) if r is not None]
    write_records(records, out_path)
    return len(records)


if __name__ == "__main__":
    sitemap = sys.argv[1] if len(sys.argv) > 1 else "https://nitinpai.in/sitemap.xml"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/blog.json"
    count = ingest_blog(sitemap, out)
    print(f"Ingested {count} blog posts to {out}")
