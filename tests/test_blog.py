from unittest.mock import Mock, patch

from ingest.blog import discover_post_urls, extract_date, fetch_post, html_to_text, ingest_blog
from ingest.common import read_records

SAMPLE_HTML = """
<html><head><title>A Sample Post</title>
<meta property="article:published_time" content="2023-05-01T10:00:00Z">
</head>
<body>
<h1>A Sample Post</h1>
<p>This is the <b>first</b> paragraph.</p>
<p>This is the second paragraph.</p>
<script>console.log("should be skipped")</script>
</body></html>
"""

SAMPLE_SITEMAP = """<?xml version="1.0"?>
<urlset>
  <url><loc>https://nitinpai.in/posts/one</loc></url>
  <url><loc>https://nitinpai.in/posts/two</loc></url>
</urlset>
"""


def test_html_to_text_strips_tags_and_scripts():
    text = html_to_text(SAMPLE_HTML)
    assert "first" in text
    assert "second paragraph" in text
    assert "should be skipped" not in text
    assert "<p>" not in text


def test_extract_date_reads_meta_tag():
    assert extract_date(SAMPLE_HTML) == "2023-05-01T10:00:00Z"


def test_extract_date_returns_empty_string_when_absent():
    assert extract_date("<html><body>No date here</body></html>") == ""


def test_discover_post_urls_parses_sitemap():
    with patch("ingest.blog.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=200, text=SAMPLE_SITEMAP)
        urls = discover_post_urls("https://nitinpai.in/sitemap.xml")
    assert urls == ["https://nitinpai.in/posts/one", "https://nitinpai.in/posts/two"]


def test_fetch_post_builds_record():
    with patch("ingest.blog.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=200, text=SAMPLE_HTML)
        record = fetch_post("https://nitinpai.in/posts/one")
    assert record.source == "blog"
    assert record.title == "A Sample Post"
    assert record.reference == "https://nitinpai.in/posts/one"
    assert record.date == "2023-05-01T10:00:00Z"
    assert "first" in record.text


def test_fetch_post_returns_none_on_non_200():
    with patch("ingest.blog.requests.get") as mock_get:
        mock_get.return_value = Mock(status_code=404, text="")
        record = fetch_post("https://nitinpai.in/missing")
    assert record is None


def test_ingest_blog_writes_records(tmp_path):
    out_path = str(tmp_path / "blog.json")
    with patch("ingest.blog.requests.get") as mock_get:
        mock_get.side_effect = [
            Mock(status_code=200, text=SAMPLE_SITEMAP),
            Mock(status_code=200, text=SAMPLE_HTML),
            Mock(status_code=200, text=SAMPLE_HTML),
        ]
        count = ingest_blog("https://nitinpai.in/sitemap.xml", out_path)

    assert count == 2
    records = read_records(out_path)
    assert len(records) == 2
    assert all(r.source == "blog" for r in records)
