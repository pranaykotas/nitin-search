# Nitin Personal Archive Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, single-user search tool that lets Nitin Pai ask a question and get back relevant excerpts from his own blog, Obsidian notes, tweets, and sent emails.

**Architecture:** Three sequential stages — per-source ingestion scripts write a common JSON record shape to `data/`, `indexing/build_index.py` chunks and embeds everything into `data/chunks.json` + `data/vectors.npy`, and a local FastAPI app (`search/app.py`) serves both the `/ask` endpoint and a static single-page search UI on `localhost:8000`. No public hosting, no scheduled sync — Nitin re-runs `build_all.py` manually when he wants fresh content indexed.

**Tech Stack:** Python 3.11+, FastAPI + uvicorn, fastembed (local ONNX embeddings, no torch), numpy, PyYAML, google-auth/google-auth-oauthlib/google-api-python-client (Gmail only), pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-nitin-personal-archive-search-design.md`

## Global Constraints

- Everything runs locally on Nitin's machine. No deployment, no public URL, no auth system.
- Embeddings are computed locally via `fastembed` — never send document text to an external embedding API.
- The LLM connector-summary call (`search/llm.py`) is skipped entirely for `sent_mail`-sourced excerpts — email content must never be sent to a third-party API.
- Common record shape across all sources: `{source, title, reference, date, text}` — `source` is one of `"blog" | "notes" | "tweets" | "sent_mail"`.
- Chunking: max 400 words per chunk, 50-word overlap, paragraph-aware splitting — same parameters as the proven ATU pipeline at `~/Projects/frameworks/ask/build_index.py`.
- All ingestion scripts are idempotent and safe to re-run — each overwrites its own `data/*.json` output file.
- Run tests with `PYTHONPATH=. pytest tests/<file> -v` from the repo root.

---

## Task 1: Project scaffolding + shared record schema

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `ingest/__init__.py`
- Create: `ingest/common.py`
- Test: `tests/test_common.py`

**Interfaces:**
- Produces: `Record` dataclass (`source: str, title: str, reference: str, date: str, text: str`), `write_records(records: list[Record], path: str) -> None`, `read_records(path: str) -> list[Record]`

- [ ] **Step 1: Create the scaffolding files**

`requirements.txt`:
```
fastapi
uvicorn[standard]
numpy
requests
pydantic
pytest
httpx
fastembed
pyyaml
google-auth
google-auth-oauthlib
google-api-python-client
```

`.gitignore`:
```
data/
client_secret.json
.env
__pycache__/
*.pyc
.venv/
*.egg-info/
```

`.env.example`:
```
LLM_API_KEY=
LLM_API_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
```

`ingest/__init__.py`: (empty file)

- [ ] **Step 2: Write the failing test**

`tests/test_common.py`:
```python
import json

from ingest.common import Record, read_records, write_records


def test_write_then_read_round_trips(tmp_path):
    records = [
        Record(source="blog", title="A Post", reference="https://example.com/a", date="2024-01-01", text="Hello world"),
        Record(source="notes", title="A Note", reference="notes/a.md", date="2024-01-02", text="Some thoughts"),
    ]
    out_path = str(tmp_path / "records.json")

    write_records(records, out_path)
    loaded = read_records(out_path)

    assert loaded == records


def test_write_records_creates_parent_directories(tmp_path):
    out_path = str(tmp_path / "nested" / "dir" / "records.json")
    write_records([Record(source="blog", title="X", reference="x", date="", text="y")], out_path)

    with open(out_path) as f:
        data = json.load(f)
    assert len(data) == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_common.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest.common'`

- [ ] **Step 4: Write minimal implementation**

`ingest/common.py`:
```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Record:
    source: str
    title: str
    reference: str
    date: str
    text: str


def write_records(records: list[Record], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in records], f)


def read_records(path: str) -> list[Record]:
    with open(path) as f:
        raw = json.load(f)
    return [Record(**r) for r in raw]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_common.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore .env.example ingest/__init__.py ingest/common.py tests/test_common.py
git commit -m "feat: project scaffolding and shared record schema"
```

---

## Task 2: Chunking module

**Files:**
- Create: `indexing/__init__.py`
- Create: `indexing/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: nothing (pure function, no dependency on Task 1)
- Produces: `sub_chunk(text: str, max_words: int = 400, overlap_words: int = 50) -> list[str]`

- [ ] **Step 1: Write the failing test**

`tests/test_chunking.py`:
```python
from indexing.chunking import sub_chunk


def test_short_text_returned_as_single_chunk():
    text = "Just a short paragraph."
    assert sub_chunk(text) == [text]


def test_long_text_is_split_at_paragraph_boundaries():
    para = " ".join(["word"] * 300)
    text = f"{para}\n\n{para}\n\n{para}"

    chunks = sub_chunk(text, max_words=400, overlap_words=50)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.split()) <= 400 + 50


def test_consecutive_chunks_share_overlap_words():
    para = " ".join([f"word{i}" for i in range(300)])
    text = f"{para}\n\n{para}"

    chunks = sub_chunk(text, max_words=300, overlap_words=50)

    assert len(chunks) == 2
    first_tail = chunks[0].split()[-50:]
    second_head = chunks[1].split()[:50]
    assert first_tail == second_head
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'indexing'`

- [ ] **Step 3: Write minimal implementation**

`indexing/__init__.py`: (empty file)

`indexing/chunking.py`:
```python
from __future__ import annotations

MAX_CHUNK_WORDS = 400
OVERLAP_WORDS = 50


def sub_chunk(text: str, max_words: int = MAX_CHUNK_WORDS, overlap_words: int = OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping sub-chunks by word count.

    Short texts are returned as-is. Longer texts are split at paragraph
    boundaries where possible, with word-level fallback.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    paragraphs = text.split("\n\n")
    sub_chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        para_words = para.split()
        if current_words and len(current_words) + len(para_words) > max_words:
            sub_chunks.append(" ".join(current_words))
            overlap = current_words[-overlap_words:] if len(current_words) > overlap_words else []
            current_words = overlap + para_words
        else:
            current_words.extend(para_words)

    if current_words:
        sub_chunks.append(" ".join(current_words))

    return sub_chunks if sub_chunks else [text]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_chunking.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add indexing/__init__.py indexing/chunking.py tests/test_chunking.py
git commit -m "feat: add source-agnostic text chunking"
```

---

## Task 3: Local embeddings module

**Files:**
- Create: `indexing/embed.py`
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: nothing
- Produces: `embed_texts(texts: list[str]) -> np.ndarray` (shape `(len(texts), 384)`, float32, L2-normalized by fastembed)

- [ ] **Step 1: Write the failing test**

`tests/test_embed.py`:
```python
import numpy as np

from indexing.embed import EMBED_DIM, embed_texts


def test_empty_input_returns_empty_array():
    result = embed_texts([])
    assert result.shape == (0, EMBED_DIM)


def test_embeds_texts_to_expected_shape():
    result = embed_texts(["hello world", "a second sentence"])
    assert result.shape == (2, EMBED_DIM)
    assert result.dtype == np.float32


def test_similar_texts_score_higher_than_unrelated_ones():
    vectors = embed_texts([
        "The state budget deficit widened this year",
        "Government fiscal spending increased sharply",
        "My cat sleeps most of the afternoon",
    ])
    sim_related = float(vectors[0] @ vectors[1])
    sim_unrelated = float(vectors[0] @ vectors[2])
    assert sim_related > sim_unrelated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_embed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'indexing.embed'`

- [ ] **Step 3: Write minimal implementation**

`indexing/embed.py`:
```python
from __future__ import annotations

import numpy as np

EMBED_DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64

_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    model = _get_model()
    return np.array(list(model.embed(texts, batch_size=BATCH_SIZE)), dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_embed.py -v`
Expected: PASS (3 tests) — first run downloads the ~90MB model, subsequent runs are fast

- [ ] **Step 5: Commit**

```bash
git add indexing/embed.py tests/test_embed.py
git commit -m "feat: add local fastembed embedding wrapper"
```

---

## Task 4: Blog ingestion

**Files:**
- Create: `ingest/blog.py`
- Test: `tests/test_blog.py`

**Interfaces:**
- Consumes: `Record`, `write_records` from `ingest.common` (Task 1)
- Produces: `html_to_text(html: str) -> str`, `extract_date(html: str) -> str`, `discover_post_urls(sitemap_url: str) -> list[str]`, `fetch_post(url: str) -> Record | None`, `ingest_blog(sitemap_url: str, out_path: str) -> int`

- [ ] **Step 1: Write the failing test**

`tests/test_blog.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_blog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest.blog'`

- [ ] **Step 3: Write minimal implementation**

`ingest/blog.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_blog.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add ingest/blog.py tests/test_blog.py
git commit -m "feat: add blog ingestion via sitemap crawl"
```

---

## Task 5: Obsidian vault ingestion

**Files:**
- Create: `ingest/obsidian.py`
- Test: `tests/test_obsidian.py`

**Interfaces:**
- Consumes: `Record`, `write_records` from `ingest.common` (Task 1)
- Produces: `strip_markdown_syntax(text: str) -> str`, `title_from_content(text: str, fallback: str) -> str`, `ingest_obsidian(vault_path: str, out_path: str) -> int`

- [ ] **Step 1: Write the failing test**

`tests/test_obsidian.py`:
```python
from ingest.common import read_records
from ingest.obsidian import ingest_obsidian, strip_markdown_syntax, title_from_content


def test_strip_markdown_syntax_removes_frontmatter():
    text = "---\ntags: [policy]\ndate: 2024-01-01\n---\n# Title\n\nBody text here."
    result = strip_markdown_syntax(text)
    assert "tags:" not in result
    assert "Body text here." in result


def test_strip_markdown_syntax_resolves_wikilinks():
    text = "See [[Some Other Note]] and [[Other Note|a custom label]] for more."
    result = strip_markdown_syntax(text)
    assert "Some Other Note" in result
    assert "a custom label" in result
    assert "[[" not in result


def test_strip_markdown_syntax_drops_embeds():
    text = "Here is an embed: ![[diagram.png]] and some text after."
    result = strip_markdown_syntax(text)
    assert "![[" not in result
    assert "some text after" in result


def test_title_from_content_uses_first_heading():
    assert title_from_content("# My Real Title\n\nBody", "fallback-name") == "My Real Title"


def test_title_from_content_falls_back_when_no_heading():
    assert title_from_content("Just body text, no heading.", "fallback-name") == "fallback-name"


def test_ingest_obsidian_walks_vault_and_writes_records(tmp_path):
    vault = tmp_path / "vault"
    (vault / "sub").mkdir(parents=True)
    (vault / "note-one.md").write_text("# Note One\n\nSome content about federalism.")
    (vault / "sub" / "note-two.md").write_text("---\ntags: [x]\n---\nNo heading here, just text.")
    (vault / "not-markdown.txt").write_text("should be ignored")

    out_path = str(tmp_path / "notes.json")
    count = ingest_obsidian(str(vault), out_path)

    assert count == 2
    records = read_records(out_path)
    assert all(r.source == "notes" for r in records)
    titles = {r.title for r in records}
    assert "Note One" in titles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_obsidian.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest.obsidian'`

- [ ] **Step 3: Write minimal implementation**

`ingest/obsidian.py`:
```python
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from ingest.common import Record, write_records

FRONTMATTER_PATTERN = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
EMBED_PATTERN = re.compile(r"!\[\[([^\]]+)\]\]")
WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
HEADING_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def strip_markdown_syntax(text: str) -> str:
    text = FRONTMATTER_PATTERN.sub("", text)
    text = EMBED_PATTERN.sub("", text)
    text = WIKILINK_PATTERN.sub(lambda m: m.group(2) or m.group(1), text)
    return text.strip()


def title_from_content(text: str, fallback: str) -> str:
    match = HEADING_PATTERN.search(text)
    return match.group(1).strip() if match else fallback


def ingest_obsidian(vault_path: str, out_path: str) -> int:
    vault = Path(vault_path)
    records = []
    for md_file in sorted(vault.rglob("*.md")):
        raw = md_file.read_text(encoding="utf-8", errors="ignore")
        text = strip_markdown_syntax(raw)
        if not text:
            continue
        title = title_from_content(text, md_file.stem)
        mtime = datetime.fromtimestamp(md_file.stat().st_mtime, tz=timezone.utc).isoformat()
        rel_path = str(md_file.relative_to(vault))
        records.append(Record(source="notes", title=title, reference=rel_path, date=mtime, text=text))
    write_records(records, out_path)
    return len(records)


if __name__ == "__main__":
    vault = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "data/notes.json"
    count = ingest_obsidian(vault, out)
    print(f"Ingested {count} notes to {out}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_obsidian.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add ingest/obsidian.py tests/test_obsidian.py
git commit -m "feat: add Obsidian vault ingestion"
```

---

## Task 6: Twitter archive ingestion

**Files:**
- Create: `ingest/twitter.py`
- Test: `tests/test_twitter.py`

**Interfaces:**
- Consumes: `Record`, `write_records` from `ingest.common` (Task 1)
- Produces: `parse_tweets_js(raw: str) -> list[dict]`, `normalize_date(raw: str) -> str`, `ingest_twitter(tweets_js_path: str, handle: str, out_path: str, include_replies: bool = False) -> int`

- [ ] **Step 1: Write the failing test**

`tests/test_twitter.py`:
```python
import json

from ingest.common import read_records
from ingest.twitter import ingest_twitter, normalize_date, parse_tweets_js

SAMPLE_TWEETS_JS = 'window.YTD.tweets.part0 = ' + json.dumps([
    {"tweet": {
        "id_str": "1001",
        "full_text": "An original thought about fiscal federalism.",
        "created_at": "Wed Oct 10 20:19:24 +0000 2018",
    }},
    {"tweet": {
        "id_str": "1002",
        "full_text": "RT @someone: something they said",
        "created_at": "Wed Oct 11 20:19:24 +0000 2018",
    }},
    {"tweet": {
        "id_str": "1003",
        "full_text": "@someone replying to a thread",
        "created_at": "Wed Oct 12 20:19:24 +0000 2018",
    }},
])


def test_parse_tweets_js_strips_assignment_prefix_and_parses_json():
    entries = parse_tweets_js(SAMPLE_TWEETS_JS)
    assert len(entries) == 3
    assert entries[0]["tweet"]["id_str"] == "1001"


def test_normalize_date_converts_twitter_format_to_iso():
    result = normalize_date("Wed Oct 10 20:19:24 +0000 2018")
    assert result.startswith("2018-10-10")


def test_normalize_date_returns_raw_string_on_unparseable_input():
    assert normalize_date("not a date") == "not a date"


def test_ingest_twitter_excludes_retweets_and_replies_by_default(tmp_path):
    tweets_js_path = tmp_path / "tweets.js"
    tweets_js_path.write_text(SAMPLE_TWEETS_JS)
    out_path = str(tmp_path / "tweets.json")

    count = ingest_twitter(str(tweets_js_path), "nitin", out_path)

    assert count == 1
    records = read_records(out_path)
    assert records[0].text == "An original thought about fiscal federalism."
    assert records[0].source == "tweets"
    assert records[0].reference == "https://x.com/nitin/status/1001"


def test_ingest_twitter_can_include_replies(tmp_path):
    tweets_js_path = tmp_path / "tweets.js"
    tweets_js_path.write_text(SAMPLE_TWEETS_JS)
    out_path = str(tmp_path / "tweets.json")

    count = ingest_twitter(str(tweets_js_path), "nitin", out_path, include_replies=True)

    assert count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_twitter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest.twitter'`

- [ ] **Step 3: Write minimal implementation**

`ingest/twitter.py`:
```python
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

from ingest.common import Record, write_records

ASSIGNMENT_PATTERN = re.compile(r"^\s*window\.YTD\.tweets\.part\d+\s*=\s*", re.MULTILINE)
TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def parse_tweets_js(raw: str) -> list[dict]:
    stripped = ASSIGNMENT_PATTERN.sub("", raw, count=1)
    return json.loads(stripped)


def normalize_date(raw: str) -> str:
    try:
        return datetime.strptime(raw, TWITTER_DATE_FORMAT).isoformat()
    except ValueError:
        return raw


def ingest_twitter(tweets_js_path: str, handle: str, out_path: str, include_replies: bool = False) -> int:
    raw = Path(tweets_js_path).read_text(encoding="utf-8")
    entries = parse_tweets_js(raw)

    records = []
    for entry in entries:
        tweet = entry.get("tweet", entry)
        text = tweet.get("full_text", tweet.get("text", ""))
        if text.startswith("RT @"):
            continue
        if text.startswith("@") and not include_replies:
            continue
        tweet_id = tweet.get("id_str", str(tweet.get("id", "")))
        records.append(Record(
            source="tweets",
            title=text[:60],
            reference=f"https://x.com/{handle}/status/{tweet_id}",
            date=normalize_date(tweet.get("created_at", "")),
            text=text,
        ))

    write_records(records, out_path)
    return len(records)


if __name__ == "__main__":
    tweets_js = sys.argv[1]
    handle = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "data/tweets.json"
    count = ingest_twitter(tweets_js, handle, out)
    print(f"Ingested {count} tweets to {out}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_twitter.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add ingest/twitter.py tests/test_twitter.py
git commit -m "feat: add Twitter archive ingestion"
```

---

## Task 7: Gmail sent-mail ingestion

**Files:**
- Create: `ingest/gmail.py`
- Test: `tests/test_gmail.py`

**Interfaces:**
- Consumes: `Record`, `write_records` from `ingest.common` (Task 1)
- Produces: `strip_quoted_reply(body: str) -> str`, `decode_body(payload: dict) -> str`, `message_to_record(msg: dict) -> Record | None`, `ingest_gmail(token_path: str, client_secret_path: str, out_path: str, max_messages: int = 2000) -> int`

This task's pure helper functions (`strip_quoted_reply`, `decode_body`, `message_to_record`) are fully unit tested. `ingest_gmail` itself drives the real Gmail OAuth flow and API and is verified manually (see Task 12's end-to-end checklist) rather than in the automated suite, since it requires live user consent.

- [ ] **Step 1: Write the failing test**

`tests/test_gmail.py`:
```python
import base64

from ingest.gmail import decode_body, message_to_record, strip_quoted_reply


def test_strip_quoted_reply_cuts_at_on_wrote():
    body = "My actual reply here.\n\nOn Mon, Jan 1, 2024, Someone <x@example.com> wrote:\n> quoted text"
    assert strip_quoted_reply(body) == "My actual reply here."


def test_strip_quoted_reply_cuts_at_original_message_marker():
    body = "My reply.\n\n-----Original Message-----\nFrom: someone"
    assert strip_quoted_reply(body) == "My reply."


def test_strip_quoted_reply_returns_full_body_when_no_marker():
    body = "Just a plain reply with no quoting."
    assert strip_quoted_reply(body) == body


def test_decode_body_handles_plain_text_part():
    text = "Hello, this is the email body."
    encoded = base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")
    payload = {"parts": [{"mimeType": "text/plain", "body": {"data": encoded}}]}
    assert decode_body(payload) == text


def test_decode_body_handles_single_part_message():
    text = "Simple non-multipart body."
    encoded = base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")
    payload = {"body": {"data": encoded}}
    assert decode_body(payload) == text


def test_message_to_record_filters_short_messages():
    short_text = "Too short."
    encoded = base64.urlsafe_b64encode(short_text.encode("utf-8")).decode("ascii")
    msg = {
        "id": "abc123",
        "payload": {
            "headers": [{"name": "Subject", "value": "Re: quick note"}, {"name": "Date", "value": "Mon, 1 Jan 2024 10:00:00 +0000"}],
            "body": {"data": encoded},
        },
    }
    assert message_to_record(msg) is None


def test_message_to_record_builds_record_for_substantive_message():
    long_text = " ".join(["word"] * 200)
    encoded = base64.urlsafe_b64encode(long_text.encode("utf-8")).decode("ascii")
    msg = {
        "id": "abc123",
        "payload": {
            "headers": [{"name": "Subject", "value": "Thoughts on federalism"}, {"name": "Date", "value": "Mon, 1 Jan 2024 10:00:00 +0000"}],
            "body": {"data": encoded},
        },
    }
    record = message_to_record(msg)
    assert record is not None
    assert record.source == "sent_mail"
    assert record.title == "Thoughts on federalism"
    assert record.reference == "gmail:abc123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_gmail.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest.gmail'`

- [ ] **Step 3: Write minimal implementation**

`ingest/gmail.py`:
```python
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from ingest.common import Record, write_records

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MIN_WORDS = 150
QUOTE_MARKERS = ("\nOn ", "\n> ", "\n-----Original Message-----", "\nFrom: ")


def strip_quoted_reply(body: str) -> str:
    cut_points = [body.find(marker) for marker in QUOTE_MARKERS if marker in body]
    cut_points = [p for p in cut_points if p > 0]
    if cut_points:
        body = body[: min(cut_points)]
    return body.strip()


def decode_body(payload: dict) -> str:
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        for part in payload["parts"]:
            text = decode_body(part)
            if text:
                return text
        return ""
    data = payload.get("body", {}).get("data")
    if not data:
        return ""
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")


def message_to_record(msg: dict) -> Record | None:
    headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
    subject = headers.get("Subject", "(no subject)")
    date = headers.get("Date", "")
    body = strip_quoted_reply(decode_body(msg["payload"]))
    if len(body.split()) < MIN_WORDS:
        return None
    return Record(source="sent_mail", title=subject, reference=f"gmail:{msg['id']}", date=date, text=body)


def get_credentials(token_path: str, client_secret_path: str):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=0)
        Path(token_path).write_text(creds.to_json())
    return creds


def ingest_gmail(token_path: str, client_secret_path: str, out_path: str, max_messages: int = 2000) -> int:
    from googleapiclient.discovery import build

    creds = get_credentials(token_path, client_secret_path)
    service = build("gmail", "v1", credentials=creds)

    records = []
    request = service.users().messages().list(userId="me", q="in:sent", maxResults=min(500, max_messages))
    while request is not None and len(records) < max_messages:
        response = request.execute()
        for item in response.get("messages", []):
            msg = service.users().messages().get(userId="me", id=item["id"], format="full").execute()
            record = message_to_record(msg)
            if record:
                records.append(record)
        request = service.users().messages().list_next(request, response)

    write_records(records, out_path)
    return len(records)


if __name__ == "__main__":
    token_path = sys.argv[1] if len(sys.argv) > 1 else "data/gmail_token.json"
    client_secret_path = sys.argv[2] if len(sys.argv) > 2 else "client_secret.json"
    out = sys.argv[3] if len(sys.argv) > 3 else "data/sent_mail.json"
    count = ingest_gmail(token_path, client_secret_path, out)
    print(f"Ingested {count} sent emails to {out}")
```

Note: `get_credentials` and `ingest_gmail` import `google.*` libraries inside the function body, not at module top level, so the pure helper functions above can be imported and unit tested without those packages needing to be importable in every environment — though they're still listed in `requirements.txt` from Task 1 and are needed for the real ingestion run.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_gmail.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add ingest/gmail.py tests/test_gmail.py
git commit -m "feat: add Gmail sent-mail ingestion with quote-stripping and length filter"
```

---

## Task 8: Index builder

**Files:**
- Create: `indexing/build_index.py`
- Test: `tests/test_build_index.py`

**Interfaces:**
- Consumes: `Record`, `read_records` from `ingest.common` (Task 1); `sub_chunk` from `indexing.chunking` (Task 2); `embed_texts` from `indexing.embed` (Task 3)
- Produces: `SOURCE_FILES: dict[str, str]`, `chunks_from_records(records: list[Record]) -> list[dict]`, `build_and_save(source_files: dict[str, str], chunks_out_path: str, vectors_out_path: str) -> int`

- [ ] **Step 1: Write the failing test**

`tests/test_build_index.py`:
```python
import json

import numpy as np

from ingest.common import Record, write_records
from indexing.build_index import build_and_save, chunks_from_records


def test_chunks_from_records_produces_one_chunk_for_short_text():
    records = [Record(source="blog", title="A Post", reference="https://x.com/a", date="2024-01-01", text="Short body.")]
    chunks = chunks_from_records(records)
    assert len(chunks) == 1
    assert chunks[0]["source"] == "blog"
    assert chunks[0]["text"] == "Short body."
    assert "A Post" in chunks[0]["embed_text"]


def test_chunks_from_records_splits_long_text():
    long_text = "\n\n".join([" ".join(["word"] * 300)] * 3)
    records = [Record(source="notes", title="Big Note", reference="n.md", date="", text=long_text)]
    chunks = chunks_from_records(records)
    assert len(chunks) >= 2


def test_build_and_save_combines_multiple_sources(tmp_path, monkeypatch):
    def fake_embed_texts(texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr("indexing.build_index.embed_texts", fake_embed_texts)

    blog_path = str(tmp_path / "blog.json")
    notes_path = str(tmp_path / "notes.json")
    write_records([Record(source="blog", title="P", reference="u", date="", text="Blog text.")], blog_path)
    write_records([Record(source="notes", title="N", reference="n.md", date="", text="Note text.")], notes_path)

    chunks_out = str(tmp_path / "chunks.json")
    vectors_out = str(tmp_path / "vectors.npy")
    source_files = {"blog": blog_path, "notes": notes_path, "tweets": str(tmp_path / "missing.json")}

    count = build_and_save(source_files, chunks_out, vectors_out)

    assert count == 2
    with open(chunks_out) as f:
        chunks = json.load(f)
    assert {c["source"] for c in chunks} == {"blog", "notes"}
    vectors = np.load(vectors_out)
    assert vectors.shape == (2, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_build_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'indexing.build_index'`

- [ ] **Step 3: Write minimal implementation**

`indexing/build_index.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ingest.common import Record, read_records
from indexing.chunking import sub_chunk
from indexing.embed import embed_texts

SOURCE_FILES = {
    "blog": "data/blog.json",
    "notes": "data/notes.json",
    "tweets": "data/tweets.json",
    "sent_mail": "data/sent_mail.json",
}


def chunks_from_records(records: list[Record]) -> list[dict]:
    chunks = []
    for record in records:
        for sub_text in sub_chunk(record.text):
            chunks.append({
                "source": record.source,
                "title": record.title,
                "reference": record.reference,
                "date": record.date,
                "text": sub_text,
                "embed_text": f"{record.title}\n\n{sub_text}",
            })
    return chunks


def build_and_save(source_files: dict[str, str], chunks_out_path: str, vectors_out_path: str) -> int:
    all_records: list[Record] = []
    for path in source_files.values():
        if Path(path).exists():
            all_records.extend(read_records(path))

    chunks = chunks_from_records(all_records)
    texts = [c["embed_text"] for c in chunks]
    vectors = embed_texts(texts)

    Path(chunks_out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_out_path, "w") as f:
        json.dump(chunks, f)
    np.save(vectors_out_path, vectors)

    return len(chunks)


if __name__ == "__main__":
    count = build_and_save(SOURCE_FILES, "data/chunks.json", "data/vectors.npy")
    print(f"Built {count} chunks")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_build_index.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add indexing/build_index.py tests/test_build_index.py
git commit -m "feat: add multi-source index builder"
```

---

## Task 9: Retrieval with source-diverse ranking

**Files:**
- Create: `search/__init__.py`
- Create: `search/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: nothing beyond numpy/json (standalone module, same as `~/Projects/frameworks/ask/retrieval.py`)
- Produces: `load_index(chunks_path: str, vectors_path: str) -> tuple[list[dict], np.ndarray]`, `search_diverse(query_vector, vectors, chunks, top_k=10, min_floor=1, max_per_source=6, min_similarity=0.25) -> list[tuple[int, float]]`, `group_by_source(chunks: list[dict], matches: list[tuple[int, float]]) -> dict[str, list[dict]]`

- [ ] **Step 1: Write the failing test**

`tests/test_retrieval.py`:
```python
import json

import numpy as np

from search.retrieval import group_by_source, load_index, search_diverse


def _unit(vec):
    arr = np.array(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_load_index_reads_chunks_and_vectors(tmp_path):
    chunks = [{"source": "blog", "title": "A", "text": "x"}]
    chunks_path = tmp_path / "chunks.json"
    vectors_path = tmp_path / "vectors.npy"
    chunks_path.write_text(json.dumps(chunks))
    np.save(vectors_path, np.ones((1, 4), dtype=np.float32))

    loaded_chunks, vectors = load_index(str(chunks_path), str(vectors_path))

    assert loaded_chunks == chunks
    assert vectors.shape == (1, 4)


def test_search_diverse_ranks_by_similarity():
    chunks = [
        {"source": "blog", "text": "a"},
        {"source": "blog", "text": "b"},
        {"source": "notes", "text": "c"},
    ]
    vectors = np.array([_unit([1, 0]), _unit([0.9, 0.1]), _unit([0, 1])])
    query = _unit([1, 0])

    matches = search_diverse(query, vectors, chunks, top_k=3, min_similarity=0.0)

    assert matches[0][0] == 0
    assert matches[0][1] >= matches[1][1]


def test_search_diverse_caps_dominant_source():
    chunks = [{"source": "blog", "text": str(i)} for i in range(10)]
    vectors = np.tile(_unit([1, 0]), (10, 1))
    query = _unit([1, 0])

    matches = search_diverse(query, vectors, chunks, top_k=10, max_per_source=6, min_similarity=0.0)

    assert len(matches) == 6


def test_search_diverse_backfills_floor_for_minority_source():
    chunks = (
        [{"source": "blog", "text": str(i)} for i in range(9)]
        + [{"source": "notes", "text": "n"}]
    )
    vectors = np.vstack([
        np.tile(_unit([1, 0]), (9, 1)),
        _unit([0.5, 0.5]).reshape(1, 2),
    ])
    query = _unit([1, 0])

    matches = search_diverse(query, vectors, chunks, top_k=9, min_floor=1, min_similarity=0.0)

    matched_sources = {chunks[idx]["source"] for idx, _ in matches}
    assert "notes" in matched_sources


def test_group_by_source_groups_and_attaches_score():
    chunks = [{"source": "blog", "text": "a"}, {"source": "notes", "text": "b"}]
    grouped = group_by_source(chunks, [(0, 0.9), (1, 0.7)])

    assert set(grouped.keys()) == {"blog", "notes"}
    assert grouped["blog"][0]["score"] == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'search'`

- [ ] **Step 3: Write minimal implementation**

`search/__init__.py`: (empty file)

`search/retrieval.py`:
```python
from __future__ import annotations

import json

import numpy as np


def load_index(chunks_path: str, vectors_path: str) -> tuple[list[dict], np.ndarray]:
    with open(chunks_path) as f:
        chunks = json.load(f)
    vectors = np.load(vectors_path)
    return chunks, vectors


def search_diverse(
    query_vector: np.ndarray,
    vectors: np.ndarray,
    chunks: list[dict],
    top_k: int = 10,
    min_floor: int = 1,
    max_per_source: int = 6,
    min_similarity: float = 0.25,
) -> list[tuple[int, float]]:
    """Return the top_k matches by relevance, capping any single source at
    max_per_source and backfilling a floor of min_floor for sources that
    would otherwise be shut out entirely."""
    if vectors.shape[0] == 0:
        return []
    sims = vectors @ query_vector
    order = [i for i in np.argsort(-sims) if sims[i] >= min_similarity]

    source_counts: dict[str, int] = {}
    results: list[tuple[int, float]] = []
    for i in order:
        if len(results) >= top_k:
            break
        source = chunks[int(i)].get("source", "?")
        if source_counts.get(source, 0) >= max_per_source:
            continue
        source_counts[source] = source_counts.get(source, 0) + 1
        results.append((int(i), float(sims[i])))

    for i in order:
        source = chunks[int(i)].get("source", "?")
        if source_counts.get(source, 0) >= min_floor:
            continue
        if any(idx == int(i) for idx, _ in results):
            continue
        source_counts[source] = source_counts.get(source, 0) + 1
        results.append((int(i), float(sims[i])))

    results.sort(key=lambda x: -x[1])
    return results


def group_by_source(chunks: list[dict], matches: list[tuple[int, float]]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for idx, score in matches:
        chunk = {**chunks[idx], "score": score}
        groups.setdefault(chunk["source"], []).append(chunk)
    return groups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_retrieval.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add search/__init__.py search/retrieval.py tests/test_retrieval.py
git commit -m "feat: add source-diverse relevance ranking"
```

---

## Task 10: LLM connector summaries

**Files:**
- Create: `search/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: nothing beyond `requests` and env vars
- Produces: `generate_connector(source: str, question: str, excerpts: list[str]) -> str | None`

- [ ] **Step 1: Write the failing test**

`tests/test_llm.py`:
```python
from unittest.mock import Mock, patch

from search import llm


def test_generate_connector_returns_none_without_api_key(monkeypatch):
    monkeypatch.setattr(llm, "API_KEY", "")
    result = llm.generate_connector("blog", "a question", ["an excerpt"])
    assert result is None


def test_generate_connector_returns_model_text_on_success(monkeypatch):
    monkeypatch.setattr(llm, "API_KEY", "fake-key")
    fake_response = Mock()
    fake_response.raise_for_status = Mock()
    fake_response.json.return_value = {"choices": [{"message": {"content": "  A summary line.  "}}]}

    with patch("search.llm.requests.post", return_value=fake_response) as mock_post:
        result = llm.generate_connector("blog", "a question", ["an excerpt"])

    assert result == "A summary line."
    assert mock_post.called


def test_generate_connector_returns_none_on_request_failure(monkeypatch):
    monkeypatch.setattr(llm, "API_KEY", "fake-key")
    import requests as requests_module

    with patch("search.llm.requests.post", side_effect=requests_module.RequestException("boom")):
        result = llm.generate_connector("blog", "a question", ["an excerpt"])

    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'search.llm'`

- [ ] **Step 3: Write minimal implementation**

`search/llm.py`:
```python
from __future__ import annotations

import os

import requests

API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = (
    "You summarise what Nitin has written, using ONLY the excerpts given "
    "below. Write one or two sentences. Do not add claims, examples, or "
    "reasoning that is not directly present in the excerpts."
)


def generate_connector(source: str, question: str, excerpts: list[str]) -> str | None:
    """Write a short connector line grounded in the given excerpts.

    Returns None if no API key is configured or the request fails for any
    reason — callers must treat None as "show excerpts without a connector
    line," never as an error to surface to the user.
    """
    if not API_KEY:
        return None

    excerpt_block = "\n\n".join(f"- {e}" for e in excerpts)
    user_prompt = (
        f"Question: {question}\n\n"
        f"Excerpts from {source}:\n{excerpt_block}\n\n"
        f"Write the one-to-two sentence connector line now."
    )

    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 120,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError):
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_llm.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add search/llm.py tests/test_llm.py
git commit -m "feat: add LLM connector-summary generation"
```

---

## Task 11: FastAPI app with Gmail-privacy rule

**Files:**
- Create: `search/app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `load_index`, `search_diverse`, `group_by_source` from `search.retrieval` (Task 9); `generate_connector` from `search.llm` (Task 10); `embed_texts` from `indexing.embed` (Task 3)
- Produces: `app` (FastAPI instance) with `POST /ask` and `GET /` routes

- [ ] **Step 1: Write the failing test**

`tests/test_app.py`:
```python
import json
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    chunks = [
        {"source": "blog", "title": "A Post", "reference": "https://x.com/a", "date": "2024-01-01", "text": "About fiscal federalism."},
        {"source": "sent_mail", "title": "An Email", "reference": "gmail:1", "date": "2024-01-02", "text": "Also about fiscal federalism."},
    ]
    chunks_path = tmp_path / "chunks.json"
    vectors_path = tmp_path / "vectors.npy"
    chunks_path.write_text(json.dumps(chunks))
    np.save(vectors_path, np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32))

    monkeypatch.setenv("CHUNKS_PATH", str(chunks_path))
    monkeypatch.setenv("VECTORS_PATH", str(vectors_path))
    monkeypatch.setenv("MIN_SIMILARITY", "0.0")

    from search import app as app_module
    app_module._index_cache = None

    with patch("search.app.embed_texts", return_value=np.array([[1.0, 0.0]], dtype=np.float32)), \
         patch("search.app.generate_connector", return_value="A generated summary.") as mock_connector:
        yield TestClient(app_module.app), mock_connector


def test_ask_returns_groups_for_matching_question(client):
    test_client, _ = client
    response = test_client.post("/ask", json={"question": "fiscal federalism"})
    assert response.status_code == 200
    body = response.json()
    sources = {g["source"] for g in body["groups"]}
    assert sources == {"blog", "sent_mail"}


def test_ask_skips_connector_for_sent_mail_source(client):
    test_client, mock_connector = client
    test_client.post("/ask", json={"question": "fiscal federalism"})
    called_sources = {call.args[0] for call in mock_connector.call_args_list}
    assert "sent_mail" not in called_sources
    assert "blog" in called_sources


def test_ask_sent_mail_group_has_no_connector_text(client):
    test_client, _ = client
    response = test_client.post("/ask", json={"question": "fiscal federalism"})
    body = response.json()
    sent_mail_group = next(g for g in body["groups"] if g["source"] == "sent_mail")
    assert sent_mail_group["connector_text"] is None


def test_ask_empty_question_returns_error(client):
    test_client, _ = client
    response = test_client.post("/ask", json={"question": "   "})
    assert response.json()["error"] == "empty_question"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'search.app'`

- [ ] **Step 3: Write minimal implementation**

`search/app.py`:
```python
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from indexing.embed import embed_texts
from search.llm import generate_connector
from search.retrieval import group_by_source, load_index, search_diverse

NO_CONNECTOR_SOURCES = {"sent_mail"}

app = FastAPI()

_index_cache: dict | None = None


def get_index() -> dict:
    global _index_cache
    if _index_cache is None:
        chunks_path = os.environ.get("CHUNKS_PATH", "data/chunks.json")
        vectors_path = os.environ.get("VECTORS_PATH", "data/vectors.npy")
        chunks, vectors = load_index(chunks_path, vectors_path)
        _index_cache = {"chunks": chunks, "vectors": vectors}
    return _index_cache


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        return {"error": "empty_question", "message": "Ask a question first."}

    top_k = int(os.environ.get("TOP_K", "10"))
    min_similarity = float(os.environ.get("MIN_SIMILARITY", "0.20"))

    index = get_index()
    query_vector = embed_texts([question])[0]
    matches = search_diverse(query_vector, index["vectors"], index["chunks"], top_k=top_k, min_similarity=min_similarity)

    if not matches:
        return {"groups": [], "message": "Nothing in the archive matches this yet."}

    grouped = group_by_source(index["chunks"], matches)
    groups = []
    for source, items in grouped.items():
        excerpts = [item["text"] for item in items]
        connector = None
        if source not in NO_CONNECTOR_SOURCES:
            connector = generate_connector(source, question, excerpts)
        groups.append({
            "source": source,
            "connector_text": connector,
            "excerpts": [
                {"text": item["text"], "title": item["title"], "reference": item["reference"], "date": item["date"]}
                for item in items
            ],
        })

    return {"groups": groups, "message": None}


STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index_page():
        return FileResponse(str(STATIC_DIR / "index.html"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_app.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add search/app.py tests/test_app.py
git commit -m "feat: add FastAPI /ask endpoint with Gmail-privacy connector rule"
```

---

## Task 12: Static search UI

**Files:**
- Create: `search/static/index.html`
- Create: `search/static/finder.js`
- Create: `search/static/styles.css`

**Interfaces:**
- Consumes: `POST /ask` from `search/app.py` (Task 11)
- Produces: a browser-facing single page — no automated test; verified manually per the checklist in Step 3

- [ ] **Step 1: Write the HTML shell**

`search/static/index.html`:
```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Nitin's Archive Search</title>
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
  <main class="page">
    <h1>Search the archive</h1>
    <p class="subtitle">Ask a question. Get back what you've actually written or said about it — from the blog, notes, tweets, and sent mail.</p>

    <div class="search-row">
      <input type="text" id="search-input" placeholder="e.g. how should India think about data localisation?" />
      <button id="search-btn">Search</button>
    </div>

    <div id="results"></div>
  </main>
  <script src="/static/finder.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write the frontend logic**

`search/static/finder.js`:
```javascript
(function () {
  var SOURCE_LABELS = { blog: "Blog", notes: "Notes", tweets: "Tweets", sent_mail: "Sent Mail" };

  function escapeHtml(str) {
    var div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function truncate(text, maxLen) {
    if (text.length <= maxLen) return text;
    return text.slice(0, maxLen).replace(/\s+\S*$/, "") + "…";
  }

  function formatDate(iso) {
    var d = new Date(iso);
    if (isNaN(d)) return iso || "";
    return d.toLocaleDateString("en-GB", { year: "numeric", month: "short", day: "numeric" });
  }

  function renderResults(data) {
    var container = document.getElementById("results");
    if (data.error) {
      container.innerHTML = '<p class="error">' + escapeHtml(data.message) + "</p>";
      return;
    }
    if (!data.groups || data.groups.length === 0) {
      container.innerHTML = '<p class="empty">' + escapeHtml(data.message || "Nothing found.") + "</p>";
      return;
    }

    var html = "";
    for (var g = 0; g < data.groups.length; g++) {
      var group = data.groups[g];
      html += '<div class="group">';
      html += '<div class="group-label">' + escapeHtml(SOURCE_LABELS[group.source] || group.source) + "</div>";
      if (group.connector_text) {
        html += '<p class="connector">' + escapeHtml(group.connector_text) + "</p>";
      }
      html += '<div class="excerpts">';
      for (var i = 0; i < group.excerpts.length; i++) {
        var ex = group.excerpts[i];
        var isLink = ex.reference && ex.reference.indexOf("http") === 0;
        var metaContent = escapeHtml(ex.title) + " · " + escapeHtml(formatDate(ex.date));
        html += '<div class="excerpt">';
        html += '<p class="excerpt-text">' + escapeHtml(truncate(ex.text, 400)) + "</p>";
        if (isLink) {
          html += '<a class="excerpt-meta" href="' + escapeHtml(ex.reference) + '" target="_blank" rel="noopener">' + metaContent + "</a>";
        } else {
          html += '<div class="excerpt-meta">' + metaContent + "</div>";
        }
        html += "</div>";
      }
      html += "</div></div>";
    }
    container.innerHTML = html;
  }

  function doSearch() {
    var input = document.getElementById("search-input");
    var question = input.value.trim();
    if (!question) return;

    var container = document.getElementById("results");
    container.innerHTML = '<p class="loading">Searching…</p>';

    fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question }),
    })
      .then(function (r) { return r.json(); })
      .then(renderResults)
      .catch(function () {
        container.innerHTML = '<p class="error">Something went wrong — try again.</p>';
      });
  }

  function init() {
    var btn = document.getElementById("search-btn");
    var input = document.getElementById("search-input");
    btn.addEventListener("click", doSearch);
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") doSearch();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
```

- [ ] **Step 3: Write minimal styling and verify manually**

`search/static/styles.css`:
```css
:root {
  --ink: #171413;
  --ink-70: rgba(23, 20, 19, 0.7);
  --ink-50: rgba(23, 20, 19, 0.5);
  --ink-20: rgba(23, 20, 19, 0.16);
  --wine: #620d3c;
}
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Inter", sans-serif; color: var(--ink); background: #fff; }
.page { max-width: 760px; margin: 0 auto; padding: 3rem 1.5rem; }
h1 { font-weight: 400; margin-bottom: 0.25rem; }
.subtitle { color: var(--ink-70); margin-bottom: 2rem; }
.search-row { display: flex; gap: 0.5rem; margin-bottom: 2rem; }
#search-input { flex: 1; padding: 0.75rem 1rem; font-size: 1rem; border: 1px solid var(--ink-20); border-radius: 4px; }
#search-btn { padding: 0.75rem 1.5rem; background: var(--wine); color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 1rem; }
.group { margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--ink-20); }
.group-label { font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase; color: var(--wine); margin-bottom: 0.5rem; }
.connector { margin-bottom: 1rem; }
.excerpt { background: #f7f5f2; border: 1px solid var(--ink-20); border-radius: 6px; padding: 1rem; margin-bottom: 0.75rem; }
.excerpt-meta { font-size: 0.8rem; color: var(--ink-50); margin-top: 0.5rem; font-family: ui-monospace, monospace; }
.loading, .empty, .error { color: var(--ink-50); }
```

Manual verification checklist (run once `search/app.py` is wired up in Task 13):
1. Start the server, open `http://127.0.0.1:8000`
2. Confirm the page loads with the search box visible
3. Type a question that matches seeded test data, click Search — confirm results render grouped by source label
4. Confirm a `sent_mail` group (if present in results) shows excerpts but no connector summary line
5. Confirm an empty/no-match query shows the "Nothing found" message rather than an error

- [ ] **Step 4: Commit**

```bash
git add search/static/
git commit -m "feat: add static single-page search UI"
```

---

## Task 13: Orchestration script and config

**Files:**
- Create: `config.yaml`
- Create: `build_all.py`
- Test: `tests/test_build_all.py`

**Interfaces:**
- Consumes: `ingest_blog` (Task 4), `ingest_obsidian` (Task 5), `ingest_twitter` (Task 6), `ingest_gmail` (Task 7), `build_and_save`, `SOURCE_FILES` (Task 8)
- Produces: `load_config(path: str = "config.yaml") -> dict`, `main() -> None`

- [ ] **Step 1: Write the config template**

`config.yaml`:
```yaml
blog:
  sitemap_url: "https://nitinpai.in/sitemap.xml"

obsidian:
  vault_path: "/path/to/obsidian/vault"

twitter:
  tweets_js_path: "data/raw/tweets.js"
  handle: "your_twitter_handle"
  include_replies: false

gmail:
  enabled: true
  client_secret_path: "client_secret.json"
  token_path: "data/gmail_token.json"
```

- [ ] **Step 2: Write the failing test**

`tests/test_build_all.py`:
```python
from unittest.mock import patch

import yaml

from build_all import load_config, main


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({"blog": {"sitemap_url": "https://x.com/sitemap.xml"}}))

    config = load_config(str(config_path))

    assert config["blog"]["sitemap_url"] == "https://x.com/sitemap.xml"


def test_main_calls_only_configured_sources(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({
        "blog": {"sitemap_url": "https://x.com/sitemap.xml"},
        "obsidian": {},
        "twitter": {},
        "gmail": {"enabled": False},
    }))
    monkeypatch.chdir(tmp_path)

    with patch("build_all.ingest_blog", return_value=3) as mock_blog, \
         patch("build_all.ingest_obsidian") as mock_obsidian, \
         patch("build_all.ingest_twitter") as mock_twitter, \
         patch("build_all.ingest_gmail") as mock_gmail, \
         patch("build_all.build_and_save", return_value=3) as mock_build:
        main()

    mock_blog.assert_called_once()
    mock_obsidian.assert_not_called()
    mock_twitter.assert_not_called()
    mock_gmail.assert_not_called()
    mock_build.assert_called_once()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_build_all.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_all'`

- [ ] **Step 4: Write minimal implementation**

`build_all.py`:
```python
from __future__ import annotations

import yaml

from ingest.blog import ingest_blog
from ingest.gmail import ingest_gmail
from ingest.obsidian import ingest_obsidian
from ingest.twitter import ingest_twitter
from indexing.build_index import SOURCE_FILES, build_and_save


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()

    blog_cfg = config.get("blog") or {}
    if blog_cfg.get("sitemap_url"):
        n = ingest_blog(blog_cfg["sitemap_url"], SOURCE_FILES["blog"])
        print(f"Blog: {n} posts")

    obsidian_cfg = config.get("obsidian") or {}
    if obsidian_cfg.get("vault_path"):
        n = ingest_obsidian(obsidian_cfg["vault_path"], SOURCE_FILES["notes"])
        print(f"Notes: {n} notes")

    twitter_cfg = config.get("twitter") or {}
    if twitter_cfg.get("tweets_js_path"):
        n = ingest_twitter(
            twitter_cfg["tweets_js_path"],
            twitter_cfg["handle"],
            SOURCE_FILES["tweets"],
            include_replies=twitter_cfg.get("include_replies", False),
        )
        print(f"Tweets: {n} tweets")

    gmail_cfg = config.get("gmail") or {}
    if gmail_cfg.get("enabled"):
        n = ingest_gmail(
            gmail_cfg.get("token_path", "data/gmail_token.json"),
            gmail_cfg.get("client_secret_path", "client_secret.json"),
            SOURCE_FILES["sent_mail"],
        )
        print(f"Sent mail: {n} emails")

    total = build_and_save(SOURCE_FILES, "data/chunks.json", "data/vectors.npy")
    print(f"Indexed {total} chunks total")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_build_all.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add config.yaml build_all.py tests/test_build_all.py
git commit -m "feat: add config-driven ingestion orchestration"
```

---

## Task 14: Local server entry point and README

**Files:**
- Create: `serve.py`
- Create: `README.md`

**Interfaces:**
- Consumes: `search.app:app` (Task 11)
- Produces: a runnable entry point (`python serve.py`) — no automated test; verified manually per Step 3

- [ ] **Step 1: Write the server entry point**

`serve.py`:
```python
from __future__ import annotations

import threading
import webbrowser

import uvicorn


def open_browser() -> None:
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    uvicorn.run("search.app:app", host="127.0.0.1", port=8000)
```

- [ ] **Step 2: Write the README**

`README.md`:
```markdown
# Nitin's Archive Search

A local search tool over your own writing: blog posts (nitinpai.in), Obsidian
notes, tweets, and substantive sent emails. Ask a question, get back what
you've actually written about it. Runs entirely on your machine — nothing is
hosted publicly, and email content is never sent to any external API.

## One-time setup

1. Install Python 3.11+ and clone this repo.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your own Groq API key (used only
   for the one-line "what you've said about this" summaries — get a free
   key at console.groq.com). Leave it blank to skip summaries entirely and
   see raw search results only.
4. Edit `config.yaml`:
   - `blog.sitemap_url`: your blog's sitemap URL (usually
     `https://yoursite.com/sitemap.xml`)
   - `obsidian.vault_path`: full path to your Obsidian vault folder
   - `twitter.handle`: your X/Twitter handle
5. **Twitter**: go to X Settings → Your Account → Download an archive.
   Once it arrives (can take a day), unzip it and copy `data/tweets.js`
   from inside the archive to `data/raw/tweets.js` in this repo.
6. **Gmail**: go to console.cloud.google.com, create a project, enable the
   Gmail API, create OAuth credentials (Desktop app type), download the
   JSON file and save it as `client_secret.json` in this repo's root. The
   first time you run ingestion, a browser window will open asking you to
   authorize access to your own Gmail — this is a one-time consent.
7. Run `python build_all.py` — this ingests all four sources and builds the
   search index. Takes a few minutes depending on how much content you have.
8. Run `python serve.py` — opens `http://127.0.0.1:8000` in your browser.
   Ask it a question.

## Updating the index

Whenever you publish something new, run `python build_all.py` again — it
re-fetches everything and rebuilds the index. There's no automatic sync.

## Privacy notes

- Blog, Notes, and Tweet excerpts get a short AI-generated summary (via your
  Groq key) when you search.
- Sent Mail excerpts never do — they're shown as plain search results only,
  so email content never leaves your machine via that path. It's still used
  locally to build the search index (embedded with a model that runs
  entirely on your machine, no API call).
- Only your **sent** mail is indexed, and only messages over ~150 words
  after stripping quoted reply chains — short replies and routine
  scheduling emails are excluded automatically.

## Development

Run tests: `PYTHONPATH=. pytest tests/ -v`
```

- [ ] **Step 3: Manual end-to-end verification**

1. From a clean checkout, follow the README's setup steps 1–7 with real
   (or minimal test) data for at least the blog source
2. Confirm `data/chunks.json` and `data/vectors.npy` exist after
   `python build_all.py`
3. Run `python serve.py`, confirm the browser opens automatically to
   `http://127.0.0.1:8000`
4. Ask a question you know the indexed content covers, confirm relevant
   results appear grouped by source
5. If a Groq key is set in `.env`, confirm non-`sent_mail` groups show a
   connector summary line; confirm `sent_mail` groups (if any matched)
   never do

- [ ] **Step 4: Commit**

```bash
git add serve.py README.md
git commit -m "docs: add local server entry point and setup README"
```

---

## Self-Review Notes

- **Spec coverage:** all four ingestion sources (Task 4–7), indexing
  (Task 8), source-diverse retrieval (Task 9), Gmail-privacy connector rule
  (Task 10–11), local search UI (Task 12), config-driven orchestration
  (Task 13), and packaging/handoff docs (Task 14) are each covered by a task.
- **Placeholder scan:** no TBD/TODO markers; every step has runnable code
  or a concrete manual-verification checklist.
- **Type consistency:** `Record` fields (`source, title, reference, date,
  text`) are used identically from Task 1 through Task 13. `search_diverse`'s
  signature in Task 9 matches its usage in Task 11's `app.py`.
