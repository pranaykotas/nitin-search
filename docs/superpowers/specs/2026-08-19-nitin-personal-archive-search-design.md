# Nitin Pai Personal Archive Search — Design

Status: draft, pending Nitin's review
Author: Pranay Kotasthane (with Claude)
Date: 2026-08-19

## Problem

Nitin Pai wants a tool, for his own use, that works the way "Ask Our Views"
works on frameworks.pranaykotas.com: when an idea occurs to him, he can ask
it a question and get back what he's actually already written or said about
it — pulled from his blog (nitinpai.in), his Obsidian notes, his tweets, and
his own sent emails that contain substantive thinking.

Unlike the ATU tool, this is single-author, single-user, and privacy-sensitive
(sent email is personal correspondence). It should run entirely on his own
machine, with no public hosting and no ongoing infrastructure to maintain.

## Non-goals

- Not a multi-tenant SaaS product. Built for one person. If others want it
  later, generalize then.
- Not a public website. No Render/hosting, no domain, no auth system.
- Not a live-syncing service. Nitin re-runs ingestion manually when he wants
  fresh content indexed — no scheduled jobs, no daemons.
- Not a framework-recommendation tool (unlike frameworks.pranaykotas.com's
  "Find a framework" half). This is archive search only.

## Architecture

Three independent, sequential stages, mirroring the proven pattern in
`~/Projects/frameworks/ask/`:

```
[4 ingestion scripts] → [chunks.json + vectors.npy] → [local search app]
```

### 1. Ingestion

One script per source. Each produces a JSON list of records in a common
shape:

```json
{
  "source": "blog" | "notes" | "tweets" | "sent_mail",
  "title": "string",
  "reference": "URL or file path or tweet permalink",
  "date": "ISO 8601",
  "text": "plain text content"
}
```

| Source | Script | Method | Who runs it |
|---|---|---|---|
| Blog (nitinpai.in) | `ingest_blog.py` | Crawl sitemap.xml (or paginated index if none), fetch each post, strip HTML to text | Us — public data, no credentials |
| Obsidian vault | `ingest_obsidian.py` | Walk vault directory for `.md` files, strip frontmatter/wikilinks, keep title/path/mtime | Nitin points us at the vault path; script runs locally |
| Twitter | `ingest_twitter.py` | Parse `tweets.js` from a manually downloaded X archive; own original tweets only (retweets/replies-to-others excluded by default) | **Nitin** — only he can request his own X archive |
| Gmail (sent) | `ingest_gmail.py` | Gmail API, `in:sent`, strip quoted chains/signatures, keep messages over ~150 words | **Nitin** — one-time OAuth consent under his own Google account |

Ingestion is idempotent and re-runnable: each script overwrites its own
output file (`data/blog.json`, `data/notes.json`, `data/tweets.json`,
`data/sent_mail.json`). No incremental-sync complexity needed at this scale.

### 2. Indexing

`build_index.py`, adapted directly from
`~/Projects/frameworks/ask/build_index.py`:

- Reads all four `data/*.json` files
- Chunks any text over ~400 words, with paragraph-aware splitting and overlap
  (same logic as ATU's `_sub_chunk`)
- Embeds every chunk locally via `fastembed` (`all-MiniLM-L6-v2`, ONNX
  runtime, no torch, no external API call, no cost)
- Writes `data/chunks.json` + `data/vectors.npy`

No author-attribution step is needed here (unlike ATU's `attribution.py`)
since there's only one author. `source` takes the place `author` played in
the ATU grouping logic.

### 3. Local search app

- `serve.py`: a small FastAPI app (adapted from `ask/app.py` /
  `ask/retrieval.py`) that loads the index into memory and exposes a local
  `/ask` endpoint
- A single static HTML page (adapted from `finder.js` /
  `index.qmd`'s unified search box), served at `localhost:8000`
- One command starts it: `python serve.py` — opens the browser automatically
- Results grouped by **source** (Blog / Notes / Tweets / Sent Mail) instead
  of by author, using the same `search_diverse()` relevance-with-floor logic
  from the ATU build so no one source can crowd out the others entirely

**Privacy rule for the LLM connector-summary step:** Blog, Notes, and Tweet
excerpts get the one-line AI-generated summary per source (same
`generate_connector()` call as ATU, via Nitin's own Groq API key). **Sent
Mail excerpts skip this step entirely** — they're shown as raw search
results only, so email content never gets sent to a third-party API. This
was an explicit decision: everything else about the app is local-only, and
email is the one source where an external call felt like it broke that
promise even though blog/notes/tweets already involve one.

### Packaging for handoff

- `README.md`: numbered one-time setup steps — clone repo, `pip install -r
  requirements.txt`, download X archive and drop the zip in `data/raw/`, run
  Gmail OAuth script once (opens a consent screen in his browser), point
  `config.yaml` at his blog URL and Obsidian vault path, run
  `python build_all.py` to ingest + index everything, then `python serve.py`
  to search.
- `config.yaml`: blog root URL, Obsidian vault path, any source-specific
  toggles (e.g., include retweets: false).
- `.env.example`: `GROQ_API_KEY=` — his own key, so usage is billed to him,
  not to this project's maintainer.
- No deployment step. No hosting account needed. Re-running ingestion +
  `build_all.py` is how he refreshes the index when he's written something
  new.

## Data flow diagram

```
nitinpai.in ──┐
Obsidian vault ┤
X archive.zip ─┼──► ingest_*.py ──► data/*.json ──► build_index.py ──► chunks.json + vectors.npy
Gmail (sent) ──┘                                                              │
                                                                               ▼
                                                                    serve.py (FastAPI, local)
                                                                               │
                                                                               ▼
                                                              localhost:8000 (search UI, browser)
```

## Open questions for Nitin (before implementation starts)

1. Does nitinpai.in have a sitemap.xml, or do we need to crawl via an index
   page? (affects `ingest_blog.py` complexity)
2. Rough size of his Obsidian vault and sent-mail history — matters for
   first-run indexing time, though at ATU's scale (3,700 chunks) this took
   well under a minute locally.
3. Does he already have a Groq (or other LLM) API key, or does that need
   setting up as part of onboarding?
4. Any Gmail labels/folders he'd want excluded even within Sent (e.g. an
   "admin" or "scheduling" label) beyond the length filter?

## Testing approach

Mirrors ATU's `ask/tests/`: unit tests per ingestion parser (feed it sample
HTML/markdown/tweets.js/Gmail API response fixtures, assert correct
record shape), unit tests for chunking and `search_diverse()` (already
proven code, minimal changes expected), and a manual end-to-end pass once
Nitin's real data is available — same verification approach used to catch
the ATU headerless-early-editions bug.
