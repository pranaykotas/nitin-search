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
   Once it arrives (can take a day), unzip it. Create the directory `mkdir -p data/raw`,
   then copy `data/tweets.js` from inside the archive to `data/raw/tweets.js` in this repo.
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
