from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

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

    enabled_sources: set[str] = set()

    blog_cfg = config.get("blog") or {}
    if blog_cfg.get("sitemap_url"):
        enabled_sources.add("blog")
        try:
            n = ingest_blog(blog_cfg["sitemap_url"], SOURCE_FILES["blog"])
            print(f"Blog: {n} posts")
        except Exception as e:
            print(f"Warning: blog ingestion failed: {e}")

    obsidian_cfg = config.get("obsidian") or {}
    if obsidian_cfg.get("vault_path"):
        enabled_sources.add("notes")
        try:
            n = ingest_obsidian(obsidian_cfg["vault_path"], SOURCE_FILES["notes"])
            print(f"Notes: {n} notes")
        except Exception as e:
            print(f"Warning: obsidian ingestion failed: {e}")

    twitter_cfg = config.get("twitter") or {}
    if twitter_cfg.get("tweets_js_path"):
        enabled_sources.add("tweets")
        try:
            n = ingest_twitter(
                twitter_cfg["tweets_js_path"],
                twitter_cfg["handle"],
                SOURCE_FILES["tweets"],
                include_replies=twitter_cfg.get("include_replies", False),
            )
            print(f"Tweets: {n} tweets")
        except Exception as e:
            print(f"Warning: twitter ingestion failed: {e}")

    gmail_cfg = config.get("gmail") or {}
    if gmail_cfg.get("enabled"):
        enabled_sources.add("sent_mail")
        try:
            n = ingest_gmail(
                gmail_cfg.get("token_path", "data/gmail_token.json"),
                gmail_cfg.get("client_secret_path", "client_secret.json"),
                SOURCE_FILES["sent_mail"],
            )
            print(f"Sent mail: {n} emails")
        except Exception as e:
            print(f"Warning: gmail ingestion failed: {e}")

    # Only feed currently-configured/enabled sources into the index, so a
    # source that was disabled after a previous successful ingestion doesn't
    # keep contributing stale data to every rebuild forever.
    source_files = {k: v for k, v in SOURCE_FILES.items() if k in enabled_sources}

    total = build_and_save(source_files, "data/chunks.json", "data/vectors.npy")
    print(f"Indexed {total} chunks total")


if __name__ == "__main__":
    main()
