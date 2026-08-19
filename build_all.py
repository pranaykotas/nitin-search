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
