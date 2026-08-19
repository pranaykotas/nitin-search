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
