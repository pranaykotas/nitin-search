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
