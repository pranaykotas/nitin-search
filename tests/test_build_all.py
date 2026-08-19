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
