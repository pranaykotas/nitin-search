from unittest.mock import patch

import yaml

from build_all import load_config, main
from indexing.build_index import SOURCE_FILES


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

    with patch("build_all.ingest_blog", return_value=3, autospec=True) as mock_blog, \
         patch("build_all.ingest_obsidian", autospec=True) as mock_obsidian, \
         patch("build_all.ingest_twitter", autospec=True) as mock_twitter, \
         patch("build_all.ingest_gmail", autospec=True) as mock_gmail, \
         patch("build_all.build_and_save", return_value=3, autospec=True) as mock_build:
        main()

    mock_blog.assert_called_once()
    mock_obsidian.assert_not_called()
    mock_twitter.assert_not_called()
    mock_gmail.assert_not_called()
    mock_build.assert_called_once()


def test_main_continues_when_one_source_ingestion_raises(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({
        "blog": {"sitemap_url": "https://x.com/sitemap.xml"},
        "obsidian": {"vault_path": "/some/vault"},
        "twitter": {},
        "gmail": {"enabled": False},
    }))
    monkeypatch.chdir(tmp_path)

    with patch("build_all.ingest_blog", side_effect=RuntimeError("boom"), autospec=True) as mock_blog, \
         patch("build_all.ingest_obsidian", return_value=2, autospec=True) as mock_obsidian, \
         patch("build_all.ingest_twitter", autospec=True) as mock_twitter, \
         patch("build_all.ingest_gmail", autospec=True) as mock_gmail, \
         patch("build_all.build_and_save", return_value=2, autospec=True) as mock_build:
        main()

    mock_blog.assert_called_once()
    mock_obsidian.assert_called_once()
    mock_build.assert_called_once()
    assert "Warning" in capsys.readouterr().out


def test_main_excludes_disabled_sources_from_index(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({
        "blog": {"sitemap_url": "https://x.com/sitemap.xml"},
        "obsidian": {},
        "twitter": {},
        "gmail": {"enabled": False},
    }))
    monkeypatch.chdir(tmp_path)

    with patch("build_all.ingest_blog", return_value=3, autospec=True), \
         patch("build_all.ingest_obsidian", autospec=True), \
         patch("build_all.ingest_twitter", autospec=True), \
         patch("build_all.ingest_gmail", autospec=True), \
         patch("build_all.build_and_save", return_value=3, autospec=True) as mock_build:
        main()

    source_files_arg = mock_build.call_args.args[0]
    assert source_files_arg == {"blog": SOURCE_FILES["blog"]}
