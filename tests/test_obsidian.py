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
