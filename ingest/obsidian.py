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
