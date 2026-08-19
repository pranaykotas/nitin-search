import json

import numpy as np

from ingest.common import Record, write_records
from indexing.build_index import build_and_save, chunks_from_records


def test_chunks_from_records_produces_one_chunk_for_short_text():
    records = [Record(source="blog", title="A Post", reference="https://x.com/a", date="2024-01-01", text="Short body.")]
    chunks = chunks_from_records(records)
    assert len(chunks) == 1
    assert chunks[0]["source"] == "blog"
    assert chunks[0]["text"] == "Short body."
    assert "A Post" in chunks[0]["embed_text"]


def test_chunks_from_records_splits_long_text():
    long_text = "\n\n".join([" ".join(["word"] * 300)] * 3)
    records = [Record(source="notes", title="Big Note", reference="n.md", date="", text=long_text)]
    chunks = chunks_from_records(records)
    assert len(chunks) >= 2


def test_build_and_save_combines_multiple_sources(tmp_path, monkeypatch):
    def fake_embed_texts(texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr("indexing.build_index.embed_texts", fake_embed_texts)

    blog_path = str(tmp_path / "blog.json")
    notes_path = str(tmp_path / "notes.json")
    write_records([Record(source="blog", title="P", reference="u", date="", text="Blog text.")], blog_path)
    write_records([Record(source="notes", title="N", reference="n.md", date="", text="Note text.")], notes_path)

    chunks_out = str(tmp_path / "chunks.json")
    vectors_out = str(tmp_path / "vectors.npy")
    source_files = {"blog": blog_path, "notes": notes_path, "tweets": str(tmp_path / "missing.json")}

    count = build_and_save(source_files, chunks_out, vectors_out)

    assert count == 2
    with open(chunks_out) as f:
        chunks = json.load(f)
    assert {c["source"] for c in chunks} == {"blog", "notes"}
    vectors = np.load(vectors_out)
    assert vectors.shape == (2, 4)
