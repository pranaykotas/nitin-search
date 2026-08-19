from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ingest.common import Record, read_records
from indexing.chunking import sub_chunk
from indexing.embed import embed_texts

SOURCE_FILES = {
    "blog": "data/blog.json",
    "notes": "data/notes.json",
    "tweets": "data/tweets.json",
    "sent_mail": "data/sent_mail.json",
}


def chunks_from_records(records: list[Record]) -> list[dict]:
    chunks = []
    for record in records:
        for sub_text in sub_chunk(record.text):
            chunks.append({
                "source": record.source,
                "title": record.title,
                "reference": record.reference,
                "date": record.date,
                "text": sub_text,
                "embed_text": f"{record.title}\n\n{sub_text}",
            })
    return chunks


def build_and_save(source_files: dict[str, str], chunks_out_path: str, vectors_out_path: str) -> int:
    all_records: list[Record] = []
    for path in source_files.values():
        if Path(path).exists():
            all_records.extend(read_records(path))

    chunks = chunks_from_records(all_records)
    texts = [c["embed_text"] for c in chunks]
    vectors = embed_texts(texts)

    Path(chunks_out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_out_path, "w") as f:
        json.dump(chunks, f)
    np.save(vectors_out_path, vectors)

    return len(chunks)


if __name__ == "__main__":
    count = build_and_save(SOURCE_FILES, "data/chunks.json", "data/vectors.npy")
    print(f"Built {count} chunks")
