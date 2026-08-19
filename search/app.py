from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from indexing.embed import embed_texts
from search.llm import generate_connector
from search.retrieval import group_by_source, load_index, search_diverse

NO_CONNECTOR_SOURCES = {"sent_mail"}

app = FastAPI()

_index_cache: dict | None = None


def get_index() -> dict:
    global _index_cache
    if _index_cache is None:
        chunks_path = os.environ.get("CHUNKS_PATH", "data/chunks.json")
        vectors_path = os.environ.get("VECTORS_PATH", "data/vectors.npy")
        chunks, vectors = load_index(chunks_path, vectors_path)
        _index_cache = {"chunks": chunks, "vectors": vectors}
    return _index_cache


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        return {"error": "empty_question", "message": "Ask a question first."}

    top_k = int(os.environ.get("TOP_K", "10"))
    min_similarity = float(os.environ.get("MIN_SIMILARITY", "0.20"))

    index = get_index()
    query_vector = embed_texts([question])[0]
    matches = search_diverse(query_vector, index["vectors"], index["chunks"], top_k=top_k, min_similarity=min_similarity)

    if not matches:
        return {"groups": [], "message": "Nothing in the archive matches this yet."}

    grouped = group_by_source(index["chunks"], matches)
    groups = []
    for source, items in grouped.items():
        excerpts = [item["text"] for item in items]
        connector = None
        if source not in NO_CONNECTOR_SOURCES:
            connector = generate_connector(source, question, excerpts)
        groups.append({
            "source": source,
            "connector_text": connector,
            "excerpts": [
                {"text": item["text"], "title": item["title"], "reference": item["reference"], "date": item["date"]}
                for item in items
            ],
        })

    return {"groups": groups, "message": None}


STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index_page():
        return FileResponse(str(STATIC_DIR / "index.html"))
