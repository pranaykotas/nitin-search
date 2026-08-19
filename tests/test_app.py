import json
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    chunks = [
        {"source": "blog", "title": "A Post", "reference": "https://x.com/a", "date": "2024-01-01", "text": "About fiscal federalism."},
        {"source": "sent_mail", "title": "An Email", "reference": "gmail:1", "date": "2024-01-02", "text": "Also about fiscal federalism."},
    ]
    chunks_path = tmp_path / "chunks.json"
    vectors_path = tmp_path / "vectors.npy"
    chunks_path.write_text(json.dumps(chunks))
    np.save(vectors_path, np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32))

    monkeypatch.setenv("CHUNKS_PATH", str(chunks_path))
    monkeypatch.setenv("VECTORS_PATH", str(vectors_path))
    monkeypatch.setenv("MIN_SIMILARITY", "0.0")

    from search import app as app_module
    app_module._index_cache = None

    with patch("search.app.embed_texts", return_value=np.array([[1.0, 0.0]], dtype=np.float32)), \
         patch("search.app.generate_connector", return_value="A generated summary.") as mock_connector:
        yield TestClient(app_module.app), mock_connector


def test_ask_returns_groups_for_matching_question(client):
    test_client, _ = client
    response = test_client.post("/ask", json={"question": "fiscal federalism"})
    assert response.status_code == 200
    body = response.json()
    sources = {g["source"] for g in body["groups"]}
    assert sources == {"blog", "sent_mail"}


def test_ask_skips_connector_for_sent_mail_source(client):
    test_client, mock_connector = client
    test_client.post("/ask", json={"question": "fiscal federalism"})
    called_sources = {call.args[0] for call in mock_connector.call_args_list}
    assert "sent_mail" not in called_sources
    assert "blog" in called_sources


def test_ask_sent_mail_group_has_no_connector_text(client):
    test_client, _ = client
    response = test_client.post("/ask", json={"question": "fiscal federalism"})
    body = response.json()
    sent_mail_group = next(g for g in body["groups"] if g["source"] == "sent_mail")
    assert sent_mail_group["connector_text"] is None


def test_ask_empty_question_returns_error(client):
    test_client, _ = client
    response = test_client.post("/ask", json={"question": "   "})
    assert response.json()["error"] == "empty_question"


def test_ask_missing_index_returns_friendly_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNKS_PATH", str(tmp_path / "does-not-exist-chunks.json"))
    monkeypatch.setenv("VECTORS_PATH", str(tmp_path / "does-not-exist-vectors.npy"))

    from search import app as app_module
    app_module._index_cache = None

    with patch("search.app.embed_texts", return_value=np.array([[1.0, 0.0]], dtype=np.float32)):
        test_client = TestClient(app_module.app)
        response = test_client.post("/ask", json={"question": "fiscal federalism"})

    assert response.status_code == 200
    body = response.json()
    assert body["error"] == "no_index"
    assert "build_all.py" in body["message"]
