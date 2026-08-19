from unittest.mock import Mock, patch

from search import llm


def test_generate_connector_returns_none_without_api_key(monkeypatch):
    monkeypatch.setattr(llm, "API_KEY", "")
    result = llm.generate_connector("blog", "a question", ["an excerpt"])
    assert result is None


def test_generate_connector_returns_model_text_on_success(monkeypatch):
    monkeypatch.setattr(llm, "API_KEY", "fake-key")
    fake_response = Mock()
    fake_response.raise_for_status = Mock()
    fake_response.json.return_value = {"choices": [{"message": {"content": "  A summary line.  "}}]}

    with patch("search.llm.requests.post", return_value=fake_response) as mock_post:
        result = llm.generate_connector("blog", "a question", ["an excerpt"])

    assert result == "A summary line."
    assert mock_post.called


def test_generate_connector_returns_none_on_request_failure(monkeypatch):
    monkeypatch.setattr(llm, "API_KEY", "fake-key")
    import requests as requests_module

    with patch("search.llm.requests.post", side_effect=requests_module.RequestException("boom")):
        result = llm.generate_connector("blog", "a question", ["an excerpt"])

    assert result is None
