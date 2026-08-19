from __future__ import annotations

import os

import requests

API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = (
    "You summarise what Nitin has written, using ONLY the excerpts given "
    "below. Write one or two sentences. Do not add claims, examples, or "
    "reasoning that is not directly present in the excerpts."
)


def generate_connector(source: str, question: str, excerpts: list[str]) -> str | None:
    """Write a short connector line grounded in the given excerpts.

    Returns None if no API key is configured or the request fails for any
    reason — callers must treat None as "show excerpts without a connector
    line," never as an error to surface to the user.
    """
    if not API_KEY:
        return None

    excerpt_block = "\n\n".join(f"- {e}" for e in excerpts)
    user_prompt = (
        f"Question: {question}\n\n"
        f"Excerpts from {source}:\n{excerpt_block}\n\n"
        f"Write the one-to-two sentence connector line now."
    )

    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 120,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError):
        return None
