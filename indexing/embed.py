from __future__ import annotations

import numpy as np

EMBED_DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64

_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    model = _get_model()
    return np.array(list(model.embed(texts, batch_size=BATCH_SIZE)), dtype=np.float32)
