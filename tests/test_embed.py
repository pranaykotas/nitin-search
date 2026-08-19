import numpy as np

from indexing.embed import EMBED_DIM, embed_texts


def test_empty_input_returns_empty_array():
    result = embed_texts([])
    assert result.shape == (0, EMBED_DIM)


def test_embeds_texts_to_expected_shape():
    result = embed_texts(["hello world", "a second sentence"])
    assert result.shape == (2, EMBED_DIM)
    assert result.dtype == np.float32


def test_similar_texts_score_higher_than_unrelated_ones():
    vectors = embed_texts([
        "The state budget deficit widened this year",
        "Government fiscal spending increased sharply",
        "My cat sleeps most of the afternoon",
    ])
    sim_related = float(vectors[0] @ vectors[1])
    sim_unrelated = float(vectors[0] @ vectors[2])
    assert sim_related > sim_unrelated
