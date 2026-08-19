import json

import numpy as np

from search.retrieval import group_by_source, load_index, search_diverse


def _unit(vec):
    arr = np.array(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_load_index_reads_chunks_and_vectors(tmp_path):
    chunks = [{"source": "blog", "title": "A", "text": "x"}]
    chunks_path = tmp_path / "chunks.json"
    vectors_path = tmp_path / "vectors.npy"
    chunks_path.write_text(json.dumps(chunks))
    np.save(vectors_path, np.ones((1, 4), dtype=np.float32))

    loaded_chunks, vectors = load_index(str(chunks_path), str(vectors_path))

    assert loaded_chunks == chunks
    assert vectors.shape == (1, 4)


def test_search_diverse_ranks_by_similarity():
    chunks = [
        {"source": "blog", "text": "a"},
        {"source": "blog", "text": "b"},
        {"source": "notes", "text": "c"},
    ]
    vectors = np.array([_unit([1, 0]), _unit([0.9, 0.1]), _unit([0, 1])])
    query = _unit([1, 0])

    matches = search_diverse(query, vectors, chunks, top_k=3, min_similarity=0.0)

    assert matches[0][0] == 0
    assert matches[0][1] >= matches[1][1]


def test_search_diverse_caps_dominant_source():
    chunks = [{"source": "blog", "text": str(i)} for i in range(10)]
    vectors = np.tile(_unit([1, 0]), (10, 1))
    query = _unit([1, 0])

    matches = search_diverse(query, vectors, chunks, top_k=10, max_per_source=6, min_similarity=0.0)

    assert len(matches) == 6


def test_search_diverse_backfills_floor_for_minority_source():
    chunks = (
        [{"source": "blog", "text": str(i)} for i in range(9)]
        + [{"source": "notes", "text": "n"}]
    )
    vectors = np.vstack([
        np.tile(_unit([1, 0]), (9, 1)),
        _unit([0.5, 0.5]).reshape(1, 2),
    ])
    query = _unit([1, 0])

    matches = search_diverse(query, vectors, chunks, top_k=9, min_floor=1, min_similarity=0.0)

    matched_sources = {chunks[idx]["source"] for idx, _ in matches}
    assert "notes" in matched_sources


def test_group_by_source_groups_and_attaches_score():
    chunks = [{"source": "blog", "text": "a"}, {"source": "notes", "text": "b"}]
    grouped = group_by_source(chunks, [(0, 0.9), (1, 0.7)])

    assert set(grouped.keys()) == {"blog", "notes"}
    assert grouped["blog"][0]["score"] == 0.9
