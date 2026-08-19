from __future__ import annotations

import json

import numpy as np


def load_index(chunks_path: str, vectors_path: str) -> tuple[list[dict], np.ndarray]:
    with open(chunks_path) as f:
        chunks = json.load(f)
    vectors = np.load(vectors_path)
    return chunks, vectors


def search_diverse(
    query_vector: np.ndarray,
    vectors: np.ndarray,
    chunks: list[dict],
    top_k: int = 10,
    min_floor: int = 1,
    max_per_source: int = 6,
    min_similarity: float = 0.25,
) -> list[tuple[int, float]]:
    """Return the top_k matches by relevance, capping any single source at
    max_per_source and backfilling a floor of min_floor for sources that
    would otherwise be shut out entirely."""
    if vectors.shape[0] == 0:
        return []
    sims = vectors @ query_vector
    order = [i for i in np.argsort(-sims) if sims[i] >= min_similarity]

    source_counts: dict[str, int] = {}
    results: list[tuple[int, float]] = []
    for i in order:
        if len(results) >= top_k:
            break
        source = chunks[int(i)].get("source", "?")
        if source_counts.get(source, 0) >= max_per_source:
            continue
        source_counts[source] = source_counts.get(source, 0) + 1
        results.append((int(i), float(sims[i])))

    for i in order:
        source = chunks[int(i)].get("source", "?")
        if source_counts.get(source, 0) >= min_floor:
            continue
        if any(idx == int(i) for idx, _ in results):
            continue
        source_counts[source] = source_counts.get(source, 0) + 1
        results.append((int(i), float(sims[i])))

    results.sort(key=lambda x: -x[1])
    return results


def group_by_source(chunks: list[dict], matches: list[tuple[int, float]]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for idx, score in matches:
        chunk = {**chunks[idx], "score": score}
        groups.setdefault(chunk["source"], []).append(chunk)
    return groups
