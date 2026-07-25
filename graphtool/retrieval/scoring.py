import math
from collections.abc import Mapping, Sequence

from graphtool.retrieval.bm25 import BM25Document, BM25Index

SEMANTIC_SIMILARITY_FLOOR = 0.70
BM25_SATURATION = 6.0


def bm25_index(text_by_id: Mapping[str, str]) -> BM25Index:
    return BM25Index(
        [
            BM25Document(id=item_id, text=text)
            for item_id, text in text_by_id.items()
        ]
    )


def bm25_scores(query: str, index: BM25Index) -> dict[str, float]:
    return calibrate_bm25_scores(
        {document.id: score for document, score in index.rank(query)}
    )


def calibrate_bm25_score(score: float) -> float:
    """Map a raw BM25 score onto [0, 1) against a fixed reference point, so
    scores stay comparable across separate indexes and candidate sets."""
    return score / (score + BM25_SATURATION)


def calibrate_bm25_scores(scores: Mapping[str, float]) -> dict[str, float]:
    return {
        item_id: calibrate_bm25_score(score)
        for item_id, score in scores.items()
        if score > 0.0
    }


def combine_weighted_scores(
    contributions: Sequence[tuple[float, float]],
) -> float:
    """Combine independent calibrated signals into [0, 1). Averaging instead
    divides by every weight that could theoretically fire, which no real query
    does, so a chunk matching one field strongly would score near zero.

    A signal reaching 1.0 at the heaviest weight zeroes the product and pins
    the result at 1.0, hiding every other signal, so callers must keep
    saturating signals below the heaviest weight."""
    max_weight = max(weight for _, weight in contributions)
    remaining = 1.0
    for score, weight in contributions:
        remaining *= 1.0 - score * weight / max_weight
    return 1.0 - remaining


def calibrate_similarity(similarity: float) -> float:
    """Rescale a cosine similarity against a fixed floor. Unrelated text still
    scores well above zero with these embeddings, so the floor is what makes a
    calibrated score mean the same thing on every query."""
    return max(
        0.0,
        (similarity - SEMANTIC_SIMILARITY_FLOOR)
        / (1.0 - SEMANTIC_SIMILARITY_FLOOR),
    )


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(
        a * b for a, b in zip(left, right, strict=True)
    ) / (left_norm * right_norm)


def semantic_similarity_scores(
    query_vector: Sequence[float] | None,
    vectors: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    if query_vector is None:
        return {}
    return {
        item_id: score
        for item_id, vector in vectors.items()
        if (
            score := calibrate_similarity(
                cosine_similarity(query_vector, vector)
            )
        ) > 0.0
    }
