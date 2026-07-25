import pytest

from graphtool.retrieval.scoring import (
    BM25_SATURATION,
    SEMANTIC_SIMILARITY_FLOOR,
    calibrate_bm25_score,
    calibrate_bm25_scores,
    calibrate_similarity,
    combine_weighted_scores,
    semantic_similarity_scores,
)


def test_calibrate_bm25_score_saturates_without_reaching_one():
    assert calibrate_bm25_score(0.0) == 0.0
    assert calibrate_bm25_score(BM25_SATURATION) == pytest.approx(0.5)
    assert calibrate_bm25_score(1000.0) < 1.0


def test_calibrate_bm25_score_preserves_relative_strength():
    strong = calibrate_bm25_score(8.4)
    weak = calibrate_bm25_score(2.1)

    assert strong > weak
    assert weak > 0.0


def test_calibrate_bm25_scores_drops_non_positive_and_ignores_candidate_set():
    alone = calibrate_bm25_scores({"weak": 2.1})
    crowded = calibrate_bm25_scores({"weak": 2.1, "strong": 8.4, "absent": 0.0})

    assert "absent" not in crowded
    assert alone["weak"] == pytest.approx(crowded["weak"])
    assert crowded["strong"] > crowded["weak"]


def test_combine_weighted_scores_lets_one_strong_signal_register():
    only_content = combine_weighted_scores(((0.6, 1.0), (0.0, 2.0)))

    assert only_content == pytest.approx(0.3)


def test_combine_weighted_scores_adds_confidence_for_agreeing_signals():
    single = combine_weighted_scores(((0.6, 1.0), (0.0, 1.0)))
    both = combine_weighted_scores(((0.6, 1.0), (0.6, 1.0)))

    assert both > single
    assert both < 1.0


def test_combine_weighted_scores_stays_below_one_for_unsaturated_signals():
    combined = combine_weighted_scores(((0.9, 2.0), (0.99, 1.0), (0.99, 0.5)))

    assert combined < 1.0


def test_combine_weighted_scores_saturates_when_a_signal_reaches_top_weight():
    """Pins the failure mode callers must avoid: a signal at 1.0 carrying the
    heaviest weight zeroes the product and hides every other signal."""
    saturated = combine_weighted_scores(((1.0, 2.0), (0.0, 2.0), (0.0, 1.0)))

    assert saturated == 1.0


def test_calibrate_similarity_uses_fixed_floor():
    assert calibrate_similarity(SEMANTIC_SIMILARITY_FLOOR) == 0.0
    assert calibrate_similarity(0.30) == 0.0
    assert calibrate_similarity(1.0) == pytest.approx(1.0)
    assert calibrate_similarity(0.85) == pytest.approx(0.5)


def test_semantic_similarity_scores_drop_matches_below_floor():
    scores = semantic_similarity_scores(
        [1.0, 0.0],
        {
            "aligned": [1.0, 0.0],
            "below_floor": [0.6, 0.8],
            "orthogonal": [0.0, 1.0],
        },
    )

    assert scores == {"aligned": pytest.approx(1.0)}


def test_semantic_similarity_scores_do_not_stretch_a_weak_best_match():
    scores = semantic_similarity_scores(
        [1.0, 0.0],
        {"best": [0.75, 0.6614], "next": [0.72, 0.6940]},
    )

    assert scores["best"] < 0.2
    assert scores["best"] > scores["next"]


def test_semantic_similarity_scores_are_independent_of_the_candidate_set():
    alone = semantic_similarity_scores([1.0, 0.0], {"target": [0.9, 0.4359]})
    crowded = semantic_similarity_scores(
        [1.0, 0.0],
        {"target": [0.9, 0.4359], "stronger": [1.0, 0.0]},
    )

    assert alone["target"] == pytest.approx(crowded["target"])


def test_semantic_similarity_scores_without_query_vector_return_empty():
    assert semantic_similarity_scores(None, {"target": [1.0, 0.0]}) == {}
