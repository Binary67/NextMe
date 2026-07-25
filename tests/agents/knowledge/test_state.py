import json

import pytest

from graphtool.agents.knowledge.state import (
    ExpandRecommendation,
    SearchRecommendation,
    SufficiencyDecision,
)


def test_sufficiency_decision_schema_uses_supported_union():
    schema = SufficiencyDecision.model_json_schema()

    assert "oneOf" not in json.dumps(schema)
    assert len(schema["properties"]["recommendation"]["anyOf"]) == 3


@pytest.mark.parametrize(
    ("recommendation", "expected_type"),
    [
        (
            {
                "action": "search",
                "reason": "More evidence is needed.",
                "search_focus": "latest result",
            },
            SearchRecommendation,
        ),
        (
            {
                "action": "expand",
                "reason": "Adjacent context is needed.",
                "source": "results.md",
                "chunk_id": "chunk-1",
            },
            ExpandRecommendation,
        ),
    ],
)
def test_sufficiency_decision_parses_recommendation_variants(
    recommendation,
    expected_type,
):
    decision = SufficiencyDecision.model_validate(
        {
            "verdict": "insufficient",
            "missing_information": ["The latest result is missing."],
            "recommendation": recommendation,
        }
    )

    assert isinstance(decision.recommendation, expected_type)
