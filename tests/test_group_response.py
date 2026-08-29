"""Tests for the group recommendation response contract."""

import json

import pytest

from app.schemas.run import GroupRecommendationOut
from cinematch.recommendation import (
    AggregationStrategy,
    GroupRecommendationItem,
    GroupRecommendationResponse,
    MemberPredictedScore,
    MovieResponseMetadata,
    build_group_recommendation_response,
    group_response_to_backend_payload,
)

def test_group_response_contains_required_fairness_fields() -> None:
    """The core response must expose all fields required by Backend."""

    member_scores = (
        MemberPredictedScore(
            user_id=1,
            display_name="Member 1",
            predicted_score=4.8,
        ),
        MemberPredictedScore(
            user_id=2,
            display_name="Member 2",
            predicted_score=4.4,
        ),
    )

    recommendation = GroupRecommendationItem(
        movie_id=50,
        rank=1,
        group_score=4.6,
        minimum_score=4.4,
        disagreement=0.2,
        member_scores=member_scores,
        misery_warning=False,
        explanations=(
            "Aggregated using average",
            "No member is below the misery threshold",
        ),
    )

    response = GroupRecommendationResponse(
        room_id=1,
        strategy=AggregationStrategy.AVERAGE,
        recommendations=(recommendation,),
    )

    assert response.schema_version == "1.0"
    assert response.room_id == 1
    assert response.strategy is AggregationStrategy.AVERAGE
    assert len(response.recommendations) == 1

    first_item = response.recommendations[0]

    assert first_item.movie_id == 50
    assert first_item.rank == 1
    assert first_item.group_score == 4.6
    assert first_item.minimum_score == 4.4
    assert first_item.disagreement == 0.2
    assert first_item.member_scores == member_scores
    assert first_item.misery_warning is False
    assert len(first_item.explanations) == 2



def test_build_group_recommendation_response_ranks_movies() -> None:
    """Response must rank movies and align scores with members."""

    response = build_group_recommendation_response(
        room_id=7,
        member_user_ids=[101, 202],
        candidate_scores={
            50: [4.8, 4.4],
            1: [4.0, 4.0],
            100: [3.0, 3.0],
        },
        strategy="average",
        top_k=2,
        member_display_names={
            101: "Member 1",
            202: "Member 2",
        },
    )

    assert response.room_id == 7
    assert response.strategy is AggregationStrategy.AVERAGE
    assert len(response.recommendations) == 2
    assert [
        item.movie_id
        for item in response.recommendations
    ] == [50, 1]
    assert [
        item.rank
        for item in response.recommendations
    ] == [1, 2]

    first_item = response.recommendations[0]

    assert first_item.group_score == pytest.approx(4.6)
    assert first_item.minimum_score == pytest.approx(4.4)
    assert first_item.member_scores[0].user_id == 101
    assert first_item.member_scores[0].display_name == "Member 1"
    assert first_item.member_scores[0].predicted_score == 4.8
    assert first_item.member_scores[1].user_id == 202
    assert first_item.member_scores[1].predicted_score == 4.4
    assert first_item.misery_warning is False
    assert len(first_item.explanations) == 3


def test_build_group_recommendation_response_adds_warning() -> None:
    """Average keeps a low member score but emits a warning."""

    response = build_group_recommendation_response(
        room_id=1,
        member_user_ids=[1, 2],
        candidate_scores={
            50: [1.5, 5.0],
        },
        strategy="average",
        misery_threshold=2.0,
    )

    item = response.recommendations[0]

    assert item.group_score == pytest.approx(3.25)
    assert item.minimum_score == pytest.approx(1.5)
    assert item.misery_warning is True
    assert "below misery threshold" in item.explanations[2]


def test_build_group_recommendation_response_filters_misery() -> None:
    """Average Without Misery must remove low-scoring movies."""

    response = build_group_recommendation_response(
        room_id=1,
        member_user_ids=[1, 2],
        candidate_scores={
            50: [1.5, 5.0],
            100: [4.0, 4.5],
        },
        strategy="average_without_misery",
        misery_threshold=2.0,
    )

    assert len(response.recommendations) == 1
    assert response.recommendations[0].movie_id == 100
    assert response.recommendations[0].misery_warning is False


def test_build_group_recommendation_response_rejects_score_count() -> None:
    """Every movie must have one score for every member."""

    with pytest.raises(
        ValueError,
        match="must match the member count",
    ):
        build_group_recommendation_response(
            room_id=1,
            member_user_ids=[1, 2, 3],
            candidate_scores={
                50: [4.0, 5.0],
            },
            strategy="average",
        )


def test_build_group_recommendation_response_rejects_duplicate_members() -> None:
    """A group response must not contain the same member twice."""

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        build_group_recommendation_response(
            room_id=1,
            member_user_ids=[1, 1],
            candidate_scores={
                50: [4.0, 5.0],
            },
            strategy="average",
        )


@pytest.mark.parametrize("invalid_room_id", [0, -1, True])
def test_build_group_recommendation_response_rejects_room_id(
    invalid_room_id: object,
) -> None:
    """Backend room identifiers must be positive integers."""

    with pytest.raises(
        ValueError,
        match="Room ID must be a positive integer",
    ):
        build_group_recommendation_response(
            room_id=invalid_room_id,  # type: ignore[arg-type]
            member_user_ids=[1, 2],
            candidate_scores={
                50: [4.0, 5.0],
            },
            strategy="average",
        )



def test_backend_payload_matches_pydantic_contract() -> None:
    """Serialized group output must validate as Backend response."""

    response = build_group_recommendation_response(
        room_id=7,
        member_user_ids=[101, 202],
        candidate_scores={
            50: [4.8, 4.4],
        },
        strategy="average",
        member_display_names={
            101: "Member 1",
            202: "Member 2",
        },
    )

    payload = group_response_to_backend_payload(
        response=response,
        movie_metadata={
            50: MovieResponseMetadata(
                title="Star Wars (1977)",
                genres=("Action", "Adventure", "Sci-Fi"),
                poster_url=None,
                runtime_minutes=None,
            ),
        },
    )

    validated_response = GroupRecommendationOut.model_validate(
        payload
    )

    assert validated_response.schema_version == "1.0"
    assert validated_response.room_id == 7
    assert len(validated_response.recommendations) == 1

    first_item = validated_response.recommendations[0]

    assert first_item.movie_id == 50
    assert first_item.rank == 1
    assert first_item.title == "Star Wars (1977)"
    assert first_item.genres == [
        "Action",
        "Adventure",
        "Sci-Fi",
    ]
    assert first_item.group_score == pytest.approx(4.6)
    assert first_item.minimum_score == pytest.approx(4.4)
    assert len(first_item.member_scores) == 2
    assert first_item.member_scores[0].user_id == 101

    json.dumps(payload)


def test_backend_payload_rejects_missing_movie_metadata() -> None:
    """Every ranked movie must be enriched by Backend metadata."""

    response = build_group_recommendation_response(
        room_id=1,
        member_user_ids=[1, 2],
        candidate_scores={
            50: [4.0, 4.5],
        },
        strategy="average",
    )

    with pytest.raises(
        ValueError,
        match="Missing metadata for movie ID 50",
    ):
        group_response_to_backend_payload(
            response=response,
            movie_metadata={},
        )


def test_backend_payload_rejects_empty_movie_title() -> None:
    """Frontend must not receive an empty movie title."""

    response = build_group_recommendation_response(
        room_id=1,
        member_user_ids=[1, 2],
        candidate_scores={
            50: [4.0, 4.5],
        },
        strategy="average",
    )

    with pytest.raises(
        ValueError,
        match="Movie title must not be empty",
    ):
        group_response_to_backend_payload(
            response=response,
            movie_metadata={
                50: MovieResponseMetadata(title="   "),
            },
        )


def test_backend_payload_rejects_invalid_runtime() -> None:
    """Runtime must be positive when Backend provides it."""

    response = build_group_recommendation_response(
        room_id=1,
        member_user_ids=[1, 2],
        candidate_scores={
            50: [4.0, 4.5],
        },
        strategy="average",
    )

    with pytest.raises(
        ValueError,
        match="Runtime minutes",
    ):
        group_response_to_backend_payload(
            response=response,
            movie_metadata={
                50: MovieResponseMetadata(
                    title="Star Wars (1977)",
                    runtime_minutes=0,
                ),
            },
        )