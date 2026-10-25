"""Turn a ready room into a ranked Top-K using the trained model.

Every member is scored by the AI library through the cold-start path,
the individual predictions are aggregated with the room's group strategy,
and the ranked result is stored so the voting screens and the final
result stay explainable without recomputing anything.
"""

from __future__ import annotations

import json

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.movie import Movie, Rating
from app.models.recommendation import RecommendationRun, RunItem
from app.models.room import Room, RoomMember
from app.services.predictor_service import (
    MemberProfileTooThin,
    RecommenderUnavailable,
    load_recommender,
)
from cinematch.recommendation.group import (
    MAX_GROUP_SIZE,
    MIN_GROUP_SIZE,
    AggregationStrategy,
)
from cinematch.recommendation.response import (
    build_group_recommendation_response,
)


def _display_name(email: str) -> str:
    """Use the local part of the email as a friendly member name."""
    return email.split("@", 1)[0] or email


def _resolve_room_settings(room: Room) -> tuple[str, int, float]:
    """Read strategy, Top-K and misery threshold for one room.

    The room stores constraints as a JSON string set by the host. Unknown
    or malformed values fall back to the configured defaults instead of
    failing the run.
    """
    strategy = settings.RECOMMENDER_STRATEGY
    top_k = settings.RECOMMENDER_TOP_K
    misery_threshold = settings.RECOMMENDER_MISERY_THRESHOLD

    if not room.constraints:
        return strategy, top_k, misery_threshold

    try:
        constraints = json.loads(room.constraints)
    except (TypeError, ValueError):
        return strategy, top_k, misery_threshold

    if not isinstance(constraints, dict):
        return strategy, top_k, misery_threshold

    raw_strategy = constraints.get("strategy")
    if isinstance(raw_strategy, str):
        try:
            strategy = AggregationStrategy(raw_strategy).value
        except ValueError:
            pass

    raw_top_k = constraints.get("top_k")
    if isinstance(raw_top_k, int) and not isinstance(raw_top_k, bool):
        top_k = min(20, max(1, raw_top_k))

    raw_threshold = constraints.get("misery_threshold")
    if isinstance(raw_threshold, (int, float)) and not isinstance(
        raw_threshold, bool
    ):
        if 1.0 <= float(raw_threshold) <= 5.0:
            misery_threshold = float(raw_threshold)

    return strategy, top_k, misery_threshold


def trigger_recommendation(
    db: Session,
    room_id: int,
    host_id: int,
) -> RecommendationRun:
    """Score the room, store the ranked Top-K and open voting."""
    room = db.query(Room).filter(Room.id == room_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    if room.host_id != host_id:
        raise HTTPException(
            status_code=403,
            detail="Only host can trigger recommendation",
        )
    if room.status != "OPEN":
        raise HTTPException(status_code=409, detail="Room is not open")

    members = (
        db.query(RoomMember)
        .filter(RoomMember.room_id == room_id)
        .order_by(RoomMember.user_id)
        .all()
    )
    if not MIN_GROUP_SIZE <= len(members) <= MAX_GROUP_SIZE:
        raise HTTPException(
            status_code=400,
            detail=(
                "Room must contain between "
                f"{MIN_GROUP_SIZE} and {MAX_GROUP_SIZE} members"
            ),
        )
    if not all(member.is_ready for member in members):
        raise HTTPException(
            status_code=400,
            detail="Not all members are ready",
        )

    movies = db.query(Movie).order_by(Movie.movielens_id).all()
    if not movies:
        raise HTTPException(
            status_code=409,
            detail="Movie catalog is empty; run scripts/seed_movies.py first",
        )
    movie_by_movielens_id = {movie.movielens_id: movie for movie in movies}

    member_user_ids = [member.user_id for member in members]
    ratings = (
        db.query(Rating, Movie.movielens_id)
        .join(Movie, Rating.movie_id == Movie.id)
        .filter(Rating.user_id.in_(member_user_ids))
        .all()
    )

    ratings_by_user: dict[int, dict[int, float]] = {
        user_id: {} for user_id in member_user_ids
    }
    for rating, movielens_id in ratings:
        ratings_by_user[rating.user_id][int(movielens_id)] = float(
            rating.rating
        )

    members_without_ratings = [
        _display_name(member.user.email)
        for member in members
        if not ratings_by_user[member.user_id]
    ]
    if members_without_ratings:
        raise HTTPException(
            status_code=400,
            detail=(
                "These members have not rated any movie yet: "
                + ", ".join(members_without_ratings)
            ),
        )

    # A movie somebody in the room has already rated is not a useful
    # suggestion for tonight, so it is removed from the candidates.
    already_rated = {
        movielens_id
        for user_ratings in ratings_by_user.values()
        for movielens_id in user_ratings
    }
    candidate_movielens_ids = [
        movie.movielens_id
        for movie in movies
        if movie.movielens_id not in already_rated
    ]
    if not candidate_movielens_ids:
        raise HTTPException(
            status_code=409,
            detail="Every movie in the catalog is already rated by the room",
        )

    try:
        recommender = load_recommender()
    except RecommenderUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    try:
        candidate_scores = recommender.score_members(
            [ratings_by_user[user_id] for user_id in member_user_ids],
            candidate_movielens_ids,
        )
    except MemberProfileTooThin as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RecommenderUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    strategy, top_k, misery_threshold = _resolve_room_settings(room)

    try:
        response = build_group_recommendation_response(
            room_id=room.id,
            member_user_ids=member_user_ids,
            candidate_scores=candidate_scores,
            strategy=strategy,
            top_k=top_k,
            misery_threshold=misery_threshold,
            member_display_names={
                member.user_id: _display_name(member.user.email)
                for member in members
            },
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if not response.recommendations:
        raise HTTPException(
            status_code=409,
            detail=(
                "No movie passed the misery threshold of "
                f"{misery_threshold:.1f}. Try the average strategy or "
                "lower the threshold."
            ),
        )

    run = RecommendationRun(
        room_id=room_id,
        status="VOTING",
        strategy=response.strategy.value,
        model_name=recommender.model_name,
        top_k=len(response.recommendations),
    )
    db.add(run)
    db.flush()

    room.status = "RUNNING"

    for item in response.recommendations:
        movie = movie_by_movielens_id[item.movie_id]
        db.add(
            RunItem(
                run_id=run.id,
                movie_id=movie.id,
                rank=item.rank,
                ai_score=item.group_score,
                minimum_score=item.minimum_score,
                disagreement=item.disagreement,
                misery_warning=item.misery_warning,
                member_scores_json=json.dumps(
                    [
                        {
                            "user_id": member_score.user_id,
                            "display_name": member_score.display_name,
                            "predicted_score": member_score.predicted_score,
                        }
                        for member_score in item.member_scores
                    ]
                ),
                explanations_json=json.dumps(list(item.explanations)),
            )
        )

    db.commit()
    db.refresh(run)
    return run
