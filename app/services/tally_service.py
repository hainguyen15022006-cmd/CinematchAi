from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.recommendation import RecommendationRun, RunItem, Vote


def tally_votes_and_get_winner(
    db: Session,
    run_id: int,
) -> RecommendationRun | None:
    run = (
        db.query(RecommendationRun)
        .filter(RecommendationRun.id == run_id)
        .first()
    )
    if not run:
        return None

    items = (
        db.query(RunItem)
        .filter(RunItem.run_id == run_id)
        .order_by(RunItem.rank)
        .all()
    )
    if not items:
        return run

    # Calculate sum of votes per movie.
    vote_sums = (
        db.query(
            Vote.movie_id,
            func.sum(Vote.vote_value).label("total_score"),
        )
        .filter(Vote.run_id == run_id)
        .group_by(Vote.movie_id)
        .all()
    )

    if not vote_sums:
        winner_item = items[0]
    else:
        totals = {row.movie_id: float(row.total_score) for row in vote_sums}
        # Higher vote total wins; equal totals are broken by original rank.
        winner_item = max(
            items,
            key=lambda item: (totals.get(item.movie_id, 0.0), -item.rank),
        )

    run.winner_movie_id = winner_item.movie_id
    run.status = "FINISHED"
    run.group_score = winner_item.ai_score
    # The database mock has no per-member predictions to calculate disagreement.
    run.disagreement = None
    run.room.status = "FINISHED"
    db.commit()
    db.refresh(run)
    return run
