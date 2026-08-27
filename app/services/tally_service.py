from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.recommendation import RecommendationRun, Vote, RunItem

def tally_votes_and_get_winner(db: Session, run_id: int):
    run = db.query(RecommendationRun).filter(RecommendationRun.id == run_id).first()
    if not run:
        return None
    
    # Calculate sum of votes per movie
    vote_sums = db.query(
        Vote.movie_id, func.sum(Vote.vote_value).label('total_score')
    ).filter(Vote.run_id == run_id).group_by(Vote.movie_id).all()
    
    if not vote_sums:
        # No votes, pick rank 1
        top_item = db.query(RunItem).filter(RunItem.run_id == run_id).order_by(RunItem.rank).first()
        if top_item:
            run.winner_movie_id = top_item.movie_id
    else:
        # Find max score
        best_movie_id = max(vote_sums, key=lambda x: x.total_score).movie_id
        run.winner_movie_id = best_movie_id

    run.status = "FINISHED"
    run.group_score = 4.2 # mock metrics
    run.disagreement = 0.35 # mock metrics
    db.commit()
    db.refresh(run)
    return run
