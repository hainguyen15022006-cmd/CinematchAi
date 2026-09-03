"""Tests for the shared predictor contract and train-only cold start."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cinematch.models import MatrixFactorization, MostPopular, NCF
from cinematch.serving.predictor import (
    PopularityPredictor,
    TorchPredictor,
    load_torch_predictor,
)


def _training_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_index": [0, 0, 1, 1],
            "movie_index": [0, 1, 0, 1],
            "rating": [5.0, 1.0, 1.0, 5.0],
        }
    )


def _mf_predictor() -> TorchPredictor:
    model = MatrixFactorization(2, 3, embedding_dim=2, global_mean=3.0)
    with torch.no_grad():
        model.user_embedding.weight[:] = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        model.movie_embedding.weight[:] = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]
        )
        model.user_bias.weight.zero_()
        model.movie_bias.weight.zero_()
    return TorchPredictor(
        model,
        _training_ratings(),
        model_name="mf",
        number_of_users=2,
        number_of_movies=3,
        neighbor_count=1,
    )


def test_score_preserves_candidate_order() -> None:
    predictor = _mf_predictor()
    scores = predictor.score(0, [1, 0, 2])
    assert scores.tolist() == pytest.approx([2.0, 4.0, 3.0])


def test_score_rejects_non_integer_user_index() -> None:
    with pytest.raises(ValueError, match="user_index is outside"):
        _mf_predictor().score(0.5, [0, 1])  # type: ignore[arg-type]


def test_fold_in_matches_different_training_neighbors() -> None:
    predictor = _mf_predictor()
    action_fan = predictor.predict_for_new_user({0: 5.0, 1: 1.0}, [0, 1])
    romance_fan = predictor.predict_for_new_user({0: 1.0, 1: 5.0}, [0, 1])

    assert action_fan[0] > action_fan[1]
    assert romance_fan[1] > romance_fan[0]
    assert predictor.match_new_user({0: 5.0, 1: 1.0}).neighbors[0].user_index == 0
    assert predictor.match_new_user({0: 1.0, 1: 5.0}).neighbors[0].user_index == 1


def test_fold_in_rejects_unseen_movie_index() -> None:
    predictor = _mf_predictor()
    with pytest.raises(ValueError, match="outside the model catalog"):
        predictor.predict_for_new_user({99: 5.0}, [0, 1])


def test_fold_in_rejects_no_overlap_with_training() -> None:
    predictor = _mf_predictor()
    with pytest.raises(ValueError, match="no rated movie in common"):
        predictor.predict_for_new_user({2: 5.0}, [0, 1])


def test_cold_start_evaluation_can_exclude_the_simulated_user() -> None:
    predictor = _mf_predictor()
    profile = predictor.match_new_user(
        {0: 5.0, 1: 1.0}, exclude_user_indices=frozenset({0})
    )
    assert all(neighbor.user_index != 0 for neighbor in profile.neighbors)


def test_popularity_predictor_is_intentionally_user_independent() -> None:
    model = MostPopular().fit(_training_ratings())
    predictor = PopularityPredictor(model, number_of_movies=3)
    known = predictor.score(0, [0, 1])
    new = predictor.predict_for_new_user({0: 5.0}, [0, 1])
    np.testing.assert_allclose(known, new)


def test_checkpoint_round_trip_uses_shared_interface(tmp_path: Path) -> None:
    predictor = _mf_predictor()
    checkpoint = tmp_path / "mf.pt"
    torch.save(
        {
            "format_version": 1,
            "model_config": predictor.model.config(),
            "model_state_dict": predictor.model.state_dict(),
            "best_epoch": 1,
            "seed": 42,
        },
        checkpoint,
    )
    loaded = load_torch_predictor(checkpoint, _training_ratings(), neighbor_count=1)
    np.testing.assert_allclose(loaded.score(0, [0, 1]), predictor.score(0, [0, 1]))


def test_ncf_checkpoint_uses_the_same_loader(tmp_path: Path) -> None:
    model = NCF(
        num_users=2,
        num_items=3,
        embedding_dim=2,
        layers=(4, 2),
        dropout=0.0,
    )
    checkpoint = tmp_path / "ncf.pt"
    torch.save(
        {
            "format_version": 1,
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "best_epoch": 1,
            "seed": 42,
        },
        checkpoint,
    )
    predictor = load_torch_predictor(checkpoint, _training_ratings())
    assert predictor.model_name == "ncf"
    assert predictor.score(0, [0, 1]).shape == (2,)
    assert predictor.predict_for_new_user({0: 5.0}, [1, 2]).shape == (2,)


@pytest.mark.parametrize("candidates", [[], [0, 0], [3]])
def test_predictor_rejects_invalid_candidates(candidates: list[int]) -> None:
    with pytest.raises(ValueError):
        _mf_predictor().score(0, candidates)
