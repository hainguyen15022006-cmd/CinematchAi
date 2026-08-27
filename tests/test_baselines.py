"""Tests for Most Popular, MF, GMF and shared training utilities."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from cinematch.models import (
    GeneralizedMatrixFactorization,
    MatrixFactorization,
    MostPopular,
)
from cinematch.training.baselines import (
    ExplicitRatingDataset,
    fit_explicit_model,
    regression_metrics,
)


def make_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_index": [0, 0, 1, 1, 2, 2],
            "movie_index": [0, 1, 0, 2, 0, 2],
            "rating": [5.0, 4.0, 5.0, 2.0, 4.0, 2.0],
        }
    )


def test_most_popular_ranks_smoothed_items() -> None:
    model = MostPopular(prior_count=2).fit(make_ratings())

    ranking = model.recommend(user_index=99, top_k=3)

    assert [item.movie_index for item in ranking] == [0, 1, 2]
    assert ranking[0].rating_count == 3


def test_most_popular_excludes_seen_items() -> None:
    model = MostPopular(prior_count=2).fit(make_ratings())

    ranking = model.recommend(user_index=0, top_k=3, exclude_seen=True)

    assert [item.movie_index for item in ranking] == [2]


def test_most_popular_falls_back_to_global_mean() -> None:
    ratings = make_ratings()
    model = MostPopular().fit(ratings)

    prediction = model.predict([0], [999])

    assert prediction[0] == pytest.approx(ratings["rating"].mean())


@pytest.mark.parametrize(
    "model_class",
    [MatrixFactorization, GeneralizedMatrixFactorization],
)
def test_neural_baseline_forward_and_backward(model_class: type[nn.Module]) -> None:
    model = model_class(num_users=3, num_movies=4, embedding_dim=8)
    users = torch.tensor([0, 1, 2], dtype=torch.long)
    movies = torch.tensor([1, 2, 3], dtype=torch.long)

    output = model(users, movies)
    output.square().mean().backward()

    assert output.shape == (3,)
    assert model.user_embedding.weight.grad is not None
    assert model.movie_embedding.weight.grad is not None


@pytest.mark.parametrize(
    "model_class",
    [MatrixFactorization, GeneralizedMatrixFactorization],
)
def test_neural_baseline_rejects_misaligned_inputs(
    model_class: type[nn.Module],
) -> None:
    model = model_class(num_users=3, num_movies=4)

    with pytest.raises(ValueError, match="must align"):
        model(torch.tensor([0, 1]), torch.tensor([1]))


def test_regression_metrics_clamp_predictions() -> None:
    metrics = regression_metrics([1.0, 5.0], [-10.0, 10.0])

    assert metrics == {"mse": 0.0, "rmse": 0.0, "mae": 0.0}


def test_mf_smoke_training_improves_validation_rmse() -> None:
    torch.manual_seed(7)
    ratings = pd.DataFrame(
        {
            "user_index": [0, 0, 1, 1],
            "movie_index": [0, 1, 0, 1],
            "rating": [5.0, 1.0, 1.0, 5.0],
        }
    )
    dataset = ExplicitRatingDataset(ratings)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = MatrixFactorization(
        num_users=2,
        num_movies=2,
        embedding_dim=4,
        global_mean=3.0,
    )

    result = fit_explicit_model(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        epochs=80,
        learning_rate=0.05,
        weight_decay=0.0,
        patience=20,
        device=torch.device("cpu"),
    )

    assert result.best_validation_rmse < 0.5
    assert result.best_epoch > 1


def test_most_popular_does_not_mutate_training_data() -> None:
    ratings = make_ratings()
    original = ratings.copy(deep=True)

    MostPopular().fit(ratings)

    pd.testing.assert_frame_equal(ratings, original)
    assert np.isfinite(ratings["rating"]).all()
