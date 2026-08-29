"""Feature contract shared by Hybrid NCF training and serving code."""

from __future__ import annotations

import torch

GENRE_FEATURE_DIM = 19
YEAR_FEATURE_DIM = 1
HISTORY_FEATURE_DIM = 19
TEXT_FEATURE_DIM = 128
HYBRID_SIDE_FEATURE_DIM = (
    GENRE_FEATURE_DIM
    + YEAR_FEATURE_DIM
    + HISTORY_FEATURE_DIM
    + TEXT_FEATURE_DIM
)


def build_hybrid_side_features(
    genres: torch.Tensor,
    normalized_year: torch.Tensor,
    history_profile: torch.Tensor,
    text_vector: torch.Tensor,
) -> torch.Tensor:
    """Validate and concatenate side features in the documented order."""

    named_features = {
        "genres": (genres, GENRE_FEATURE_DIM),
        "normalized_year": (normalized_year, YEAR_FEATURE_DIM),
        "history_profile": (history_profile, HISTORY_FEATURE_DIM),
        "text_vector": (text_vector, TEXT_FEATURE_DIM),
    }
    batch_size = genres.shape[0] if genres.ndim == 2 else None
    for name, (feature, expected_dim) in named_features.items():
        if feature.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, features]")
        if feature.shape[0] != batch_size:
            raise ValueError(f"{name} must use the same batch size")
        if feature.shape[1] != expected_dim:
            raise ValueError(
                f"{name} must have {expected_dim} columns, "
                f"received {feature.shape[1]}"
            )
        if not torch.isfinite(feature).all():
            raise ValueError(f"{name} must contain only finite values")

    return torch.cat(
        [genres, normalized_year, history_profile, text_vector],
        dim=-1,
    ).to(dtype=torch.float32)
