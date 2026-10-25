"""Feature contract shared by Hybrid NCF training and serving code."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

GENRE_FEATURE_DIM = 19
YEAR_FEATURE_DIM = 1
HISTORY_FEATURE_DIM = 19
TEXT_FEATURE_DIM = 128
TEXT_SIMILARITY_DIM = 1

HYBRID_SIDE_FEATURE_DIM = (
    GENRE_FEATURE_DIM
    + YEAR_FEATURE_DIM
    + HISTORY_FEATURE_DIM
    + TEXT_FEATURE_DIM
    + TEXT_SIMILARITY_DIM
)  # Đúng 168 chiều


def build_hybrid_side_features(
    genres: torch.Tensor,
    normalized_year: torch.Tensor,
    history_profile: torch.Tensor,
    text_vector: torch.Tensor,
    text_similarity: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Validate and concatenate side features in the documented order."""
    if text_similarity is None:
        text_similarity = torch.zeros((genres.size(0), 1), dtype=torch.float32, device=genres.device)
    elif text_similarity.ndim == 1:
        text_similarity = text_similarity.unsqueeze(1)

    named_features = {
        "genres": (genres, GENRE_FEATURE_DIM),
        "normalized_year": (normalized_year, YEAR_FEATURE_DIM),
        "history_profile": (history_profile, HISTORY_FEATURE_DIM),
        "text_vector": (text_vector, TEXT_FEATURE_DIM),
        "text_similarity": (text_similarity, TEXT_SIMILARITY_DIM),
    }

    batch_size = genres.shape[0] if genres.ndim == 2 else None
    for name, (feature, expected_dim) in named_features.items():
        if feature.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, features]")
        if feature.shape[0] != batch_size:
            raise ValueError(f"{name} must use the same batch size")
        if feature.shape[1] != expected_dim:
            raise ValueError(f"{name} expected dim {expected_dim}, got {feature.shape[1]}")

    return torch.cat(
        [genres, normalized_year, history_profile, text_vector, text_similarity],
        dim=-1,
    )


class FeatureMask(nn.Module):
    def __init__(
        self,
        use_genre: bool = True,
        use_year: bool = True,
        use_history: bool = True,
        use_text: bool = True,
        use_text_sim: bool = True,
    ):
        super().__init__()
        mask_vector = torch.ones(HYBRID_SIDE_FEATURE_DIM, dtype=torch.float32)

        idx = 0
        if not use_genre:
            mask_vector[idx : idx + GENRE_FEATURE_DIM] = 0.0
        idx += GENRE_FEATURE_DIM

        if not use_year:
            mask_vector[idx : idx + YEAR_FEATURE_DIM] = 0.0
        idx += YEAR_FEATURE_DIM

        if not use_history:
            mask_vector[idx : idx + HISTORY_FEATURE_DIM] = 0.0
        idx += HISTORY_FEATURE_DIM

        if not use_text:
            mask_vector[idx : idx + TEXT_FEATURE_DIM] = 0.0
        idx += TEXT_FEATURE_DIM

        if not use_text_sim:
            mask_vector[idx : idx + TEXT_SIMILARITY_DIM] = 0.0

        self.register_buffer("mask", mask_vector)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.mask