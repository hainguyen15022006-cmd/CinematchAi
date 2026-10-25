"""Feature contract shared by Hybrid NCF training and serving code.

The Data handoff freezes a 167-dimensional side-feature block:
19 movie genres, one normalized release year, 19 train-only user genre
history values and a 128-dimensional user-movie text interaction vector.

Week 2-3 model work adds one optional user-movie text-similarity scalar.
When that scalar is supplied the block becomes 168 dimensions. The scalar
is an opt-in extension so the frozen 167-dimensional data contract, its
schemas and the artifacts already generated from it stay valid.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

GENRE_FEATURE_DIM = 19
YEAR_FEATURE_DIM = 1
HISTORY_FEATURE_DIM = 19
TEXT_FEATURE_DIM = 128
TEXT_SIMILARITY_DIM = 1

# Frozen Data contract: hybrid-v1-167.
HYBRID_SIDE_FEATURE_DIM = (
    GENRE_FEATURE_DIM
    + YEAR_FEATURE_DIM
    + HISTORY_FEATURE_DIM
    + TEXT_FEATURE_DIM
)

# Optional extension used by the week 2-3 Hybrid experiments.
HYBRID_SIDE_FEATURE_DIM_WITH_SIMILARITY = (
    HYBRID_SIDE_FEATURE_DIM + TEXT_SIMILARITY_DIM
)

SUPPORTED_SIDE_FEATURE_DIMS = (
    HYBRID_SIDE_FEATURE_DIM,
    HYBRID_SIDE_FEATURE_DIM_WITH_SIMILARITY,
)


def build_hybrid_side_features(
    genres: torch.Tensor,
    normalized_year: torch.Tensor,
    history_profile: torch.Tensor,
    text_vector: torch.Tensor,
    text_similarity: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Validate and concatenate side features in the documented order.

    Args:
        genres: Movie genre flags, shape [batch, 19].
        normalized_year: Train-normalized release year, shape [batch, 1].
        history_profile: Train-only user genre history, shape [batch, 19].
        text_vector: User-movie text interaction, shape [batch, 128].
        text_similarity:
            Optional user-movie text similarity scalar, shape [batch, 1]
            or [batch]. Omit it to keep the frozen 167-dimensional
            contract; supply it to produce 168 dimensions.

    Returns:
        The concatenated side-feature block, 167 or 168 columns wide.

    Raises:
        ValueError: If any input has the wrong rank, batch size or width.
    """

    named_features = [
        ("genres", genres, GENRE_FEATURE_DIM),
        ("normalized_year", normalized_year, YEAR_FEATURE_DIM),
        ("history_profile", history_profile, HISTORY_FEATURE_DIM),
        ("text_vector", text_vector, TEXT_FEATURE_DIM),
    ]

    if text_similarity is not None:
        if text_similarity.ndim == 1:
            text_similarity = text_similarity.unsqueeze(1)
        named_features.append(
            ("text_similarity", text_similarity, TEXT_SIMILARITY_DIM)
        )

    if genres.ndim != 2:
        raise ValueError("genres must have shape [batch, features]")
    batch_size = genres.shape[0]

    for name, feature, expected_dim in named_features:
        if feature.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, features]")
        if feature.shape[0] != batch_size:
            raise ValueError(f"{name} must use the same batch size")
        if feature.shape[1] != expected_dim:
            raise ValueError(
                f"{name} expected dim {expected_dim}, "
                f"got {feature.shape[1]}"
            )

    return torch.cat(
        [feature for _, feature, _ in named_features],
        dim=-1,
    )


class FeatureMask(nn.Module):
    """Zero out whole feature groups for ablation experiments.

    The mask adapts to the width of the block it receives, so the same
    module works with the 167-dimensional contract and with the
    168-dimensional text-similarity extension.
    """

    def __init__(
        self,
        use_genre: bool = True,
        use_year: bool = True,
        use_history: bool = True,
        use_text: bool = True,
        use_text_sim: bool = True,
    ) -> None:
        super().__init__()
        self.use_genre = use_genre
        self.use_year = use_year
        self.use_history = use_history
        self.use_text = use_text
        self.use_text_sim = use_text_sim

    def build_mask(
        self,
        side_feature_dim: int = HYBRID_SIDE_FEATURE_DIM,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the 0/1 mask vector for one side-feature width."""
        if side_feature_dim not in SUPPORTED_SIDE_FEATURE_DIMS:
            raise ValueError(
                "side_feature_dim must be one of "
                f"{SUPPORTED_SIDE_FEATURE_DIMS}, got {side_feature_dim}"
            )

        mask_vector = torch.ones(
            side_feature_dim, dtype=dtype, device=device
        )

        groups = [
            (self.use_genre, GENRE_FEATURE_DIM),
            (self.use_year, YEAR_FEATURE_DIM),
            (self.use_history, HISTORY_FEATURE_DIM),
            (self.use_text, TEXT_FEATURE_DIM),
        ]
        if side_feature_dim == HYBRID_SIDE_FEATURE_DIM_WITH_SIMILARITY:
            groups.append((self.use_text_sim, TEXT_SIMILARITY_DIM))

        start = 0
        for keep, width in groups:
            if not keep:
                mask_vector[start : start + width] = 0.0
            start += width

        return mask_vector

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, features]")
        mask = self.build_mask(
            side_feature_dim=features.shape[1],
            device=features.device,
            dtype=features.dtype,
        )
        return features * mask
