"""Feature encoders used by CineMatch hybrid recommendation models."""

from cinematch.features.hybrid_features import (
    HYBRID_SIDE_FEATURE_DIM,
    build_hybrid_side_features,
)
from cinematch.features.text_encoder import (
    PreferenceTextEncoder,
    TextEncoderConfig,
    VietnameseTextEncoder,
)

__all__ = [
    "HYBRID_SIDE_FEATURE_DIM",
    "PreferenceTextEncoder",
    "TextEncoderConfig",
    "VietnameseTextEncoder",  # Backward-compatible alias of PreferenceTextEncoder.
    "build_hybrid_side_features",
]
