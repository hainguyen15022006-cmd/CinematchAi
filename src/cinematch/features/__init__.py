"""Feature encoders used by CineMatch hybrid recommendation models."""

from cinematch.features.hybrid_features import (
    HYBRID_SIDE_FEATURE_DIM,
    build_hybrid_side_features,
)
from cinematch.features.numeric_features import (
    NUMERIC_FEATURE_DIM,
    NumericFeatureArtifacts,
    ReleaseYearScaler,
    build_interaction_numeric_features,
    build_numeric_feature_artifacts,
    load_numeric_feature_artifacts,
)
from cinematch.features.text_encoder import (
    PreferenceTextEncoder,
    TextEncoderConfig,
    VietnameseTextEncoder,
)
from cinematch.features.pseudo_text import (
    TEXT_FEATURE_DIM,
    TEXT_FUSION,
    TextFeatureArtifacts,
    build_interaction_text_features,
    build_text_feature_artifacts,
    load_text_feature_artifacts,
)

__all__ = [
    "HYBRID_SIDE_FEATURE_DIM",
    "NUMERIC_FEATURE_DIM",
    "NumericFeatureArtifacts",
    "PreferenceTextEncoder",
    "ReleaseYearScaler",
    "TextEncoderConfig",
    "TextFeatureArtifacts",
    "TEXT_FEATURE_DIM",
    "TEXT_FUSION",
    "VietnameseTextEncoder",  # Backward-compatible alias of PreferenceTextEncoder.
    "build_hybrid_side_features",
    "build_interaction_numeric_features",
    "build_numeric_feature_artifacts",
    "build_interaction_text_features",
    "build_text_feature_artifacts",
    "load_numeric_feature_artifacts",
    "load_text_feature_artifacts",
]
