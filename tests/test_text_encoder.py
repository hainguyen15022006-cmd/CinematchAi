import json

import pytest
import torch

from cinematch.features.hybrid_features import (
    HYBRID_SIDE_FEATURE_DIM,
    build_hybrid_side_features,
)
from cinematch.features.text_encoder import (
    TextEncoderConfig,
    VietnameseTextEncoder,
)


def test_encode_returns_finite_float_vector_with_expected_dimension():
    encoder = VietnameseTextEncoder(TextEncoderConfig(dimension=128))

    vector = encoder.encode("Thích phim hành động hài, có plot twist.")

    assert vector.shape == (128,)
    assert vector.dtype == torch.float32
    assert torch.isfinite(vector).all()
    assert torch.isclose(torch.linalg.vector_norm(vector), torch.tensor(1.0))


def test_same_text_produces_the_same_vector():
    encoder = VietnameseTextEncoder()
    sentence = "Không thích phim kinh dị quá bạo lực."

    assert torch.equal(encoder.encode(sentence), encoder.encode(sentence))


def test_different_text_does_not_produce_the_same_vector():
    encoder = VietnameseTextEncoder()

    action = encoder.encode("Tôi thích phim hành động.")
    romance = encoder.encode("Tôi thích phim tình cảm.")

    assert not torch.equal(action, romance)


@pytest.mark.parametrize("text", ["", "   ", "!!!"])
def test_empty_preference_text_is_rejected(text):
    encoder = VietnameseTextEncoder()

    with pytest.raises(ValueError, match="cannot be empty"):
        encoder.encode(text)


def test_encode_batch_returns_one_row_per_sentence():
    encoder = VietnameseTextEncoder()
    vectors = encoder.encode_batch(
        ["Thích phim hài.", "Không thích phim kinh dị."]
    )

    assert vectors.shape == (2, 128)


def test_artifact_round_trip_preserves_predictions(tmp_path):
    encoder = VietnameseTextEncoder(TextEncoderConfig(dimension=64))
    artifact_path = encoder.save_artifact(tmp_path / "text_encoder.json")
    restored = VietnameseTextEncoder.load_artifact(artifact_path)
    sentence = "Thích phim khoa học viễn tưởng."

    assert restored.config == encoder.config
    assert torch.equal(restored.encode(sentence), encoder.encode(sentence))
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["dimension"] == 64


def test_hybrid_feature_builder_follows_documented_contract():
    batch_size = 3
    encoder = VietnameseTextEncoder()
    text = encoder.encode_batch(["Thích phim hài."] * batch_size)
    side_features = build_hybrid_side_features(
        genres=torch.zeros(batch_size, 19),
        normalized_year=torch.zeros(batch_size, 1),
        history_profile=torch.zeros(batch_size, 19),
        text_vector=text,
    )

    assert side_features.shape == (batch_size, HYBRID_SIDE_FEATURE_DIM)
    assert side_features.dtype == torch.float32
