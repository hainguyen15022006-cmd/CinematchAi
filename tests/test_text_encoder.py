import json
import unicodedata

import pytest
import torch

from cinematch.features.hybrid_features import (
    HYBRID_SIDE_FEATURE_DIM,
    build_hybrid_side_features,
)
from cinematch.features.text_encoder import (
    TextEncoderConfig,
    PreferenceTextEncoder,
)


def test_encode_returns_finite_float_vector_with_expected_dimension():
    encoder = PreferenceTextEncoder(TextEncoderConfig(dimension=128))

    vector = encoder.encode("I like action comedies with a plot twist.")

    assert vector.shape == (128,)
    assert vector.dtype == torch.float32
    assert torch.isfinite(vector).all()
    assert torch.isclose(torch.linalg.vector_norm(vector), torch.tensor(1.0))


def test_same_text_produces_the_same_vector():
    encoder = PreferenceTextEncoder()
    sentence = "I do not like very violent horror films."

    assert torch.equal(encoder.encode(sentence), encoder.encode(sentence))


def test_different_text_does_not_produce_the_same_vector():
    encoder = PreferenceTextEncoder()

    action = encoder.encode("I like action films.")
    romance = encoder.encode("I like romantic films.")

    assert not torch.equal(action, romance)


def test_vietnamese_preference_text_is_supported():
    # Users may still enter their preferences in Vietnamese. The encoder is
    # language-agnostic (NFC normalization + Unicode-aware tokenization), so
    # accented text must produce a valid vector and be stable across
    # Unicode normalization forms.
    encoder = PreferenceTextEncoder()

    vector = encoder.encode(
        "Tôi thích phim hành động hài và không quá bạo lực."
    )
    decomposed = unicodedata.normalize(
        "NFD", "Tôi thích phim hành động hài và không quá bạo lực."
    )

    assert vector.shape == (encoder.dimension,)
    assert torch.isfinite(vector).all()
    assert torch.isclose(torch.linalg.vector_norm(vector), torch.tensor(1.0))
    assert torch.equal(vector, encoder.encode(decomposed))


@pytest.mark.parametrize("text", ["", "   ", "!!!"])
def test_empty_preference_text_is_rejected(text):
    encoder = PreferenceTextEncoder()

    with pytest.raises(ValueError, match="cannot be empty"):
        encoder.encode(text)


def test_encode_batch_returns_one_row_per_sentence():
    encoder = PreferenceTextEncoder()
    vectors = encoder.encode_batch(
        ["I like comedies.", "I do not like horror films."]
    )

    assert vectors.shape == (2, 128)


def test_artifact_round_trip_preserves_predictions(tmp_path):
    encoder = PreferenceTextEncoder(TextEncoderConfig(dimension=64))
    artifact_path = encoder.save_artifact(tmp_path / "text_encoder.json")
    restored = PreferenceTextEncoder.load_artifact(artifact_path)
    sentence = "I like science fiction films."

    assert restored.config == encoder.config
    assert torch.equal(restored.encode(sentence), encoder.encode(sentence))
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["dimension"] == 64


def test_hybrid_feature_builder_follows_documented_contract():
    batch_size = 3
    encoder = PreferenceTextEncoder()
    text = encoder.encode_batch(["I like comedies."] * batch_size)
    side_features = build_hybrid_side_features(
        genres=torch.zeros(batch_size, 19),
        normalized_year=torch.zeros(batch_size, 1),
        history_profile=torch.zeros(batch_size, 19),
        text_vector=text,
    )

    assert side_features.shape == (batch_size, HYBRID_SIDE_FEATURE_DIM)
    assert side_features.dtype == torch.float32
