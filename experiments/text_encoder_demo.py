"""Run the week-one preference text encoder demonstration."""

from __future__ import annotations

import torch

from cinematch.features.text_encoder import PreferenceTextEncoder


def main() -> None:
    encoder = PreferenceTextEncoder()
    sentences = [
        "I like action comedies with a plot twist that are not too violent.",
        "I want to watch a light romantic film with friends.",
        "I do not like horror films or too many gory scenes.",
    ]
    vectors = encoder.encode_batch(sentences)

    print(f"Encoder: {encoder.config.encoder_type}")
    print(f"Vector shape: {tuple(vectors.shape)}")
    print(f"Dtype: {vectors.dtype}")
    print(f"Finite: {bool(torch.isfinite(vectors).all())}")
    print("L2 norms:", torch.linalg.vector_norm(vectors, dim=1).tolist())
    for sentence, vector in zip(sentences, vectors, strict=True):
        preview = ", ".join(f"{value:.3f}" for value in vector[:8])
        print(f"- {sentence}\n  first 8 values: [{preview}]")


if __name__ == "__main__":
    main()
