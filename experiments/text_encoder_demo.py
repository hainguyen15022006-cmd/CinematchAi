"""Run the week-one Vietnamese preference encoder demonstration."""

from __future__ import annotations

import torch

from cinematch.features.text_encoder import VietnameseTextEncoder


def main() -> None:
    encoder = VietnameseTextEncoder()
    sentences = [
        "Thích phim hành động hài, có plot twist, không quá bạo lực.",
        "Muốn xem phim tình cảm nhẹ nhàng cùng bạn bè.",
        "Không thích phim kinh dị hoặc quá nhiều cảnh máu me.",
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
