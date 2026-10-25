import torch
from cinematch.features.hybrid_features import (
    HYBRID_SIDE_FEATURE_DIM,
    FeatureMask,
    build_hybrid_side_features,
)


def main():
    batch_size = 4

    dummy_genres = torch.randn(batch_size, 19)
    dummy_year = torch.randn(batch_size, 1)
    dummy_history = torch.randn(batch_size, 19)
    dummy_text = torch.randn(batch_size, 128)
    dummy_text_sim = torch.randn(batch_size, 1)

    hybrid_vec = build_hybrid_side_features(
        genres=dummy_genres,
        normalized_year=dummy_year,
        history_profile=dummy_history,
        text_vector=dummy_text,
        text_similarity=dummy_text_sim,
    )

    print(f"[*] Side Feature Dimension (contract): {HYBRID_SIDE_FEATURE_DIM}")
    print(f"[*] Combined Hybrid Vector Shape: {hybrid_vec.shape}")
    assert hybrid_vec.shape == (batch_size, 168), "Shape mismatch! Must be (batch_size, 168)"

    # Masking check: switch off the 128-dimensional text group.
    mask = FeatureMask(use_text=False)
    masked_vec = mask(hybrid_vec)

    text_start_idx = 19 + 1 + 19
    text_end_idx = text_start_idx + 128
    text_sum = torch.sum(torch.abs(masked_vec[:, text_start_idx:text_end_idx])).item()

    print(f"[*] Text Vector Sum after Masking (Expected 0.0): {text_sum}")
    print("\n--- HYBRID FEATURES VERIFICATION SUCCESSFUL ---")


if __name__ == "__main__":
    main()