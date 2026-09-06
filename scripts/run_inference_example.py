"""Script demonstrating model loading and top-k recommendation inference."""

import os
import torch
from cinematch.predictors.hybrid_predictor import HybridPredictor
from cinematch.features.hybrid_features import HYBRID_SIDE_FEATURE_DIM, FeatureMask
from cinematch.features.text_embedder import TextEmbedderCache


def main():
    checkpoint_path = "outputs/checkpoints/e4_hybrid_with_text.pt"
    manifest_path = "outputs/checkpoints/manifest_e4_hybrid_with_text.json"

    if not os.path.exists(checkpoint_path) or not os.path.exists(manifest_path):
        print("[!] Error: Checkpoint or manifest missing. Please train E4 model first.")
        return

    print("==================================================")
    print("[*] LOADING MODEL CHECKPOINT & MANIFEST")
    print("==================================================")
    predictor = HybridPredictor(checkpoint_path=checkpoint_path, manifest_path=manifest_path)
    print(f"[*] Successfully loaded model from: {checkpoint_path}")

    # Prepare candidate items and dummy side features for inference
    num_candidates = 10
    candidate_items = list(range(num_candidates))
    
    embedder = TextEmbedderCache()
    text_embeds = embedder.get_or_create_embeddings(num_candidates)

    side_features_list = []
    generator = torch.Generator().manual_seed(42)
    for i in range(num_candidates):
        genres = torch.randn(19, generator=generator)
        year = torch.randn(1, generator=generator)
        history = torch.randn(19, generator=generator)
        text_emb = text_embeds[i]
        text_sim = torch.randn(1, generator=generator)
        side_features_list.append(torch.cat([genres, year, history, text_emb, text_sim], dim=-1))

    side_features = torch.stack(side_features_list)

    # 1. Inference for Normal User
    print("\n--- Top-5 Recommendations (Normal User ID: 5) ---")
    recs_normal = predictor.recommend_top_k(user_id=5, candidate_item_ids=candidate_items, side_features=side_features, k=5)
    for rank, rec in enumerate(recs_normal, 1):
        print(f"Rank {rank}: Item ID {rec['item_id']:<2} | Predicted Rating: {rec['predicted_rating']:.4f}")

    # 2. Inference for Cold-Start User (No Text Features Masked vs Hybrid-only)
    print("\n--- Top-5 Recommendations (Cold-Start User / Hybrid-Only) ---")
    mask_hybrid_only = FeatureMask(use_text=True, use_text_sim=True)
    recs_cold = predictor.recommend_top_k(user_id=0, candidate_item_ids=candidate_items, side_features=side_features, k=5, feature_mask=mask_hybrid_only)
    for rank, rec in enumerate(recs_cold, 1):
        print(f"Rank {rank}: Item ID {rec['item_id']:<2} | Predicted Rating: {rec['predicted_rating']:.4f}")

    print("\n==================================================")
    print("[*] INFERENCE DEMO COMPLETED SUCCESSFULLY")
    print("==================================================")


if __name__ == "__main__":
    main()