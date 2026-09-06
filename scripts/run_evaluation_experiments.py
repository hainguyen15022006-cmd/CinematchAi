"""Script for Top-K evaluation, Sigmoid saturation analysis, and Cold-Start simulation."""

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn

from cinematch.features.hybrid_features import HYBRID_SIDE_FEATURE_DIM, FeatureMask
from cinematch.features.text_embedder import TextEmbedderCache
from cinematch.models.hybrid_ncf import HybridNCF


def set_seed(seed: int = 42):
    """Set global random seed for deterministic evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_topk_metrics(predictions: torch.Tensor, ground_truth: torch.Tensor, k: int = 5):
    """Compute Top-K ranking metrics: Precision@K, Recall@K, NDCG@K, and MRR."""
    # Consider ratings >= 4.0 as relevant items
    relevant_mask = ground_truth >= 4.0
    num_relevant = relevant_mask.sum().item()
    
    if num_relevant == 0:
        return None  # Skip users with no relevant items in ground truth

    _, topk_indices = torch.topk(predictions, k=k)
    hits = relevant_mask[topk_indices].float()

    precision = hits.sum().item() / k
    recall = hits.sum().item() / num_relevant

    # Compute DCG & IDCG for NDCG@K
    dcg = (hits / torch.log2(torch.arange(2, k + 2).float())).sum().item()
    idcg_hits = torch.ones(min(num_relevant, k))
    idcg = (idcg_hits / torch.log2(torch.arange(2, min(num_relevant, k) + 2).float())).sum().item()
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # Compute MRR
    hit_positions = (hits > 0).nonzero(as_tuple=False)
    mrr = 1.0 / (hit_positions[0].item() + 1) if len(hit_positions) > 0 else 0.0

    return {"precision": precision, "recall": recall, "ndcg": ndcg, "mrr": mrr}


def analyze_sigmoid_saturation(predictions: torch.Tensor):
    """Analyze output distribution to detect Sigmoid saturation (near 1.0 or 5.0)."""
    flat_preds = predictions.detach().numpy()
    near_min = np.mean(flat_preds <= 1.05) * 100
    near_max = np.mean(flat_preds >= 4.95) * 100
    avg_score = np.mean(flat_preds)
    std_score = np.std(flat_preds)

    return {
        "mean_score": float(avg_score),
        "std_score": float(std_score),
        "saturation_min_pct": float(near_min),
        "saturation_max_pct": float(near_max),
    }


def evaluate_cold_start_hybrid_only(model: HybridNCF, num_items: int = 100, k: int = 5):
    """Simulate Cold-Start by zeroing out user embeddings and relying only on side features."""
    model.eval()
    generator = torch.Generator().manual_seed(42)
    embedder = TextEmbedderCache()
    item_text_embeds = embedder.get_or_create_embeddings(num_items)

    # Simulate 20 cold-start users having only 5-10 ratings
    cold_user_metrics = []
    skipped_users = 0

    with torch.no_grad():
        for user_id in range(20):
            num_ratings = random.randint(5, 10)
            item_ids = torch.randperm(num_items, generator=generator)[:num_ratings]
            ground_truth = torch.randint(1, 6, (num_ratings,), generator=generator).float()

            # Zero-out user ID to simulate pure cold-start (Hybrid-only)
            zero_user_indices = torch.zeros(num_ratings, dtype=torch.long)

            # Build side features (genres + year + history + text + text_sim)
            side_features_list = []
            for idx in item_ids:
                genres = torch.randn(19, generator=generator)
                year = torch.randn(1, generator=generator)
                history = torch.randn(19, generator=generator)
                text_emb = item_text_embeds[idx.item()]
                text_sim = torch.randn(1, generator=generator)
                side_features_list.append(torch.cat([genres, year, history, text_emb, text_sim], dim=-1))

            side_features = torch.stack(side_features_list)

            # Zero-out User Embedding weights temporarily for Hybrid-only prediction
            original_weight = model.user_embed.embedding.weight.data.clone()
            model.user_embed.embedding.weight.data.zero_()

            preds = model(zero_user_indices, item_ids, side_features)

            # Restore original weights
            model.user_embed.embedding.weight.data = original_weight

            metrics = calculate_topk_metrics(preds, ground_truth, k=k)
            if metrics is None:
                skipped_users += 1
            else:
                cold_user_metrics.append(metrics)

    avg_metrics = {
        key: float(np.mean([m[key] for m in cold_user_metrics]))
        for key in ["precision", "recall", "ndcg", "mrr"]
    }
    avg_metrics["skipped_users_count"] = skipped_users
    return avg_metrics


def main():
    set_seed(42)
    checkpoint_dir = "outputs/checkpoints"
    num_users = 100
    num_items = 100

    print("==================================================")
    print("[*] STARTING EVALUATION & COLD-START SIMULATION")
    print("==================================================")

    # Load E4 Model Checkpoint
    e4_path = os.path.join(checkpoint_dir, "e4_hybrid_with_text.pt")
    model = HybridNCF(num_users=num_users, num_items=num_items, side_feature_dim=HYBRID_SIDE_FEATURE_DIM)

    if os.path.exists(e4_path):
        model.load_state_dict(torch.load(e4_path))
        print(f"[*] Loaded trained model weights from: {e4_path}")
    else:
        print("[!] Checkpoint not found, running evaluation with initialized weights.")

    model.eval()

    # Generate synthetic validation predictions for Sigmoid Saturation analysis
    generator = torch.Generator().manual_seed(42)
    val_users = torch.randint(0, num_users, (200,), generator=generator)
    val_items = torch.randint(0, num_items, (200,), generator=generator)
    val_side = torch.randn(200, HYBRID_SIDE_FEATURE_DIM, generator=generator)

    with torch.no_grad():
        val_preds = model(val_users, val_items, val_side)

    # 1. Sigmoid Saturation Analysis
    sat_analysis = analyze_sigmoid_saturation(val_preds)
    print("\n--- SIGMOID SATURATION & OVERFIT ANALYSIS ---")
    print(f"Mean Prediction Score:  {sat_analysis['mean_score']:.4f}")
    print(f"Std Deviation Score:    {sat_analysis['std_score']:.4f}")
    print(f"Saturated Min (<=1.05): {sat_analysis['saturation_min_pct']:.2f}%")
    print(f"Saturated Max (>=4.95): {sat_analysis['saturation_max_pct']:.2f}%")

    # 2. Cold-Start Simulation (Hybrid-only)
    cold_start_results = evaluate_cold_start_hybrid_only(model, num_items=num_items, k=5)
    print("\n--- COLD-START HYBRID-ONLY RESULTS (DEMO FOR RQ2) ---")
    print(f"Precision@5: {cold_start_results['precision']:.4f}")
    print(f"Recall@5:    {cold_start_results['recall']:.4f}")
    print(f"NDCG@5:      {cold_start_results['ndcg']:.4f}")
    print(f"MRR:         {cold_start_results['mrr']:.4f}")
    print(f"Skipped Users (No Relevant Items): {cold_start_results['skipped_users_count']}")

    # Export Evaluation Manifest
    eval_manifest = {
        "sigmoid_analysis": sat_analysis,
        "cold_start_rq2_demo": cold_start_results,
    }
    manifest_path = os.path.join(checkpoint_dir, "manifest_evaluation_rq2.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(eval_manifest, f, indent=4)

    print(f"\n[*] Evaluation manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()