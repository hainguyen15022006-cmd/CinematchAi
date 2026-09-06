"""Inference predictor module for loading trained Hybrid NCF checkpoints."""

import json
import os
import torch
from typing import List, Dict, Any, Optional

from cinematch.models.hybrid_ncf import HybridNCF
from cinematch.features.hybrid_features import HYBRID_SIDE_FEATURE_DIM, FeatureMask


class HybridPredictor:
    """Predictor class for loading models and serving Top-K recommendations."""

    def __init__(self, checkpoint_path: str, manifest_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        
        # Load Manifest metadata
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
            
        self.num_users = self.manifest.get("num_users", 100)
        self.num_items = self.manifest.get("num_items", 100)
        
        # Initialize architecture and load state dict
        self.model = HybridNCF(
            num_users=self.num_users,
            num_items=self.num_items,
            side_feature_dim=HYBRID_SIDE_FEATURE_DIM,
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    def predict_user_item_scores(
        self,
        user_id: int,
        candidate_item_ids: List[int],
        side_features: torch.Tensor,
        feature_mask: Optional[FeatureMask] = None,
    ) -> torch.Tensor:
        """Compute predicted rating scores for candidate items."""
        num_candidates = len(candidate_item_ids)
        user_tensor = torch.full((num_candidates,), user_id, dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(candidate_item_ids, dtype=torch.long).to(self.device)
        side_tensor = side_features.to(self.device)

        with torch.no_grad():
            scores = self.model(
                user_indices=user_tensor,
                item_indices=item_tensor,
                side_features=side_tensor,
                feature_mask=feature_mask,
            )
        return scores

    def recommend_top_k(
        self,
        user_id: int,
        candidate_item_ids: List[int],
        side_features: torch.Tensor,
        k: int = 5,
        feature_mask: Optional[FeatureMask] = None,
    ) -> List[Dict[str, Any]]:
        """Rank candidates and return Top-K recommendations."""
        scores = self.predict_user_item_scores(user_id, candidate_item_ids, side_features, feature_mask)
        topk_scores, topk_indices = torch.topk(scores, k=min(k, len(candidate_item_ids)))

        recommendations = []
        for score, idx in zip(topk_scores, topk_indices):
            recommendations.append({
                "item_id": int(candidate_item_ids[idx.item()]),
                "predicted_rating": float(score.item()),
            })
        return recommendations