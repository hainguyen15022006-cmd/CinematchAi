"""Script for training and evaluating E3 and E4 experiment scenarios."""

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cinematch.features.hybrid_features import HYBRID_SIDE_FEATURE_DIM, FeatureMask
from cinematch.features.text_embedder import TextEmbedderCache
from cinematch.models.hybrid_ncf import HybridNCF


def set_seed(seed: int = 42):
    """Set global random seed for deterministic operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StandardDataset(Dataset):
    """Fixed dataset shared across all experiments to ensure deterministic evaluation."""
    def __init__(self, num_samples: int = 500, num_users: int = 100, num_items: int = 100, seed: int = 42):
        generator = torch.Generator().manual_seed(seed)
        
        self.user_ids = torch.randint(0, num_users, (num_samples,), generator=generator)
        self.item_ids = torch.randint(0, num_items, (num_samples,), generator=generator)
        
        # Load cached text embeddings artifact
        embedder = TextEmbedderCache()
        item_text_embeds = embedder.get_or_create_embeddings(num_items)
        
        side_features_list = []
        for i in range(num_samples):
            item_id = self.item_ids[i].item()
            genres = torch.randn(19, generator=generator)
            year = torch.randn(1, generator=generator)
            history = torch.randn(19, generator=generator)
            text_emb = item_text_embeds[item_id]
            text_sim = torch.randn(1, generator=generator)
            
            combined = torch.cat([genres, year, history, text_emb, text_sim], dim=-1)
            side_features_list.append(combined)
            
        self.side_features = torch.stack(side_features_list)
        self.ratings = torch.randint(1, 6, (num_samples,), generator=generator).float()

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "side_features": self.side_features[idx],
            "rating": self.ratings[idx],
        }


def run_experiment(
    experiment_name: str,
    feature_mask: FeatureMask,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_users: int = 100,
    num_items: int = 100,
    max_epochs: int = 10,
    patience: int = 2,
):
    """Execute training loop with Early Stopping for a given feature mask configuration."""
    set_seed(42)
    checkpoint_dir = "outputs/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = HybridNCF(num_users=num_users, num_items=num_items, side_feature_dim=HYBRID_SIDE_FEATURE_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"\n==================================================")
    print(f"[*] RUNNING EXPERIMENT: {experiment_name}")
    print(f"==================================================")

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(batch["user_id"], batch["item_id"], batch["side_features"], feature_mask=feature_mask)
            loss = criterion(preds, batch["rating"])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch["user_id"], batch["item_id"], batch["side_features"], feature_mask=feature_mask)
                val_loss += criterion(preds, batch["rating"]).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:02d}/{max_epochs:02d} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            file_path = os.path.join(checkpoint_dir, f"{experiment_name.lower()}.pt")
            torch.save(model.state_dict(), file_path)
            print(f"  --> Saved best checkpoint to: {file_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[*] Early stopping triggered at epoch {epoch}!")
                break

    manifest = {
        "experiment": experiment_name,
        "seed": 42,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "checkpoint_path": os.path.join(checkpoint_dir, f"{experiment_name.lower()}.pt"),
    }
    manifest_path = os.path.join(checkpoint_dir, f"manifest_{experiment_name.lower()}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    return best_val_loss


def main():
    set_seed(42)
    num_users = 100
    num_items = 100

    # Initialize shared split for deterministic comparative analysis
    train_ds = StandardDataset(num_samples=400, num_users=num_users, num_items=num_items, seed=42)
    val_ds = StandardDataset(num_samples=100, num_users=num_users, num_items=num_items, seed=100)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # E3 Scenario: Hybrid WITHOUT Text Features
    mask_e3 = FeatureMask(use_text=False, use_text_sim=False)
    e3_loss = run_experiment("e3_hybrid_no_text", mask_e3, train_loader, val_loader)

    # E4 Scenario: Hybrid WITH Full Text & Similarity Features
    mask_e4 = FeatureMask(use_text=True, use_text_sim=True)
    e4_loss = run_experiment("e4_hybrid_with_text", mask_e4, train_loader, val_loader)

    print("\n==================================================")
    print("             SUMMARY OF RESULTS                   ")
    print("==================================================")
    print(f"E3 Best Val Loss (No Text):   {e3_loss:.4f}")
    print(f"E4 Best Val Loss (With Text): {e4_loss:.4f}")
    
if __name__ == "__main__":
    main()