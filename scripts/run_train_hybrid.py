import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cinematch.features.hybrid_features import HYBRID_SIDE_FEATURE_DIM, FeatureMask
from cinematch.models.hybrid_ncf import HybridNCF


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HybridDummyDataset(Dataset):
    def __init__(self, num_samples: int = 200, num_users: int = 100, num_items: int = 100):
        self.user_ids = torch.randint(0, num_users, (num_samples,))
        self.item_ids = torch.randint(0, num_items, (num_samples,))
        self.side_features = torch.randn(num_samples, HYBRID_SIDE_FEATURE_DIM)
        self.ratings = torch.randint(1, 6, (num_samples,)).float()

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "side_features": self.side_features[idx],
            "rating": self.ratings[idx],
        }


def main():
    set_seed(42)
    checkpoint_dir = "outputs/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_users = 100
    num_items = 100
    train_ds = HybridDummyDataset(num_samples=400, num_users=num_users, num_items=num_items)
    val_ds = HybridDummyDataset(num_samples=100, num_users=num_users, num_items=num_items)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = HybridNCF(
        num_users=num_users,
        num_items=num_items,
        side_feature_dim=HYBRID_SIDE_FEATURE_DIM,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    print("[*] Training HybridNCF started on device: cpu")

    for epoch in range(1, 4):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(batch["user_id"], batch["item_id"], batch["side_features"])
            if preds.ndim > 1:
                preds = preds.squeeze(-1)
            loss = criterion(preds, batch["rating"])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch["user_id"], batch["item_id"], batch["side_features"])
                if preds.ndim > 1:
                    preds = preds.squeeze(-1)
                val_loss += criterion(preds, batch["rating"]).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:02d}/03 - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(checkpoint_dir, "hybrid_ncf.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved best checkpoint to: {save_path}")

    manifest = {
        "model_type": "HybridNCF",
        "seed": 42,
        "num_users": num_users,
        "num_items": num_items,
        "side_feature_dim": HYBRID_SIDE_FEATURE_DIM,
        "best_val_loss": float(best_val_loss),
        "checkpoint_path": os.path.join(checkpoint_dir, "hybrid_ncf.pt"),
    }
    manifest_path = os.path.join(checkpoint_dir, "manifest_hybrid.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    # Verify Forward Pass with Feature Mask
    model.eval()
    dummy_u = torch.tensor([1, 2, 3])
    dummy_i = torch.tensor([10, 20, 30])
    dummy_side = torch.randn(3, HYBRID_SIDE_FEATURE_DIM)

    mask_no_text = FeatureMask(use_text=False)
    with torch.no_grad():
        out_full = model(dummy_u, dummy_i, dummy_side)
        out_masked = model(dummy_u, dummy_i, dummy_side, feature_mask=mask_no_text)

    print("\n--- HYBRID NCF TRAINING & MASK VERIFICATION ---")
    print(f"Manifest saved at: {manifest_path}")
    print(f"Full Feature Predictions:   {out_full.detach().cpu().numpy().squeeze()}")
    print(f"Masked Feature Predictions: {out_masked.detach().cpu().numpy().squeeze()}")

if __name__ == "__main__":
    main()