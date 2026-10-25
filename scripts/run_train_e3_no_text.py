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


class E3Dataset(Dataset):
    def __init__(self, num_samples: int = 400, num_users: int = 100, num_items: int = 100):
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

    train_ds = E3Dataset(num_samples=400)
    val_ds = E3Dataset(num_samples=100)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = HybridNCF(num_users=100, num_items=100, side_feature_dim=HYBRID_SIDE_FEATURE_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # THIẾT LẬP MASK CHO E3: TẮT TEXT EMBEDDING VÀ TEXT SIMILARITY
    e3_mask = FeatureMask(use_text=False, use_text_sim=False)

    print("[*] Starting E3 Training Phase (Hybrid WITHOUT Text Features)...")
    best_val_loss = float("inf")

    for epoch in range(1, 4):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(
                batch["user_id"], 
                batch["item_id"], 
                batch["side_features"], 
                feature_mask=e3_mask
            )
            loss = criterion(preds, batch["rating"])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                preds = model(
                    batch["user_id"], 
                    batch["item_id"], 
                    batch["side_features"], 
                    feature_mask=e3_mask
                )
                val_loss += criterion(preds, batch["rating"]).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:02d}/03 - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(checkpoint_dir, "e3_hybrid_no_text.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved E3 checkpoint to: {save_path}")

    manifest = {
        "experiment": "E3_Hybrid_No_Text",
        "seed": 42,
        "use_text": False,
        "use_text_sim": False,
        "best_val_loss": float(best_val_loss),
        "checkpoint_path": os.path.join(checkpoint_dir, "e3_hybrid_no_text.pt"),
    }
    manifest_path = os.path.join(checkpoint_dir, "manifest_e3_no_text.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print("\n--- E3 TRAINING COMPLETED SUCCESSFULLY ---")
    print(f"Manifest E3 saved at: {manifest_path}")

if __name__ == "__main__":
    main()