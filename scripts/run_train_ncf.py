import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cinematch.models.ncf import NCF
from cinematch.predictors.ncf_predictor_v2 import NCFPredictorV2
from cinematch.training.ncf_trainer_v2 import NCFTrainerV2


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DummyDataset(Dataset):
    def __init__(self, num_samples: int = 200, num_users: int = 100, num_items: int = 100):
        self.user_ids = torch.randint(0, num_users, (num_samples,))
        self.item_ids = torch.randint(0, num_items, (num_samples,))
        self.ratings = torch.randint(1, 6, (num_samples,)).float()

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
        }


def main():
    set_seed(42)
    checkpoint_dir = "outputs/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_users = 100
    num_items = 100
    train_ds = DummyDataset(num_samples=400, num_users=num_users, num_items=num_items)
    val_ds = DummyDataset(num_samples=100, num_users=num_users, num_items=num_items)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = NCF(num_users=num_users, num_items=num_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    trainer = NCFTrainerV2(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        patience=2,
        checkpoint_dir=checkpoint_dir,
    )

    history = trainer.fit(train_loader, val_loader, epochs=3, save_name="ncf_baseline.pt")

    manifest = {
        "model_type": "NCF_Baseline",
        "seed": 42,
        "num_users": num_users,
        "num_items": num_items,
        "best_val_loss": float(trainer.best_val_loss),
        "checkpoint_path": os.path.join(checkpoint_dir, "ncf_baseline.pt"),
    }
    manifest_path = os.path.join(checkpoint_dir, "manifest_ncf.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    predictor = NCFPredictorV2(model=model)
    test_users = [1, 2, 3]
    test_items = [10, 20, 30]
    preds = predictor.predict(test_users, test_items)

    print("\n--- NCF TRAINING & PREDICTION VERIFICATION ---")
    print(f"Manifest saved at: {manifest_path}")
    print(f"Sample Predictions shape: {preds.shape}")
    print(f"Sample Predictions values: {preds}")
    


if __name__ == "__main__":
    main()