import os
import time
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class NCFTrainerV2:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[str] = None,
        patience: int = 3,
        checkpoint_dir: str = "outputs/checkpoints",
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            user_ids = batch["user_id"].to(self.device)
            item_ids = batch["item_id"].to(self.device)
            targets = batch["rating"].to(self.device).float()

            side_features = batch.get("side_features")
            if side_features is not None:
                side_features = side_features.to(self.device)

            self.optimizer.zero_grad()

            if side_features is not None:
                outputs = self.model(user_ids, item_ids, side_features)
            else:
                outputs = self.model(user_ids, item_ids)

            loss = self.criterion(outputs.squeeze(), targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(targets)
            total_samples += len(targets)

        return total_loss / total_samples if total_samples > 0 else 0.0

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                user_ids = batch["user_id"].to(self.device)
                item_ids = batch["item_id"].to(self.device)
                targets = batch["rating"].to(self.device).float()

                side_features = batch.get("side_features")
                if side_features is not None:
                    side_features = side_features.to(self.device)

                if side_features is not None:
                    outputs = self.model(user_ids, item_ids, side_features)
                else:
                    outputs = self.model(user_ids, item_ids)

                outputs = outputs.squeeze()
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * len(targets)
                total_mae += torch.sum(torch.abs(outputs - targets)).item()
                total_samples += len(targets)

        mse = total_loss / total_samples if total_samples > 0 else 0.0
        rmse = mse**0.5
        mae = total_mae / total_samples if total_samples > 0 else 0.0

        return {"val_loss": mse, "val_rmse": rmse, "val_mae": mae}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        save_name: str = "ncf_best.pt",
    ) -> Dict[str, Any]:
        history = {"train_loss": [], "val_loss": [], "val_rmse": []}

        print(f"[*] Training started on device: {self.device}")
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            elapsed = time.time() - start_time

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_rmse"].append(val_metrics["val_rmse"])

            print(
                f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
                f"Train Loss: {train_loss:.4f} | "
                f"Val RMSE: {val_metrics['val_rmse']:.4f} | "
                f"Val MAE: {val_metrics['val_mae']:.4f}"
            )

            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
                save_path = os.path.join(self.checkpoint_dir, save_name)
                torch.save(self.model.state_dict(), save_path)
                print(f"  --> Saved best checkpoint to: {save_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"[!] Early stopping triggered at epoch {epoch}.")
                    break

        return history