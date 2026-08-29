import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cinematch.features.hybrid_features import (
    HYBRID_SIDE_FEATURE_DIM,
    build_hybrid_side_features,
)
from cinematch.features.text_encoder import VietnameseTextEncoder
from cinematch.models.hybrid_ncf import HybridNCF


def train_hybrid_smoke():
    print("Starting Hybrid NCF training pipeline...")

    torch.manual_seed(42)
    num_users, num_items = 100, 50
    num_samples = 200

    # Week-one smoke data follows the real side-feature dimensions. The genre,
    # year and history values are synthetic; the Vietnamese text vectors are
    # produced by the same deterministic encoder used by the demo.
    users = torch.randint(0, num_users, (num_samples,))
    items = torch.randint(0, num_items, (num_samples,))
    genres = torch.randint(0, 2, (num_samples, 19)).float()
    normalized_year = torch.rand(num_samples, 1)
    history_profile = torch.rand(num_samples, 19)
    encoder = VietnameseTextEncoder()
    preferences = [
        "Thích phim hành động hài, có plot twist.",
        "Thích phim tình cảm nhẹ nhàng.",
        "Không thích phim kinh dị quá bạo lực.",
    ]
    text_vectors = encoder.encode_batch(
        [preferences[index % len(preferences)] for index in range(num_samples)]
    )
    side_feats = build_hybrid_side_features(
        genres,
        normalized_year,
        history_profile,
        text_vectors,
    )
    ratings = torch.empty(num_samples).uniform_(1.0, 5.0)

    dataset = TensorDataset(users, items, side_feats, ratings)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = HybridNCF(
        num_users=num_users,
        num_items=num_items,
        side_feature_dim=HYBRID_SIDE_FEATURE_DIM,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, 4):
        total_loss = 0.0
        for u, i, sf, r in dataloader:
            optimizer.zero_grad()
            preds = model(u, i, sf)
            loss = criterion(preds, r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/3 - Loss: {avg_loss:.4f}")

    print("Hybrid NCF training completed successfully.")

if __name__ == "__main__":
    train_hybrid_smoke()
