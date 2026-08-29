import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from cinematch.models.hybrid_ncf import HybridNCF

def train_hybrid_smoke():
    print("Starting Hybrid NCF training pipeline...")
    
    num_users, num_items, side_dim = 100, 50, 10
    num_samples = 200
    
    # Tạo dữ liệu giả lập có chứa Side Features
    users = torch.randint(0, num_users, (num_samples,))
    items = torch.randint(0, num_items, (num_samples,))
    side_feats = torch.randn(num_samples, side_dim)
    ratings = torch.randn(num_samples)
    
    dataset = TensorDataset(users, items, side_feats, ratings)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = HybridNCF(num_users=num_users, num_items=num_items, side_feature_dim=side_dim)
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