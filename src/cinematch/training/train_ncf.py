import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cinematch.data.ncf_dataset import NCFDataset
from cinematch.models.ncf import NCF
from cinematch.training.ncf_trainer import NCFTrainer

def run_smoke_train():
    print(" Starting NCF training pipeline...")

    torch.manual_seed(42)
    num_users, num_items, dataset_size = 200, 100, 600

    users = torch.randint(0, num_users, (dataset_size,))
    items = torch.randint(0, num_items, (dataset_size,))
    ratings = torch.FloatTensor(dataset_size).uniform_(1.0, 5.0)

    dataset = NCFDataset(users, items, ratings)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = NCF(num_users=num_users, num_items=num_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    trainer = NCFTrainer(model, optimizer, criterion)

    for epoch in range(1, 4):
        avg_loss = trainer.train_one_epoch(dataloader)
        print(f"Epoch {epoch}/3 - Loss: {avg_loss:.4f}")

    print(" NCF training completed successfully.")

if __name__ == "__main__":
    run_smoke_train()
