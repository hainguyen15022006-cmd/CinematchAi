import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class NCFTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for users, items, ratings in dataloader:
            self.optimizer.zero_grad()
            outputs = self.model(users, items)
            loss = self.criterion(outputs, ratings)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)
