import torch
from torch.utils.data import Dataset

class NCFDataset(Dataset):
    def __init__(self, users: torch.Tensor, items: torch.Tensor, ratings: torch.Tensor):
        self.users = users
        self.items = items
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]