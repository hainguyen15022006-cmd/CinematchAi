import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, layers: list, dropout: float = 0.2):
        super(MLPBlock, self).__init__()
        
        modules = []
        curr_dim = input_dim
        for h_dim in layers:
            modules.append(nn.Linear(curr_dim, h_dim))
            modules.append(nn.BatchNorm1d(h_dim))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=dropout))
            curr_dim = h_dim
            
        self.network = nn.Sequential(*modules)
        self.output_dim = layers[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)