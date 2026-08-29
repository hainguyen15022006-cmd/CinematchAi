import torch
import torch.nn as nn
from cinematch.models.embeddings import EmbeddingLayer
from cinematch.models.mlp_layers import MLPBlock

class HybridNCF(nn.Module):   
    def __init__(
        self, 
        num_users: int, 
        num_items: int, 
        side_feature_dim: int,
        embedding_dim: int = 32, 
        layers: list = [64, 32, 16], 
        dropout: float = 0.2
    ):
        super(HybridNCF, self).__init__()
        
        self.user_embed = EmbeddingLayer(num_users, embedding_dim)
        self.item_embed = EmbeddingLayer(num_items, embedding_dim)
        
        # Tổng kích thước đầu vào MLP = User Embed + Item Embed + Side Features
        total_input_dim = (embedding_dim * 2) + side_feature_dim
        
        self.mlp = MLPBlock(input_dim=total_input_dim, layers=layers, dropout=dropout)
        self.prediction_head = nn.Linear(self.mlp.output_dim, 1)

    def forward(
        self, 
        user_indices: torch.Tensor, 
        item_indices: torch.Tensor, 
        side_features: torch.Tensor
    ) -> torch.Tensor:
        u_vector = self.user_embed(user_indices)
        i_vector = self.item_embed(item_indices)
        
        # Gộp ID embeddings cùng với side features
        x = torch.cat([u_vector, i_vector, side_features], dim=-1)
        x = self.mlp(x)
        out = self.prediction_head(x)
        return out.squeeze(-1)