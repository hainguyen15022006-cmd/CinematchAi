import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        )
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)
