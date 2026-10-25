import os
import torch
import numpy as np
from pathlib import Path

CACHE_DIR = Path("outputs/cache")

class TextEmbedderCache:
    def __init__(self, cache_name: str = "text_embeddings_v1.pt", embedding_dim: int = 128):
        self.cache_path = CACHE_DIR / cache_name
        self.embedding_dim = embedding_dim
        os.makedirs(CACHE_DIR, exist_ok=True)

    def get_or_create_embeddings(self, num_items: int) -> torch.Tensor:
        if self.cache_path.exists():
            print(f"[*] Loaded cached text embeddings from: {self.cache_path}")
            return torch.load(self.cache_path)
        
        print("[*] Generating & Caching text embeddings (Pseudo-text/SentenceTransformer fallback)...")
        # Giả lập vector 128 dim được encode từ SentenceTransformer có seed cố định
        torch.manual_seed(42)
        embeddings = torch.randn(num_items, self.embedding_dim)
        torch.save(embeddings, self.cache_path)
        print(f"[*] Saved versioned text artifact to: {self.cache_path}")
        return embeddings