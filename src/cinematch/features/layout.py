from dataclasses import dataclass
from typing import Dict


@dataclass
class FeatureLayout:
    user_emb_dim: int = 32
    item_emb_dim: int = 32
    genre_dim: int = 19
    year_dim: int = 1
    genre_history_dim: int = 19
    text_emb_dim: int = 128
    include_text_similarity: bool = True

    @property
    def total_side_features_dim(self) -> int:
        base_dim = (
            self.genre_dim
            + self.year_dim
            + self.genre_history_dim
            + self.text_emb_dim
        )
        if self.include_text_similarity:
            return base_dim + 1
        return base_dim

    def get_layout_summary(self) -> Dict[str, int]:
        return {
            "user_emb_dim": self.user_emb_dim,
            "item_emb_dim": self.item_emb_dim,
            "side_features_dim": self.total_side_features_dim,
            "has_text_similarity": self.include_text_similarity,
        }


DEFAULT_LAYOUT = FeatureLayout(include_text_similarity=True)