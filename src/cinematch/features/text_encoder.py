"""Deterministic Vietnamese text baseline for the week-one Hybrid demo.

This module deliberately uses signed feature hashing instead of a large
pretrained language model. It runs offline, has no fitted vocabulary and makes
the text-vector contract easy for the group to inspect. It is a technical
baseline, not the final semantic encoder for CineMatch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
SUPPORTED_SCHEMA_VERSION = "1.0"
SUPPORTED_ENCODER_TYPE = "signed_feature_hashing"


@dataclass(frozen=True)
class TextEncoderConfig:
    """Serializable contract for the deterministic text encoder."""

    schema_version: str = SUPPORTED_SCHEMA_VERSION
    encoder_type: str = SUPPORTED_ENCODER_TYPE
    dimension: int = 128
    lowercase: bool = True
    ngram_range: tuple[int, int] = (1, 2)

    def __post_init__(self) -> None:
        if self.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.encoder_type != SUPPORTED_ENCODER_TYPE:
            raise ValueError(f"unsupported encoder_type: {self.encoder_type}")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        minimum, maximum = self.ngram_range
        if minimum <= 0 or maximum < minimum:
            raise ValueError("ngram_range must be positive and ordered")


class VietnameseTextEncoder:
    """Encode Vietnamese preference text into a fixed float32 vector."""

    def __init__(self, config: TextEncoderConfig | None = None) -> None:
        self.config = config or TextEncoderConfig()

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def _tokens(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFC", text.strip())
        if self.config.lowercase:
            normalized = normalized.lower()
        return TOKEN_PATTERN.findall(normalized)

    def _features(self, tokens: list[str]) -> list[str]:
        minimum, maximum = self.config.ngram_range
        features: list[str] = []
        for size in range(minimum, maximum + 1):
            features.extend(
                " ".join(tokens[start : start + size])
                for start in range(len(tokens) - size + 1)
            )
        return features

    def encode(self, text: str) -> torch.Tensor:
        """Return one L2-normalized vector or reject empty preference text."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens = self._tokens(text)
        if not tokens:
            raise ValueError("preference text cannot be empty")

        vector = torch.zeros(self.dimension, dtype=torch.float32)
        for feature in self._features(tokens):
            digest = hashlib.blake2b(
                feature.encode("utf-8"),
                digest_size=8,
                person=b"cinematch",
            ).digest()
            value = int.from_bytes(digest, byteorder="big", signed=False)
            index = value % self.dimension
            sign = 1.0 if value & (1 << 63) else -1.0
            vector[index] += sign

        norm = float(torch.linalg.vector_norm(vector))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("preference text did not produce a valid vector")
        return vector / norm

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Encode a non-empty list into shape ``[batch, dimension]``."""

        if not texts:
            raise ValueError("texts cannot be empty")
        return torch.stack([self.encode(text) for text in texts])

    def artifact_payload(self) -> dict[str, Any]:
        """Return the JSON-compatible artifact contract."""

        payload = asdict(self.config)
        payload["ngram_range"] = list(self.config.ngram_range)
        return payload

    def save_artifact(self, path: str | Path) -> Path:
        """Persist encoder configuration; hashing itself has no learned weights."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.artifact_payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load_artifact(cls, path: str | Path) -> "VietnameseTextEncoder":
        """Recreate an encoder from a validated week-one artifact."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {
            "schema_version",
            "encoder_type",
            "dimension",
            "lowercase",
            "ngram_range",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"text encoder artifact is missing: {sorted(missing)}")
        payload["ngram_range"] = tuple(payload["ngram_range"])
        return cls(TextEncoderConfig(**payload))
