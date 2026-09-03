"""Validated settings for shared Top-K and cold-start evaluation."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EvaluationConfig:
    positive_rating_threshold: float
    top_k: int
    negative_sample_size: int
    seeds: tuple[int, ...]
    neighbor_count: int
    profile_sizes: tuple[int, ...]


def load_evaluation_config(path: Path) -> EvaluationConfig:
    payload = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    evaluation = payload.get("evaluation") if isinstance(payload, dict) else None
    if not isinstance(evaluation, dict):
        raise ValueError("configuration requires an evaluation mapping")
    cold_start = evaluation.get("cold_start")
    if not isinstance(cold_start, dict):
        raise ValueError("evaluation requires a cold_start mapping")
    threshold = evaluation.get("positive_rating_threshold")
    top_k = evaluation.get("top_k")
    negatives = evaluation.get("negative_sample_size")
    seeds = evaluation.get("seeds")
    neighbors = cold_start.get("neighbor_count")
    profile_sizes = cold_start.get("profile_sizes")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("positive_rating_threshold must be numeric")
    if not 1.0 <= float(threshold) <= 5.0:
        raise ValueError("positive_rating_threshold must be within [1, 5]")
    for name, value in (("top_k", top_k), ("neighbor_count", neighbors)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(negatives, bool) or not isinstance(negatives, int) or negatives < 0:
        raise ValueError("negative_sample_size must be a non-negative integer")
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in seeds
    ):
        raise ValueError("evaluation.seeds must be non-negative integers")
    if not isinstance(profile_sizes, list) or not profile_sizes or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in profile_sizes
    ):
        raise ValueError("cold_start.profile_sizes must be positive integers")
    if len(set(seeds)) != len(seeds) or len(set(profile_sizes)) != len(profile_sizes):
        raise ValueError("evaluation seeds and profile sizes must not contain duplicates")
    return EvaluationConfig(
        positive_rating_threshold=float(threshold),
        top_k=top_k,
        negative_sample_size=negatives,
        seeds=tuple(seeds),
        neighbor_count=neighbors,
        profile_sizes=tuple(profile_sizes),
    )
