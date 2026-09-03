"""Evaluate CineMatch models with one reproducible Top-K protocol."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cinematch.data.configuration import load_data_config
from cinematch.data.io import load_processed_ratings
from cinematch.data.mapping import load_id_mappings
from cinematch.evaluation.topk import evaluate_topk
from cinematch.evaluation.configuration import load_evaluation_config
from cinematch.models import MostPopular
from cinematch.serving.predictor import PopularityPredictor, load_torch_predictor


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cinematch.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation" / "topk_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["most_popular", "mf", "gmf"],
        choices=("most_popular", "mf", "gmf", "ncf"),
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Repeat for each torch model; defaults to outputs/baselines/seed_<seed>.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="One or more evaluation/checkpoint seeds, e.g. 42 43 44",
    )
    parser.add_argument("--k", type=int)
    parser.add_argument("--negative-samples", type=int)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--neighbor-count", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _checkpoint_arguments(values: list[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--checkpoint must use MODEL=PATH")
        model_name, raw_path = value.split("=", 1)
        if model_name not in {"mf", "gmf", "ncf"} or not raw_path:
            raise ValueError("checkpoint model must be mf, gmf or ncf")
        path = Path(raw_path)
        checkpoints[model_name] = path if path.is_absolute() else PROJECT_ROOT / path
    return checkpoints


def _default_checkpoint(model_name: str, seed: int) -> Path:
    seeded = PROJECT_ROOT / "outputs" / "baselines" / f"seed_{seed}" / f"{model_name}_checkpoint.pt"
    legacy = PROJECT_ROOT / "outputs" / "baselines" / f"{model_name}_checkpoint.pt"
    return seeded if seeded.exists() else legacy


def _summarize(results: list[dict[str, str | int | float]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    model_names = dict.fromkeys(str(row["model_name"]) for row in results)
    for model_name in model_names:
        model_rows = [row for row in results if row["model_name"] == model_name]
        metrics: dict[str, object] = {}
        for metric in (
            "recall_at_k",
            "ndcg_at_k",
            "hit_rate_at_k",
            "coverage_at_k",
            "score_seconds",
        ):
            values = np.asarray([float(row[metric]) for row in model_rows])
            metrics[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "values": values.tolist(),
            }
        metrics["evaluated_users"] = sorted(
            {int(row["evaluated_users"]) for row in model_rows}
        )
        metrics["skipped_users"] = sorted(
            {int(row["skipped_users"]) for row in model_rows}
        )
        summary[model_name] = metrics
    return summary


def main() -> int:
    args = parse_args()
    data_config = load_data_config(args.config, project_root=PROJECT_ROOT)
    evaluation_config = load_evaluation_config(args.config)
    seeds = list(dict.fromkeys(args.seeds or evaluation_config.seeds))
    k = evaluation_config.top_k if args.k is None else args.k
    negative_samples = (
        evaluation_config.negative_sample_size
        if args.negative_samples is None
        else args.negative_samples
    )
    neighbor_count = (
        evaluation_config.neighbor_count
        if args.neighbor_count is None
        else args.neighbor_count
    )
    train = load_processed_ratings(data_config.paths.train)
    validation = load_processed_ratings(data_config.paths.validation)
    test = load_processed_ratings(data_config.paths.test)
    _users, movies = load_id_mappings(data_config.paths.mappings)
    supplied_checkpoints = _checkpoint_arguments(args.checkpoint)
    results: list[dict[str, str | int | float]] = []

    for seed in seeds:
        for model_name in dict.fromkeys(args.models):
            if model_name == "most_popular":
                model = MostPopular().fit(train)
                predictor = PopularityPredictor(model, movies.size)
            else:
                checkpoint = supplied_checkpoints.get(
                    model_name, _default_checkpoint(model_name, seed)
                )
                predictor = load_torch_predictor(
                    checkpoint,
                    train,
                    device=args.device,
                    neighbor_count=neighbor_count,
                )
                if predictor.model_name != model_name:
                    raise ValueError(
                        f"{checkpoint} contains {predictor.model_name}, expected {model_name}"
                    )
            result = evaluate_topk(
                predictor,
                train,
                validation,
                test,
                range(movies.size),
                k=k,
                positive_threshold=data_config.positive_rating_threshold,
                negative_sample_size=negative_samples,
                seed=seed,
                max_users=args.max_users,
            )
            results.append(result.as_dict())
            LOGGER.info(
                "%s seed=%d Recall@%d=%.4f NDCG@%d=%.4f HitRate@%d=%.4f Coverage=%.4f users=%d skipped=%d",
                model_name,
                seed,
                k,
                result.recall_at_k,
                k,
                result.ndcg_at_k,
                k,
                result.hit_rate_at_k,
                result.coverage_at_k,
                result.evaluated_users,
                result.skipped_users,
            )

    payload = {
        "protocol": {
            "fit": "train",
            "seen": "train+validation",
            "final_evaluation": "test",
            "positive_threshold": data_config.positive_rating_threshold,
            "k": k,
            "negative_samples": negative_samples,
            "seeds": seeds,
            "candidate_policy": "shared_per_user",
        },
        "summary": _summarize(results),
        "results": results,
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved %s", output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
