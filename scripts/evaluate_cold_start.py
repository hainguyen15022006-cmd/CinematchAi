"""Evaluate MF nearest-neighbour fold-in with 5/10 known ratings."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from cinematch.data.configuration import load_data_config
from cinematch.data.io import load_processed_ratings
from cinematch.data.mapping import load_id_mappings
from cinematch.evaluation.cold_start import evaluate_cold_start
from cinematch.evaluation.configuration import load_evaluation_config
from cinematch.serving.predictor import load_torch_predictor


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/cinematch.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/baselines/seed_42/mf_checkpoint.pt",
    )
    parser.add_argument("--profile-sizes", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/evaluation/cold_start.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_data_config(args.config, project_root=PROJECT_ROOT)
    evaluation_config = load_evaluation_config(args.config)
    train = load_processed_ratings(config.paths.train)
    test = load_processed_ratings(config.paths.test)
    _users, movies = load_id_mappings(config.paths.mappings)
    predictor = load_torch_predictor(
        args.checkpoint, train, neighbor_count=evaluation_config.neighbor_count
    )
    results = []
    for size in args.profile_sizes or evaluation_config.profile_sizes:
        result = evaluate_cold_start(
            predictor,
            train,
            test,
            range(movies.size),
            profile_size=size,
            positive_threshold=evaluation_config.positive_rating_threshold,
            k=evaluation_config.top_k,
            negative_sample_size=evaluation_config.negative_sample_size,
            seed=args.seed,
            max_users=args.max_users,
        )
        results.append(result.as_dict())
        LOGGER.info(
            "profile=%d Recall@10=%.4f NDCG@10=%.4f HitRate@10=%.4f users=%d",
            size,
            result.recall_at_k,
            result.ndcg_at_k,
            result.hit_rate_at_k,
            result.evaluated_users,
        )
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "method": "train-only nearest-neighbor embedding fold-in",
                "target_user_excluded_from_neighbors": True,
                "checkpoint": str(args.checkpoint),
                "results": results,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved %s", output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
