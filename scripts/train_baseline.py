"""Train Most Popular, MF and GMF on the shared temporal split."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from cinematch.data.configuration import load_data_config
from cinematch.data.io import load_processed_ratings
from cinematch.data.mapping import load_id_mappings
from cinematch.evaluation.configuration import load_evaluation_config
from cinematch.models import GeneralizedMatrixFactorization, MatrixFactorization, MostPopular
from cinematch.training.baselines import (
    ExplicitRatingDataset,
    evaluate_explicit_model,
    fit_explicit_model,
    regression_metrics,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cinematch.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="One or more seeds, e.g. 42 43 44"
    )
    return parser.parse_args()


def _baseline_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = payload.get("baselines") if isinstance(payload, dict) else None
    if not isinstance(config, dict):
        raise ValueError("configuration requires a baselines mapping")
    return config


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(
    path: Path,
    *,
    model_name: str,
    seed: int,
    data_version: str,
    config_hash: str,
    checkpoint_path: Path,
    model_config: dict[str, object],
    metrics: dict[str, object],
) -> None:
    payload = {
        "schema_version": "1.0",
        "model_name": model_name,
        "version": 1,
        "seed": seed,
        "data_version": data_version,
        "config_hash": config_hash,
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "feature_layout": ["user_index", "movie_index"],
        "model_config": model_config,
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _train_one_seed(
    *,
    seed: int,
    train,
    validation,
    test,
    user_count: int,
    movie_count: int,
    baseline: dict[str, object],
    output_dir: Path,
    device: torch.device,
    data_version: str,
    config_hash: str,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batch_size = int(baseline["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        ExplicitRatingDataset(train), batch_size=batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(ExplicitRatingDataset(validation), batch_size=batch_size)
    test_loader = DataLoader(ExplicitRatingDataset(test), batch_size=batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    popularity = MostPopular(
        prior_count=float(baseline["popularity_prior_count"])
    ).fit(train)
    popular_validation = popularity.predict(
        validation["user_index"], validation["movie_index"]
    )
    popular_test = popularity.predict(test["user_index"], test["movie_index"])
    run_metrics: dict[str, object] = {
        "protocol": {
            "fit": "train",
            "model_selection": "validation",
            "final_evaluation": "test",
            "seed": seed,
            "data_version": data_version,
        },
        "most_popular": {
            "validation": regression_metrics(validation["rating"], popular_validation),
            "test": regression_metrics(test["rating"], popular_test),
        },
    }

    for model_name, model_class in {
        "mf": MatrixFactorization,
        "gmf": GeneralizedMatrixFactorization,
    }.items():
        model = model_class(
            num_users=user_count,
            num_movies=movie_count,
            embedding_dim=int(baseline["embedding_dim"]),
            global_mean=float(train["rating"].mean()),
        )
        training_result = fit_explicit_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=int(baseline["epochs"]),
            learning_rate=float(baseline["learning_rate"]),
            weight_decay=float(baseline["weight_decay"]),
            patience=int(baseline["patience"]),
            device=device,
        )
        checkpoint_path = output_dir / f"{model_name}_checkpoint.pt"
        checkpoint = {
            "format_version": 1,
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "best_epoch": training_result.best_epoch,
            "seed": seed,
            "data_version": data_version,
            "config_hash": config_hash,
        }
        torch.save(checkpoint, checkpoint_path)
        model_metrics = {
            "best_epoch": training_result.best_epoch,
            "validation": evaluate_explicit_model(model, validation_loader, device),
            "test": evaluate_explicit_model(model, test_loader, device),
            "history": list(training_result.history),
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        }
        run_metrics[model_name] = model_metrics
        _write_manifest(
            output_dir / f"{model_name}_manifest.json",
            model_name=model_name,
            seed=seed,
            data_version=data_version,
            config_hash=config_hash,
            checkpoint_path=checkpoint_path,
            model_config=model.config(),
            metrics=model_metrics,
        )
        LOGGER.info("Saved %s", checkpoint_path)

    (output_dir / "metrics.json").write_text(
        json.dumps(run_metrics, indent=2) + "\n", encoding="utf-8"
    )
    return run_metrics


def _aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for model_name in ("most_popular", "mf", "gmf"):
        partitions: dict[str, object] = {}
        for partition in ("validation", "test"):
            metric_summary: dict[str, object] = {}
            for metric in ("rmse", "mae"):
                values = np.asarray(
                    [run[model_name][partition][metric] for run in runs], dtype=np.float64
                )
                metric_summary[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "values": values.tolist(),
                }
            partitions[partition] = metric_summary
        summary[model_name] = partitions
    return summary


def main() -> int:
    args = parse_args()
    data_config = load_data_config(args.config, project_root=PROJECT_ROOT)
    baseline = _baseline_config(args.config)
    evaluation_config = load_evaluation_config(args.config)
    seeds = list(dict.fromkeys(args.seeds or evaluation_config.seeds))
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    train = load_processed_ratings(data_config.paths.train)
    validation = load_processed_ratings(data_config.paths.validation)
    test = load_processed_ratings(data_config.paths.test)
    users, movies = load_id_mappings(data_config.paths.mappings)
    output_root = (PROJECT_ROOT / str(baseline["output_dir"])).resolve()
    device = _device(args.device)
    data_version = _sha256(
        [data_config.paths.train, data_config.paths.validation, data_config.paths.test,
         data_config.paths.mappings]
    )
    config_hash = _sha256([args.config.expanduser().resolve()])

    runs: list[dict[str, object]] = []
    for seed in seeds:
        run_dir = output_root if len(seeds) == 1 else output_root / f"seed_{seed}"
        LOGGER.info("Training baselines with seed=%d on %s", seed, device)
        runs.append(
            _train_one_seed(
                seed=seed,
                train=train,
                validation=validation,
                test=test,
                user_count=users.size,
                movie_count=movies.size,
                baseline=baseline,
                output_dir=run_dir,
                device=device,
                data_version=data_version,
                config_hash=config_hash,
            )
        )

    aggregate = {
        "seeds": seeds,
        "data_version": data_version,
        "config_hash": config_hash,
        "summary": _aggregate_runs(runs),
        "runs": runs,
    }
    aggregate_path = output_root / "metrics.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved aggregate metrics to %s", aggregate_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
