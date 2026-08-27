"""Train Most Popular, MF and GMF on the shared temporal split."""

import argparse
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
from cinematch.models import (
    GeneralizedMatrixFactorization,
    MatrixFactorization,
    MostPopular,
)
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
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
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


def main() -> int:
    args = parse_args()
    data_config = load_data_config(args.config, project_root=PROJECT_ROOT)
    baseline = _baseline_config(args.config)
    seed = data_config.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train = load_processed_ratings(data_config.paths.train)
    validation = load_processed_ratings(data_config.paths.validation)
    test = load_processed_ratings(data_config.paths.test)
    users, movies = load_id_mappings(data_config.paths.mappings)
    output_dir = (PROJECT_ROOT / str(baseline["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)

    batch_size = int(baseline["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        ExplicitRatingDataset(train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        ExplicitRatingDataset(validation), batch_size=batch_size
    )
    test_loader = DataLoader(
        ExplicitRatingDataset(test), batch_size=batch_size
    )

    popularity = MostPopular(
        prior_count=float(baseline["popularity_prior_count"])
    ).fit(train)
    popular_validation = popularity.predict(
        validation["user_index"], validation["movie_index"]
    )
    popular_test = popularity.predict(
        test["user_index"], test["movie_index"]
    )
    metrics: dict[str, object] = {
        "protocol": {
            "fit": "train",
            "model_selection": "validation",
            "final_evaluation": "test",
            "seed": seed,
        },
        "most_popular": {
            "validation": regression_metrics(
                validation["rating"], popular_validation
            ),
            "test": regression_metrics(test["rating"], popular_test),
        },
    }

    model_classes = {
        "mf": MatrixFactorization,
        "gmf": GeneralizedMatrixFactorization,
    }
    for model_name, model_class in model_classes.items():
        model = model_class(
            num_users=users.size,
            num_movies=movies.size,
            embedding_dim=int(baseline["embedding_dim"]),
            global_mean=float(train["rating"].mean()),
        )
        result = fit_explicit_model(
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
        torch.save(
            {
                "format_version": 1,
                "model_config": model.config(),
                "model_state_dict": model.state_dict(),
                "best_epoch": result.best_epoch,
                "seed": seed,
            },
            checkpoint_path,
        )
        metrics[model_name] = {
            "best_epoch": result.best_epoch,
            "validation": evaluate_explicit_model(
                model, validation_loader, device
            ),
            "test": evaluate_explicit_model(model, test_loader, device),
            "history": list(result.history),
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        }
        LOGGER.info("Saved %s", checkpoint_path)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Saved metrics to %s", metrics_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
