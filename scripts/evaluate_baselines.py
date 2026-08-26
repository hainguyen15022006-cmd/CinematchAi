"""Print the reproducible baseline metrics produced by training."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "baselines" / "metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    print("model\tvalidation_rmse\tvalidation_mae\ttest_rmse\ttest_mae")
    for model_name in ("most_popular", "mf", "gmf"):
        model = payload[model_name]
        validation = model["validation"]
        test = model["test"]
        print(
            f"{model_name}\t{validation['rmse']:.4f}\t"
            f"{validation['mae']:.4f}\t{test['rmse']:.4f}\t"
            f"{test['mae']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
