"""Train the Layer 2 price forecaster with a chronological holdout.

Example:
python dl/train_forecaster.py --source-dir path/to/daily-commodity-prices-india
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from dl.forecast import COMMODITIES, train_series, load_daily_prices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, help="Extracted Kaggle dataset directory")
    parser.add_argument("--output", default="models/dl_forecaster.joblib")
    args = parser.parse_args()
    daily = load_daily_prices(args.source_dir)
    models = {}
    for commodity in COMMODITIES:
        series = daily[daily["commodity"].str.casefold() == commodity.casefold()].set_index("date")["price"]
        if len(series) < 150:
            raise ValueError(f"Insufficient history for {commodity}: {len(series)} daily observations")
        models[commodity] = train_series(series)
    artifact = {"models": models, "commodities": list(COMMODITIES), "window": 30, "horizon": 7,
                "dataset": "Daily Market Prices of Commodity India (2001-2026)",
                "dataset_source": "https://www.kaggle.com/datasets/khandelwalmanas/daily-commodity-prices-india"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    diagnostics = []
    for commodity, result in models.items():
        for method in ("persistence", "moving_average_7d", "lstm"):
            diagnostics.append({"commodity": commodity, "model": method, **result[method], "test_rows": result["test_rows"],
                                "test_start": result["test_start"], "train_end": result["train_end"]})
    Path("diagnostics").mkdir(exist_ok=True)
    pd.DataFrame(diagnostics).to_csv("diagnostics/dl_forecast_metrics.csv", index=False)
    importance = [{"commodity": commodity, **item} for commodity, result in models.items() for item in result["permutation_importance"]]
    pd.DataFrame(importance).to_csv("diagnostics/dl_permutation_importance.csv", index=False)
    Path("dl/training_summary.json").write_text(json.dumps(artifact | {"models": models}, default=str, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
