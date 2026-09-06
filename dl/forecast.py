from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

COMMODITIES = ("Onion", "Tomato", "Potato")
WINDOW = 30
HORIZON = 7
TEST_DAYS = 90


class PriceLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, HORIZON))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(values.unsqueeze(-1))
        return self.head(output[:, -1, :])


def load_daily_prices(source_dir: str | Path, commodities: tuple[str, ...] = COMMODITIES) -> pd.DataFrame:
    source = Path(source_dir)
    rows: list[pd.DataFrame] = []
    for path in sorted(source.rglob("*.csv")):
        for chunk in pd.read_csv(path, usecols=["Commodity", "Arrival_Date", "Modal_Price"], chunksize=100_000):
            filtered = chunk[chunk["Commodity"].str.casefold().isin({item.casefold() for item in commodities})].copy()
            if not filtered.empty:
                filtered["Commodity"] = filtered["Commodity"].str.title()
                rows.append(filtered)
    if not rows:
        raise FileNotFoundError(f"No matching commodity rows found under {source}")
    prices = pd.concat(rows, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["Arrival_Date"], errors="coerce")
    prices["price"] = pd.to_numeric(prices["Modal_Price"], errors="coerce")
    prices = prices.dropna(subset=["date", "price"])
    daily = prices.groupby(["Commodity", "date"], as_index=False)["price"].median()
    complete: list[pd.DataFrame] = []
    for commodity, group in daily.groupby("Commodity"):
        series = group.set_index("date")["price"].sort_index().resample("D").median().interpolate(limit_direction="both")
        complete.append(pd.DataFrame({"commodity": commodity, "date": series.index, "price": series.values}))
    return pd.concat(complete, ignore_index=True)


def make_windows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for end in range(WINDOW, len(values) - HORIZON + 1):
        x.append(values[end - WINDOW:end])
        y.append(values[end:end + HORIZON])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def train_series(series: pd.Series, seed: int = 42) -> dict[str, Any]:
    torch.manual_seed(seed)
    values = series.to_numpy(dtype=np.float32)
    split_date = series.index.max() - pd.Timedelta(days=TEST_DAYS - 1)
    train_values = values[series.index < split_date]
    mean, std = float(train_values.mean()), float(train_values.std() or 1.0)
    normalized = (values - mean) / std
    x, y = make_windows(normalized)
    target_dates = series.index[WINDOW: len(series) - HORIZON + 1]
    train_mask = target_dates < split_date
    test_mask = ~train_mask
    x_train, y_train = x[train_mask], y[train_mask]
    x_test, y_test = x[test_mask], y[test_mask]
    model = PriceLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model.train()
    train_x = torch.from_numpy(x_train)
    train_y = torch.from_numpy(y_train)
    for _ in range(25):
        optimizer.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        predicted_norm = model(torch.from_numpy(x_test)).numpy()
    actual = y_test * std + mean
    predicted = predicted_norm * std + mean
    persistence = np.repeat((x_test[:, -1:] * std + mean), HORIZON, axis=1)
    moving_average = np.repeat(((x_test[:, -7:].mean(axis=1) * std) + mean)[:, None], HORIZON, axis=1)
    importance = []
    base_error = metrics(actual, predicted)["mae"]
    for lag in range(WINDOW):
        ablated = x_test.copy()
        ablated[:, lag] = 0.0
        with torch.no_grad():
            ablated_pred = model(torch.from_numpy(ablated)).numpy() * std + mean
        importance.append({"lag_days_ago": WINDOW - lag, "mae_increase": float(metrics(actual, ablated_pred)["mae"] - base_error)})
    importance.sort(key=lambda item: item["mae_increase"], reverse=True)
    return {"state_dict": model.state_dict(), "mean": mean, "std": std, "last_prices": values[-WINDOW:].tolist(),
            "lstm": metrics(actual, predicted), "persistence": metrics(actual, persistence),
            "moving_average_7d": metrics(actual, moving_average), "permutation_importance": importance[:10],
            "train_end": str(series.index[series.index < split_date].max().date()), "test_start": str(split_date.date()),
            "test_rows": int(len(y_test))}


def forecast_from_artifact(model_data: dict[str, Any], prices: list[float]) -> list[float]:
    model = PriceLSTM()
    model.load_state_dict(model_data["state_dict"])
    model.eval()
    values = np.asarray(prices[-WINDOW:], dtype=np.float32)
    normalized = (values - model_data["mean"]) / model_data["std"]
    with torch.no_grad():
        result = model(torch.from_numpy(normalized[None, :])).numpy()[0]
    return (result * model_data["std"] + model_data["mean"]).clip(0).round(2).tolist()
