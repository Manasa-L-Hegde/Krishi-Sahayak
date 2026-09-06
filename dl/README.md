# Deep Learning Layer

## Status: Implemented as an experimental Phase 2 slice

This module adds a small PyTorch LSTM price forecaster without changing the existing ML pipeline or `/predict` API. It is deliberately experimental: the current results do not justify presenting the LSTM as the preferred forecasting model.

## Dataset

Source: Kaggle **Daily Market Prices of Commodity India (2001-2026)** by `khandelwalmanas`: https://www.kaggle.com/datasets/khandelwalmanas/daily-commodity-prices-india. The source contains Agmarknet-style records with `Commodity`, `Arrival_Date`, and `Modal_Price`. The trainer filters Onion, Tomato, and Potato, then takes the median modal price per calendar day and interpolates missing dates. Raw source data is not committed.

## Method and leakage control

`dl/train_forecaster.py` uses a 30-day lag window to predict the next 7 days. The final 90 calendar days are held out as test data; this is a chronological time split, never a random split, so future prices are not leaked into training. Scaling statistics are calculated only from the pre-test training period. The LSTM has one layer with 32 hidden units and is intentionally small.

## Baselines and measured results

The metrics below were generated from the downloaded dataset with seed 42. Each commodity has 84 test windows. MAE/RMSE are in the dataset's price units.

| Commodity | Persistence MAE / RMSE | 7-day mean MAE / RMSE | LSTM MAE / RMSE |
| --- | ---: | ---: | ---: |
| Onion | 313.26 / 568.61 | 245.22 / 421.62 | 324.58 / 504.97 |
| Tomato | 110.57 / 163.99 | 157.29 / 230.34 | 147.02 / 190.85 |
| Potato | 253.42 / 394.37 | 249.86 / 367.65 | 277.94 / 401.59 |

The LSTM does **not** beat both baselines for any commodity. It beats the 7-day mean for tomato but loses to persistence; it loses to both baselines for onion and potato. It remains in the repository as a reproducible diagnose -> baseline -> model -> re-evaluate experiment, not as a claim of forecasting superiority. The training cost is higher than both naive baselines, and 84 test windows per commodity are not enough for strong generalization claims.

## Permutation-style lag importance

For each lag position, the trainer zeros that input across the test windows and measures the increase in LSTM MAE. The largest measured contributors are exported to `diagnostics/dl_permutation_importance.csv`. This is an ablation-based importance signal, not causal evidence. Because the model underperforms the baselines, these lag rankings should be treated as diagnostic rather than operational guidance.

## API

`POST /dl/forecast` accepts a commodity (`Onion`, `Tomato`, or `Potato`) and at least 30 recent daily prices, then returns a 7-day LSTM forecast. The endpoint returns HTTP 503 until `models/dl_forecaster.joblib` has been generated with:

```powershell
python -m dl.train_forecaster --source-dir path/to/daily-commodity-prices-india/csv
```

The endpoint is integrated into the existing FastAPI application; no second API is created.
