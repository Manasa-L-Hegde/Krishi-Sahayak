# Krishi Sahayak

**ML + DL + GenAI for intelligent agriculture advisory.**

Krishi Sahayak is a multi-layer agriculture advisory platform combining Machine Learning, Deep Learning, and Generative AI for crop recommendation, risk assessment, forecasting, and farmer assistance.

**Repository:** `krishi-sahayak`  
**GitHub description:** Multi-layer agriculture advisory platform combining Machine Learning, Deep Learning, and Generative AI for crop recommendation, risk assessment, forecasting, and farmer assistance.

## Project status

| Layer | Status |
| --- | --- |
| ML | Implemented |
| DL | Experimental forecasting slice implemented |
| GenAI | Planned / Not started |
| Azure deployment | Prepared / Not deployed |

## Problem and solution

Farmers need a compact way to inspect how soil and climate measurements resemble historical crop patterns, along with a visible warning when rainfall is unusual for the recommended crop. Krishi Sahayak serves a fixed, reproducible ML pipeline through FastAPI and exposes exported diagnostics for Power BI. The HTML frontend is intentionally small so the API contract stays inspectable.

## Architecture

`Kaggle CSV -> train_model.py -> models/ + diagnostics/*.csv -> FastAPI /predict -> Jinja2 form / Power BI diagnostics`

### Layer 1: Machine Learning (implemented)

Crop recommendation, engineered agricultural risk assessment, Random Forest/XGBoost/Logistic Regression comparison, SHAP explainability, robustness analysis, diagnostics exports, and FastAPI serving.

### Layer 2: Deep Learning (experimental)

An isolated PyTorch LSTM slice now forecasts 7-day prices for onion, tomato, and potato through `/dl/forecast`. It uses a chronological final-90-day holdout and is documented as experimental because it did not beat both naive baselines.

### Layer 3: Generative AI (planned)

Future RAG-based farmer advisory grounded in reliable agricultural and government knowledge, with multilingual assistance. No LLM, vector search, RAG pipeline, or chatbot is implemented yet.

The saved ML artifact contains the selected classifier, risk regressor, label encoder, feature order, and SHAP explainer. Requests only load this artifact; they never retrain. Power BI can import the CSV files in `diagnostics/` without becoming part of the Python serving process.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/train_model.py
uvicorn api.main:app --reload
```

The training script downloads `atharvaingle/crop-recommendation-dataset` through KaggleHub when `data/raw/Crop_recommendation.csv` is absent. You can also place the CSV there and run `python scripts/train_model.py --data path/to/file.csv`.

## Current ML implementation

### Inputs and outputs

Inputs are N, P, K, temperature, humidity, soil pH, and rainfall. Outputs are a recommended crop, classification confidence, a 0-100 climate mismatch risk score, and the top three SHAP feature contributions with direction. Inference uses the exact saved seven-column feature order from training; no scaling or other preprocessing is applied to the tree models.

## Data and license note

Source: Kaggle, [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset), attributed to atharvaingle. The dataset is commonly redistributed in educational repositories; verify the current Kaggle page's license and terms before redistributing the CSV or deploying a public derivative. This repository does not commit the raw data.

## Original risk contribution

The dataset supplies crop labels but no risk label. For each crop, the training split computes rainfall mean and standard deviation. A row's continuous target is `clip(abs(rainfall - crop_mean) / (2 * crop_std) * 100, 0, 100)`. The two-sigma scale makes the result readable as 0-100: higher means farther from the crop's observed rainfall pattern. A separate RandomForestRegressor learns this target from the seven numeric inputs. This is a transparent heuristic, not an agronomic threshold; all crop rainfall statistics are training-only.

## Modeling and evaluation

`train_model.py` compares RandomForest, XGBoost, and Logistic Regression. The champion is selected by held-out macro-F1, not accuracy, because 22 crop classes can hide weak per-crop behavior behind an attractive accuracy number. It also writes:

- `models/diagnostics.json`: model comparison, per-crop precision/recall/F1, top confusion pairs, robustness flips, and risk regression MAE/RMSE/R².
- `models/shap_summary.png`: global SHAP beeswarm for the champion.
- `models/krishi_sahayak.joblib`: classifier, risk regressor, label encoder, and SHAP explainer.

SHAP explains the served classifier for the specific input and is not evidence of causality. The global beeswarm is saved as `models/shap_summary.png`. The `diagnostics/` directory also contains full classification metrics, a confusion matrix, crop-level metrics, robustness results, the improvement experiment, and crop-level risk metrics.

The robustness check perturbs each numeric feature by -10% and +10% one at a time across the held-out set and reports the fraction of predictions that flip. Treat that result as a fragility indicator, not a confidence interval. The generated diagnostics preserve the exact results rather than claiming a pre-written accuracy headline.

Current held-out run (random state 42): RandomForest_700Trees is selected as the champion with accuracy 0.9932, macro precision 0.9935, macro recall 0.9932, macro-F1 0.9932, and weighted-F1 0.9932. The top confused pairs are blackgram/maize, jute/rice, and lentil/mothbeans, with one error each. Risk validation is MAE 7.20, RMSE 10.11, and R² 0.835. These results are specific to this fixed split and should be regenerated whenever the data or modeling code changes.

## Diagnose -> explain -> improve -> re-evaluate

The baseline Random Forest used 350 trees. The evidence-based improvement targeted the measured robustness weakness by increasing the ensemble to 700 trees, with the same split and random seed. Macro-F1 stayed at 0.9932 (absolute change 0.0000), while the perturbation flip rate fell from 1.737% to 1.623% (absolute change -0.114 percentage points). This is a small practical robustness gain, not a statistically significant claim, and costs approximately twice the tree-building work. The result is retained transparently in `diagnostics/improvement_experiment.csv`.

Per-crop precision, recall, and F1 are exported in `diagnostics/per_crop_metrics.csv`; risk MAE and RMSE by crop are in `diagnostics/risk_metrics_by_crop.csv` where each crop has 20 held-out rows. The risk R²=0.835 reflects internal consistency with its own training-data-derived definition, not independent agronomic validation.

## API endpoints

- `GET /health` reports artifact availability.
- `POST /predict` accepts the seven numeric inputs and returns the crop, confidence, risk score, and top-three SHAP contributions.
- `GET /` serves the plain farmer-facing HTML form.
- `GET /docs` provides Swagger UI with example values pre-filled for Try it out.

## Technology stack

### Current

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, FastAPI, Jinja2, PyTorch, pytest, joblib, and Power BI-compatible CSV exports.

### Planned

An LLM/API provider for GenAI, vector search and RAG components, and Azure services for final deployment.

## Development roadmap

1. **Phase 1 — ML:** Crop recommendation, risk assessment, SHAP explainability, and API/UI.
2. **Phase 2 — DL:** Agricultural time-series forecasting experiment started under `dl/`; improve only after stronger validation.
3. **Phase 3 — GenAI:** RAG-based farmer advisory chatbot.
4. **Phase 4 — Integration:** Unified platform and cloud deployment.

## Azure readiness (not deployed)

The project is prepared for a later Linux App Service deployment but has not been deployed. The startup command is `sh startup.sh` (Gunicorn with Uvicorn workers). Keep the generated `models/` artifacts in the deployment package and do not commit the raw Kaggle CSV. Azure ML registration and App Service deployment are intentionally deferred until the integration phase.

Deployment is not claimed by this repository: Azure subscription, region quota, App Service plan, Azure ML workspace access, and deployment credentials are environment-specific.

## Limitation

This recommends based on soil and climate pattern matching from a fixed, small educational dataset. It is not live agronomic simulation and does not account for local weather forecasts, pests, disease, irrigation, cultivar, season, geography, market conditions, or field inspection. Do not present the output as a guaranteed crop recommendation.
