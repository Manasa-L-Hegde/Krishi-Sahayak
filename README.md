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
| GenAI | Implemented with evaluated fallback safeguard |
| Power BI diagnostics | Not built, decision pending |
| Azure deployment | Prepared / Not deployed |

## Problem and solution

Farmers need a compact way to inspect how soil and climate measurements resemble historical crop patterns, along with a visible warning when rainfall is unusual for the recommended crop. Krishi Sahayak serves a fixed, reproducible ML pipeline through FastAPI and exposes exported diagnostics for Power BI. The HTML frontend is intentionally small so the API contract stays inspectable.

## Architecture

`Kaggle CSV -> train_model.py -> models/ + diagnostics/*.csv -> FastAPI /predict -> Jinja2 form / Power BI diagnostics`

### Layer 1: Machine Learning (implemented)

Crop recommendation, engineered agricultural risk assessment, Random Forest/XGBoost/Logistic Regression comparison, SHAP explainability, robustness analysis, diagnostics exports, and FastAPI serving.

### Layer 2: Deep Learning (experimental)

An isolated PyTorch LSTM slice now forecasts 7-day prices for onion, tomato, and potato through `/dl/forecast`. It uses a chronological final-90-day holdout and is documented as experimental because it did not beat both naive baselines.

### Layer 3: Generative AI (implemented, grounded advisory slice)

The current slice provides English/Kannada RAG retrieval over official government/ICAR sources, a FAISS index, and optional Groq generation with a context-only prompt. It is not a multi-turn chatbot. The 15-case evaluation correctly triggered fallback for all 5 out-of-corpus questions, so this layer is marked implemented for this bounded advisory scope.

### Spray Timing Advisor (implemented, deterministic rule-based)

The Spray Timing Advisor provides live spray decision guidance (`SPRAY_NOW`, `WAIT`, `DO_NOT_SPRAY`) for Arecanut Koleroga (Mahali / Fruit Rot) by combining 7-day dual-model numerical weather forecasts (Open-Meteo ECMWF IFS 0.25° vs GFS Seamless 0.25°) with hand-curated, verbatim ICAR-CPCRI agronomic rules.

- **Zero LLM / Zero Hallucination Guarantee:** The `/spray/advise` endpoint is **strictly deterministic and rule-based**. It does **NOT** invoke an LLM and does **NOT** query the FAISS vector index. Dosages (1% Bordeaux, 1 ml/L Mandipropamid), preparation procedures, neutrality tests, and intervals are rendered directly from typed, verified `AgronomicSprayRule` dataclasses.
- **Handling Free-Text Questions:** General or free-text agronomic queries must be routed to the existing `/genai/ask` (or `/vision/advise`) endpoint, which performs semantic retrieval against the 369-chunk FAISS index and enforces fail-closed abstention when similarity falls below threshold (0.59).
- **Audit Logging & Privacy:** Every spray recommendation logs an anonymized evaluation record to `logs/spray_audit.csv`. Coordinates are rounded to ~1.1 km grid and zero farmer identity or PII is recorded.
- **Operational Scope & Known Limitations:**
  - **Grid Resolution Limits:** Weather forecasts are retrieved at NWP model grid cell resolution (~15–28 km for ECMWF IFS 0.25° and GFS). Adjacent villages within the same ~20 km grid block receive identical numerical forecasts. Microclimate variations (valley drafts, elevation shifts) require localized grower judgment.
  - **Geocoder Coverage Limits:** Village-level name searches rely on Open-Meteo's geocoding database. Smaller hamlets may not appear; farmers should use the draggable map pin or GPS location button.
  - **Scope Limitation:** Currently covers **Arecanut Koleroga (Fruit rot / Mahali)** only. Adding new crops or diseases strictly requires independently curated and verified agronomic corpora.
  - **Source Publication Age:** Core Bordeaux practices are grounded in ICAR-CPCRI Extension Publication No. 69 (1999) and Extension Folder No. 17 (1989), supplemented by recent 2024 CPCRI advisories (The Hindu). The historical chemical preparation instructions require validation by local KVK or agricultural officers.
- **Roadmap Before Real-World Pilot:**
  1. Complete API rate limiting and graceful 502/503 service fallback wrappers.
  2. Implement automated `/health` endpoint probes, a standardized Dockerfile, and GitHub Actions CI running pytest.
  3. Formal signoff from a native Kannada speaker on the UI and speech translation review table.
  4. Agronomic review by a local Krishi Vigyan Kendra (KVK) or horticulture department extension specialist before conducting field trials with farmers.

The saved ML artifact contains the selected classifier, risk regressor, label encoder, feature order, and SHAP explainer. Requests only load this artifact; they never retrain. Power BI can import the CSV files in `diagnostics/` without becoming part of the Python serving process.

### Vision Advisory (experimental)

A minimal `/vision/advise` endpoint accepts a multipart upload (image, question, and language `en`/`kn`). It calls the Meta Model API (`muse-spark-1.1` via an OpenAI-compatible client at `https://api.meta.ai/v1`) using `MODEL_API_KEY` to describe visible plant symptoms in 2–3 sentences without diagnosing. The symptom description and user question are queried against the existing FAISS RAG retriever; if retrieval similarity meets the threshold (0.59), it returns the grounded answer with sources, otherwise returning the fail-closed fallback message.

## Run locally

> **Prerequisite:** **Python 3.11.x is strictly required** (pinned: `Python 3.11.9`). Python 3.14 on Windows causes anyio blocking portal deadlocks in Starlette's `TestClient` and DLL conflicts in C-extensions. Keep the virtual environment outside OneDrive (e.g., `C:\Users\<user>\venvs\krishi`) to avoid file lock and synchronization latencies.

```powershell
# Create venv outside OneDrive (e.g. C:\Users\manas\venvs\krishi)
py -3.11 -m venv C:\Users\manas\venvs\krishi
C:\Users\manas\venvs\krishi\Scripts\Activate.ps1
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
- `POST /vision/advise` (experimental) accepts multipart image, question, and language (`en`/`kn`) for symptom extraction and grounded RAG advisory.
- `POST /spray/advise` provides grounded spray timing decisions (SPRAY_NOW, WAIT, DO_NOT_SPRAY) with daylight window analysis and ICAR citations.
- `GET /spray/diseases` lists crops and diseases supported by the ICAR-grounded spray corpus.
- `GET /weather/forecast` fetches 7-day multi-model forecast with 15–28 km resolution and model agreement score.
- `GET /geo/search` and `GET /geo/reverse` geocode location names or reverse-lookup GPS coordinates strictly within India.
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

The project is prepared for a later Linux App Service deployment but has not been deployed. The startup command is `sh startup.sh` (Gunicorn with Uvicorn workers). Keep the generated `models/` artifacts in the deployment package and do not commit the raw Kaggle CSV. Azure ML registration and App Service deployment are intentionally deferred until the integration phase .

Deployment is not claimed by this repository: Azure subscription, region quota, App Service plan, Azure ML workspace access, and deployment credentials are environment-specific.

## Limitation

This recommends based on soil and climate pattern matching from a fixed, small educational dataset. It is not live agronomic simulation and does not account for local weather forecasts, pests, disease, irrigation, cultivar, season, geography, market conditions, or field inspection. Do not present the output as a guaranteed crop recommendation.
