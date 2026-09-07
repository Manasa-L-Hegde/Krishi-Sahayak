from pathlib import Path
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.schemas import AdvisoryRequest, AdvisoryResponse, ForecastRequest, ForecastResponse, PredictionRequest, PredictionResponse, ShapContribution
from dl.forecast import forecast_from_artifact
from genai.rag import LocalRAG

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "models" / "krishi_sahayak.joblib"
DL_ARTIFACT_PATH = ROOT / "models" / "dl_forecaster.joblib"
RAG_INDEX_PATH = ROOT / "genai" / "index"
TEMPLATES = Jinja2Templates(directory=str(ROOT / "api" / "templates"))
app = FastAPI(title="Krishi Sahayak API", version="1.0.0", description="Crop recommendation with an engineered climate mismatch risk score.")


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, Any]:
    if not ARTIFACT_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifacts are unavailable. Run python scripts/train_model.py first.")
    return joblib.load(ARTIFACT_PATH)


def explain_prediction(artifacts: dict[str, Any], row: pd.DataFrame, class_index: int) -> list[ShapContribution]:
    explanation = artifacts["explainer"](row)
    values = explanation.values
    if values.ndim == 3:
        values = values[0, :, class_index]
    else:
        values = values[0]
    contributions = sorted(zip(artifacts["features"], values), key=lambda pair: abs(pair[1]), reverse=True)[:3]
    return [ShapContribution(feature=feature, value=float(value), direction="supports" if value >= 0 else "pulls away")
            for feature, value in contributions]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if ARTIFACT_PATH.exists() else "degraded", "model": ARTIFACT_PATH.name}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    artifacts = load_artifacts()
    row = pd.DataFrame([[getattr(payload, feature) for feature in artifacts["features"]]], columns=artifacts["features"])
    classifier = artifacts["classifier"]
    probabilities = classifier.predict_proba(row)[0]
    class_index = int(probabilities.argmax())
    encoded_crop = classifier.classes_[class_index]
    crop = artifacts["label_encoder"].inverse_transform([encoded_crop])[0]
    risk_score = float(artifacts["risk_model"].predict(row)[0])
    return PredictionResponse(crop=str(crop), confidence=float(probabilities[class_index]),
                              risk_score=max(0.0, min(100.0, risk_score)),
                              shap_contributions=explain_prediction(artifacts, row, class_index))


@app.post("/dl/forecast", response_model=ForecastResponse)
def forecast(payload: ForecastRequest) -> ForecastResponse:
    if not DL_ARTIFACT_PATH.exists():
        raise HTTPException(status_code=503, detail="DL forecast artifact is unavailable. Run dl/train_forecaster.py first.")
    artifact = joblib.load(DL_ARTIFACT_PATH)
    commodity = next((item for item in artifact["commodities"] if item.casefold() == payload.commodity.casefold()), None)
    if commodity is None:
        allowed = ", ".join(artifact["commodities"])
        raise HTTPException(status_code=422, detail=f"Unsupported commodity. Choose one of: {allowed}")
    if any(price < 0 for price in payload.recent_prices):
        raise HTTPException(status_code=422, detail="recent_prices must contain non-negative values")
    forecast_prices = forecast_from_artifact(artifact["models"][commodity], payload.recent_prices)
    return ForecastResponse(commodity=commodity, horizon_days=artifact["horizon"], forecast_prices=forecast_prices, model="LSTM")


@lru_cache(maxsize=1)
def load_rag() -> LocalRAG:
    if not (RAG_INDEX_PATH / "faiss.index").exists():
        raise HTTPException(status_code=503, detail="RAG index is unavailable. Run python -m genai.build_index first.")
    return LocalRAG(RAG_INDEX_PATH)


@app.post("/genai/advise", response_model=AdvisoryResponse)
def advise(payload: AdvisoryRequest) -> AdvisoryResponse:
    result = load_rag().answer(payload.question, payload.language)
    return AdvisoryResponse(**result)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return TEMPLATES.TemplateResponse(request=request, name="index.html", context={"request": request})
