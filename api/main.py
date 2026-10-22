from dotenv import load_dotenv
load_dotenv(override=True)

import base64
import os
from pathlib import Path
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.schemas import (
    AdvisoryRequest,
    AdvisoryResponse,
    ChatAdvisoryRequest,
    ChatAdvisoryResponse,
    ForecastRequest,
    ForecastResponse,
    PredictionRequest,
    PredictionResponse,
    ShapContribution,
)
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


@app.get("/dl/commodities-data")
def dl_commodities_data() -> dict[str, Any]:
    if not DL_ARTIFACT_PATH.exists():
        raise HTTPException(status_code=503, detail="DL forecaster is unavailable.")
    artifact = joblib.load(DL_ARTIFACT_PATH)
    return {
        "commodities": list(artifact["commodities"]),
        "default_prices": {c: artifact["models"][c]["last_prices"] for c in artifact["commodities"]},
        "horizon": artifact["horizon"],
    }


@lru_cache(maxsize=1)
def load_rag() -> LocalRAG:
    if not (RAG_INDEX_PATH / "faiss.index").exists():
        raise HTTPException(status_code=503, detail="RAG index is unavailable. Run python -m genai.build_index first.")
    return LocalRAG(RAG_INDEX_PATH)


@app.post("/genai/advise", response_model=AdvisoryResponse)
def advise(payload: AdvisoryRequest) -> AdvisoryResponse:
    result = load_rag().answer(payload.question, payload.language)
    return AdvisoryResponse(**result)


@app.post("/genai/ask", response_model=ChatAdvisoryResponse)
def ask_expert(payload: ChatAdvisoryRequest) -> ChatAdvisoryResponse:
    result = load_rag().ask_expert(payload.question, payload.language)
    return ChatAdvisoryResponse(**result)


def describe_plant_symptoms(image_bytes: bytes, content_type: str = "image/jpeg") -> str:
    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="MODEL_API_KEY is not configured.")
    from openai import OpenAI
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=api_key)
    model = os.getenv("META_MODEL", "muse-spark-1.1")
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime = content_type or "image/jpeg"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe visible plant symptoms in 2-3 sentences. Don't diagnose.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64_image}"},
                    },
                ],
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()


@app.post("/vision/advise", response_model=AdvisoryResponse)
async def vision_advise(
    image: UploadFile = File(...),
    question: str = Form(...),
    language: str = Form("en"),
) -> AdvisoryResponse:
    norm_lang = language.strip().lower()
    if norm_lang not in {"en", "kn", "english", "kannada"}:
        raise HTTPException(status_code=422, detail="Unsupported language. Choose 'en' or 'kn'.")
    image_bytes = await image.read()
    symptoms = describe_plant_symptoms(image_bytes, image.content_type or "image/jpeg")
    combined_query = f"{question}\n{symptoms}".strip()
    target_language = "kannada" if norm_lang in {"kn", "kannada"} else "english"
    result = load_rag().answer(combined_query, language=target_language)
    return AdvisoryResponse(**result)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return TEMPLATES.TemplateResponse(request=request, name="index.html", context={"request": request})

