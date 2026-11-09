from dotenv import load_dotenv
load_dotenv(override=True)

import base64
import os
from pathlib import Path
from functools import lru_cache
from typing import Any

from collections import defaultdict
import time
import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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
    SprayAdvisoryRequest,
    SprayAdvisoryResponse,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "models" / "krishi_sahayak.joblib"
DL_ARTIFACT_PATH = ROOT / "models" / "dl_forecaster.joblib"
RAG_INDEX_PATH = ROOT / "genai" / "index"
TEMPLATES = Jinja2Templates(directory=str(ROOT / "api" / "templates"))
STATIC_DIR = ROOT / "api" / "static"
app = FastAPI(title="Krishi Sahayak API", version="1.0.0", description="Crop recommendation with an engineered climate mismatch risk score.")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

PILOT_MODE = os.getenv("PILOT_MODE", "false").lower() in ("true", "1", "yes")

RATE_LIMIT_RECORD: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_WINDOW = 60.0


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/static") or path == "/health":
        return await call_next(request)

    client_ip = request.client.host if request.client else "127.0.0.1"
    if client_ip == "testclient":
        return await call_next(request)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW
    timestamps = [t for t in RATE_LIMIT_RECORD[client_ip] if t > cutoff]
    if len(timestamps) >= RATE_LIMIT_MAX:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please wait before retrying."}
        )
    timestamps.append(now)
    RATE_LIMIT_RECORD[client_ip] = timestamps
    return await call_next(request)


def check_pilot_mode() -> None:
    if os.getenv("PILOT_MODE", "false").lower() in ("true", "1", "yes"):
        raise HTTPException(
            status_code=404,
            detail="GenAI free-text Q&A is disabled in pilot mode. Please use /spray for verified spray advisories."
        )


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
def health() -> dict[str, Any]:
    model_ok = ARTIFACT_PATH.exists()
    rag_ok = (RAG_INDEX_PATH / "faiss.index").exists()
    status = "ok" if (model_ok and rag_ok) else "degraded"
    return {
        "status": status,
        "model": ARTIFACT_PATH.name if model_ok else None,
        "rag_index_chunks": 375 if rag_ok else 0,
        "pilot_mode": os.getenv("PILOT_MODE", "false").lower() in ("true", "1", "yes"),
        "supported_spray_diseases": ["arecanut:koleroga"],
        "version": "1.0.0",
    }


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
    from dl.forecast import forecast_from_artifact
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
def load_rag():
    if not (RAG_INDEX_PATH / "faiss.index").exists():
        raise HTTPException(status_code=503, detail="RAG index is unavailable. Run python -m genai.build_index first.")
    from genai.rag import LocalRAG
    return LocalRAG(RAG_INDEX_PATH)


@app.post("/genai/advise", response_model=AdvisoryResponse)
def advise(payload: AdvisoryRequest) -> AdvisoryResponse:
    check_pilot_mode()
    result = load_rag().answer(payload.question, payload.language)
    return AdvisoryResponse(**result)


@app.post("/genai/ask", response_model=ChatAdvisoryResponse)
def ask_expert(payload: ChatAdvisoryRequest) -> ChatAdvisoryResponse:
    check_pilot_mode()
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
    check_pilot_mode()
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
    pilot = os.getenv("PILOT_MODE", "false").lower() in ("true", "1", "yes")
    return TEMPLATES.TemplateResponse(request=request, name="index.html", context={"request": request, "pilot_mode": pilot})


@app.get("/spray", response_class=HTMLResponse)
def spray_page(request: Request):
    return TEMPLATES.TemplateResponse(request=request, name="spray_advisor.html", context={"request": request})


# ----------------- Spray Timing Advisor Endpoints -----------------

@app.get("/geo/search")
def geo_search(q: str = Query(..., min_length=2)):
    from api.geo import search_places
    results = search_places(q)
    return {"query": q, "count": len(results), "results": [r.__dict__ for r in results]}


@app.get("/geo/reverse")
def geo_reverse(lat: float = Query(..., ge=6.0, le=38.0), lon: float = Query(..., ge=68.0, le=98.0)):
    from api.geo import reverse_geocode
    loc = reverse_geocode(lat, lon)
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found or outside India envelope.")
    return loc.__dict__


@app.get("/weather/forecast")
def weather_forecast_endpoint(
    lat: float = Query(..., ge=6.0, le=38.0),
    lon: float = Query(..., ge=68.0, le=98.0),
):
    from api.weather import fetch_weather_forecast, WeatherServiceError
    from api.geo import is_within_india

    if not is_within_india(lat, lon):
        raise HTTPException(status_code=400, detail="Coordinates must be within India boundary.")
    try:
        fc = fetch_weather_forecast(lat, lon)
        return {
            "latitude": fc.latitude,
            "longitude": fc.longitude,
            "elevation_m": fc.elevation,
            "timezone": fc.timezone,
            "grid_resolution_desc": fc.grid_resolution_desc,
            "grid_resolution_km_range": [fc.grid_resolution_min_km, fc.grid_resolution_max_km],
            "current_time": fc.current_time,
            "past_24h_rain_mm": fc.past_24h_rain_mm,
            "past_48h_rain_mm": fc.past_48h_rain_mm,
            "past_7d_rain_mm": fc.past_7d_rain_mm,
            "model_agreement_score": fc.model_agreement_score,
            "secondary_model_used": fc.secondary_model_used,
            "hourly_count": len(fc.hourly),
        }
    except WeatherServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/spray/diseases")
def list_spray_diseases():
    return [
        {
            "crop": "arecanut",
            "crop_kn": "ಅಡಿಕೆ",
            "disease": "koleroga",
            "disease_aliases": ["mahali", "fruit rot"],
            "disease_kn": "ಕೊಳೆರೋಗ (ಮಹಾಲಿ)",
            "causal_organism": "Phytophthora meadii",
            "primary_chemical": "1% Neutral Bordeaux Mixture",
            "alternative_chemical": "Mandipropamid 23.4% SC (1 ml/L)",
            "status": "grounded_icar_cpcri",
        }
    ]


@app.post("/spray/advise", response_model=SprayAdvisoryResponse)
def spray_advise(request: SprayAdvisoryRequest):
    from api.geo import is_within_india
    from api.weather import fetch_weather_forecast, WeatherServiceError
    from api.spray_engine import evaluate_spray_decision, log_spray_decision, SprayIntent
    from genai.spray_explainer import build_spray_explanation

    if not is_within_india(request.latitude, request.longitude):
        raise HTTPException(status_code=400, detail="Coordinates must be within India boundary.")

    try:
        forecast = fetch_weather_forecast(request.latitude, request.longitude)
    except WeatherServiceError as exc:
        raise HTTPException(status_code=502, detail=f"Weather service unavailable: {exc}")

    try:
        result = evaluate_spray_decision(
            crop=request.crop,
            disease=request.disease,
            forecast=forecast,
            spray_intent=request.spray_intent,
            last_spray_date=request.last_spray_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    log_spray_decision(result, lat=request.latitude, lon=request.longitude)
    explanation = build_spray_explanation(result, language=request.language)
    return SprayAdvisoryResponse(**explanation)


