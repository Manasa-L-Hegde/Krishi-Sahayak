import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import ARTIFACT_PATH, app, load_artifacts
from api.schemas import PredictionResponse

client = TestClient(app)
EXAMPLE = {"N": 90, "P": 42, "K": 43, "temperature": 25, "humidity": 80, "ph": 6.5, "rainfall": 200}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_loading():
    assert ARTIFACT_PATH.exists()
    artifacts = load_artifacts()
    assert artifacts["features"] == ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def test_valid_prediction_matches_response_schema():
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    parsed = PredictionResponse.model_validate(response.json())
    assert parsed.crop
    assert 0 <= parsed.confidence <= 1
    assert 0 <= parsed.risk_score <= 100
    assert len(parsed.shap_contributions) == 3


def test_missing_input_is_rejected():
    payload = dict(EXAMPLE)
    del payload["rainfall"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_and_boundary_inputs():
    invalid = dict(EXAMPLE, ph=15)
    assert client.post("/predict", json=invalid).status_code == 422
    boundary = {"N": 0, "P": 0, "K": 0, "temperature": -10, "humidity": 0, "ph": 0, "rainfall": 0}
    response = client.post("/predict", json=boundary)
    assert response.status_code == 200


def test_frontend_and_docs():
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 200


def test_dl_forecast_endpoint():
    response = client.post("/dl/forecast", json={"commodity": "Onion", "recent_prices": [120.0] * 30})
    assert response.status_code == 200
    body = response.json()
    assert body["commodity"] == "Onion"
    assert body["horizon_days"] == 7
    assert len(body["forecast_prices"]) == 7
    assert body["model"] == "LSTM"


def test_dl_forecast_validation():
    assert client.post("/dl/forecast", json={"commodity": "Onion", "recent_prices": [120.0] * 29}).status_code == 422
    assert client.post("/dl/forecast", json={"commodity": "Rice", "recent_prices": [120.0] * 30}).status_code == 422
