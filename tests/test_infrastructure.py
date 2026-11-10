from __future__ import annotations

import os
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from api.main import app, RATE_LIMIT_RECORD

client = TestClient(app)


def test_health_endpoint_probes():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "model" in data
    assert "rag_index_chunks" in data
    assert "pilot_mode" in data
    assert "supported_spray_diseases" in data
    assert "arecanut:koleroga" in data["supported_spray_diseases"]
    assert data["version"] == "1.0.0"


def test_pilot_mode_guards_genai_endpoints():
    with patch.dict(os.environ, {"PILOT_MODE": "true"}):
        # GenAI advise must 404
        r_advise = client.post("/genai/advise", json={"question": "How to make bordeaux?", "language": "English"})
        assert r_advise.status_code == 404
        assert "pilot mode" in r_advise.json()["detail"].lower()

        # GenAI ask must 404
        r_ask = client.post("/genai/ask", json={"question": "How to make bordeaux?", "language": "English"})
        assert r_ask.status_code == 404
        assert "pilot mode" in r_ask.json()["detail"].lower()

        # Vision advise must 404
        r_vision = client.post(
            "/vision/advise",
            data={"question": "Check my crop", "language": "en"},
            files={"image": ("test.jpg", b"fake", "image/jpeg")}
        )
        assert r_vision.status_code == 404
        assert "pilot mode" in r_vision.json()["detail"].lower()


def test_pilot_mode_labels_ml_and_dl_as_experimental():
    with patch.dict(os.environ, {"PILOT_MODE": "true"}):
        example_soil = {"N": 90, "P": 42, "K": 43, "temperature": 25, "humidity": 80, "ph": 6.5, "rainfall": 200}
        r_predict = client.post("/predict", json=example_soil)
        assert r_predict.status_code == 200
        p_body = r_predict.json()
        assert p_body["experimental"] is True
        assert p_body["status_label"] == "experimental"
        assert "EXPERIMENTAL" in p_body["warning"]

        r_forecast = client.post("/dl/forecast", json={"commodity": "Onion", "recent_prices": [120.0] * 30})
        assert r_forecast.status_code == 200
        f_body = r_forecast.json()
        assert f_body["status_label"] == "experimental"
        assert "EXPERIMENTAL" in f_body["warning"]


def test_rate_limiting_with_forwarded_for():
    test_ip = "198.51.100.42"
    RATE_LIMIT_RECORD.pop(test_ip, None)

    # With max 60 requests/minute, let's inject 60 timestamps to simulate near-exhaustion
    import time
    now = time.time()
    RATE_LIMIT_RECORD[test_ip] = [now] * 60

    headers = {"X-Forwarded-For": test_ip}
    response = client.get("/geo/search?q=Sirsi", headers=headers)
    assert response.status_code == 429
    assert "Too many requests" in response.json()["detail"]

    # Different IP should succeed
    headers_other = {"X-Forwarded-For": "198.51.100.99"}
    response_other = client.get("/geo/search?q=Sirsi", headers=headers_other)
    assert response_other.status_code == 200

    # /health and /static are exempt from rate limiting
    resp_health = client.get("/health", headers=headers)
    assert resp_health.status_code == 200

