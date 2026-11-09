from __future__ import annotations

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.weather import HourlyDataPoint, WeatherForecast
from api.geo import GeoLocation
import api.spray_engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def redirect_audit_log_to_tmp(tmp_path, monkeypatch):
    """Ensure all test invocations write audit records to a tmp path, never polluting logs/spray_audit.csv."""
    test_audit = tmp_path / "test_spray_audit.csv"
    monkeypatch.setattr(api.spray_engine, "AUDIT_LOG_FILE", test_audit)
    yield


def dummy_forecast(agreement: float = 0.85) -> WeatherForecast:
    points = [
        HourlyDataPoint(
            time=f"2026-06-15T{i:02d}:00",
            precipitation_mm=0.0,
            precipitation_probability=10.0,
            rain_mm=0.0,
            temperature_c=25.0,
            relative_humidity=70.0,
            wind_speed_kmh=8.0,
            wind_gusts_kmh=12.0,
        )
        for i in range(6, 18)
    ]
    return WeatherForecast(
        latitude=14.17,
        longitude=75.04,
        elevation=580.0,
        timezone="Asia/Kolkata",
        grid_resolution_desc="15-28 km",
        grid_resolution_min_km=15.0,
        grid_resolution_max_km=28.0,
        current_time="2026-06-15T06:00",
        hourly=points,
        model_agreement_score=agreement,
        secondary_model_used="gfs_seamless",
        past_24h_rain_mm=0.0,
        past_7d_rain_mm=10.0,
    )


def test_geo_search_endpoint():
    with patch("api.geo.search_places") as mock_search:
        mock_search.return_value = [
            GeoLocation(
                name="Sagar",
                display_name="Sagar, Shivamogga, Karnataka, India",
                latitude=14.167,
                longitude=75.043,
                state="Karnataka",
                district="Shivamogga",
            )
        ]
        resp = client.get("/geo/search?q=Sagar")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["results"][0]["name"] == "Sagar"
        assert data["results"][0]["district"] == "Shivamogga"


def test_geo_reverse_endpoint():
    with patch("api.geo.reverse_geocode") as mock_rev:
        mock_rev.return_value = GeoLocation(
            name="Sagar",
            display_name="Sagar, Shivamogga, Karnataka, India",
            latitude=14.167,
            longitude=75.043,
            state="Karnataka",
            district="Shivamogga",
        )
        resp = client.get("/geo/reverse?lat=14.167&lon=75.043")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Sagar"


def test_geo_reverse_outside_india():
    resp = client.get("/geo/reverse?lat=45.0&lon=75.0")
    # Query validation or outside envelope returns 422 or 404
    assert resp.status_code in {404, 422}


def test_weather_forecast_endpoint():
    with patch("api.weather.fetch_weather_forecast") as mock_fc:
        mock_fc.return_value = dummy_forecast()
        resp = client.get("/weather/forecast?lat=14.167&lon=75.043")
        assert resp.status_code == 200
        data = resp.json()
        assert data["latitude"] == 14.17
        assert data["model_agreement_score"] == 0.85
        assert data["secondary_model_used"] == "gfs_seamless"


def test_spray_diseases_endpoint():
    resp = client.get("/spray/diseases")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert data[0]["crop"] == "arecanut"
    assert data[0]["disease"] == "koleroga"
    assert "Bordeaux" in data[0]["primary_chemical"]


def test_spray_advise_endpoint_english():
    with patch("api.weather.fetch_weather_forecast") as mock_fc:
        mock_fc.return_value = dummy_forecast()
        req_body = {
            "crop": "arecanut",
            "disease": "koleroga",
            "latitude": 14.167,
            "longitude": 75.043,
            "spray_intent": "first",
            "language": "en",
        }
        resp = client.post("/spray/advise", json=req_body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["crop"] == "arecanut"
        assert data["disease"] == "koleroga"
        assert data["decision"] in {"SPRAY_NOW", "WAIT", "DO_NOT_SPRAY"}
        assert "Advisory defaults, not ICAR rules" in data["disclaimer"]
        assert "Safety advisory:" in data["safety_footer"]
        assert data["seasonal_context_note"] is not None
        assert "preparation_source" in data["agronomy"]
        assert len(data["citations"]) >= 2
        assert "1% Bordeaux" in data["agronomy"]["fungicide_name"] or "Bordeaux mixture" in data["agronomy"]["fungicide_name"]


def test_spray_advise_endpoint_kannada():
    with patch("api.weather.fetch_weather_forecast") as mock_fc:
        mock_fc.return_value = dummy_forecast()
        req_body = {
            "crop": "arecanut",
            "disease": "koleroga",
            "latitude": 14.167,
            "longitude": 75.043,
            "spray_intent": "first",
            "language": "kn",
        }
        resp = client.post("/spray/advise", json=req_body)
        assert resp.status_code == 200
        data = resp.json()
        assert "ಬೋರ್ಡೋ" in data["agronomy"]["fungicide_name"]
        assert "ಕೆವಿಕೆ" in data["kvk_guidance"]
        assert "ಸುರಕ್ಷತಾ ಸಲಹೆ:" in data["safety_footer"]
        assert "ಸಲಹಾತ್ಮಕ ಡೀಫಾಲ್ಟ್ ಮಿತಿಗಳು" in data["disclaimer"]
        assert "ಕಬ್ಬಿಣದ ಚಾಕು" in data["agronomy"]["quality_test"]
        assert "ಯಾಂತ್ರಿಕ ತಡೆಗೋಡೆಯಾಗಿ" in data["agronomy"]["bunch_covering"]


def test_spray_advise_fail_closed_unsupported():
    with patch("api.weather.fetch_weather_forecast") as mock_fc:
        mock_fc.return_value = dummy_forecast()
        req_body = {
            "crop": "wheat",
            "disease": "rust",
            "latitude": 14.167,
            "longitude": 75.043,
        }
        resp = client.post("/spray/advise", json=req_body)
        assert resp.status_code == 422
        assert "not supported" in resp.json()["detail"]
