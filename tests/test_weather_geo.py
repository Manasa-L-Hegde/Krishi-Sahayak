from __future__ import annotations

import os
import pytest
import httpx

from api.weather import (
    fetch_weather_forecast,
    calculate_model_agreement,
    compute_past_rain,
    parse_hourly_payload,
    round_coordinates,
    clear_weather_cache,
    WeatherServiceError,
    HourlyDataPoint,
)
from api.geo import (
    search_places,
    reverse_geocode,
    is_within_india,
    clear_geo_cache,
    GeoLocation,
)


@pytest.fixture(autouse=True)
def reset_caches():
    clear_weather_cache()
    clear_geo_cache()
    yield
    clear_weather_cache()
    clear_geo_cache()


# ----------------- Weather Tests -----------------

def sample_hourly_dict(n_hours: int = 48) -> dict:
    times = [f"2026-06-15T{i%24:02d}:00" for i in range(n_hours)]
    return {
        "time": times,
        "precipitation": [0.2 if i % 6 == 0 else 0.0 for i in range(n_hours)],
        "precipitation_probability": [40 if i % 6 == 0 else 10 for i in range(n_hours)],
        "rain": [0.2 if i % 6 == 0 else 0.0 for i in range(n_hours)],
        "temperature_2m": [26.0 + (i % 5) for i in range(n_hours)],
        "relative_humidity_2m": [80.0 - (i % 10) for i in range(n_hours)],
        "wind_speed_10m": [10.0 + (i % 4) for i in range(n_hours)],
        "wind_gusts_10m": [15.0 + (i % 5) for i in range(n_hours)],
    }


def test_round_coordinates():
    lat, lon = round_coordinates(14.16789, 75.04321, 2)
    assert lat == 14.17
    assert lon == 75.04


def test_parse_hourly_payload():
    raw = sample_hourly_dict(24)
    points = parse_hourly_payload(raw)
    assert len(points) == 24
    assert points[0].time == "2026-06-15T00:00"
    assert points[0].precipitation_mm == 0.2
    assert points[0].precipitation_probability == 40.0
    assert points[1].precipitation_mm == 0.0


def test_compute_past_rain():
    raw = sample_hourly_dict(180)
    for i in range(24):
        raw["rain"][i] = 1.0
    points = parse_hourly_payload(raw)
    r24, r48, r7d = compute_past_rain(points, current_index=48)
    assert r48 >= r24
    assert r7d >= r48


def test_calculate_model_agreement():
    raw = sample_hourly_dict(24)
    points = parse_hourly_payload(raw)

    identical = [pt.rain_mm for pt in points]
    assert calculate_model_agreement(points, identical) == 1.0

    slight = [pt.rain_mm + 0.5 for pt in points]
    score_slight = calculate_model_agreement(points, slight)
    assert 0.8 <= score_slight < 1.0

    heavy = [pt.rain_mm + 10.0 for pt in points]
    assert calculate_model_agreement(points, heavy) == 0.0


def test_fetch_weather_forecast_success():
    hourly = sample_hourly_dict(336)
    mock_payload = {
        "latitude": 14.17,
        "longitude": 75.04,
        "elevation": 580.0,
        "timezone": "Asia/Kolkata",
        "hourly": hourly,
        "hourly_ecmwf_ifs025": {
            "rain": [0.0] * 336
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=mock_payload)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        forecast = fetch_weather_forecast(14.167, 75.043, client=client, use_cache=True)
        assert forecast.latitude == 14.17
        assert len(forecast.hourly) == 336
        assert forecast.grid_resolution_min_km == 15.0
        assert forecast.grid_resolution_max_km == 28.0
        assert "15-28 km" in forecast.grid_resolution_desc
        assert 0.0 <= forecast.model_agreement_score <= 1.0

        # Verify caching
        cached_forecast = fetch_weather_forecast(14.169, 75.041, client=client, use_cache=True)
        assert cached_forecast is forecast


def test_fetch_weather_forecast_fail_closed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(WeatherServiceError):
            fetch_weather_forecast(14.17, 75.04, client=client, use_cache=False)


# ----------------- Geo Tests -----------------

def test_is_within_india():
    assert is_within_india(14.16, 75.03) is True
    assert is_within_india(51.5, -0.12) is False


def test_search_places_open_meteo():
    mock_om_response = {
        "results": [
            {
                "name": "Sagar",
                "latitude": 14.164,
                "longitude": 75.035,
                "admin1": "Karnataka",
                "admin2": "Shivamogga",
                "admin3": "Sagara",
                "postcode": "577401",
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=mock_om_response)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        results = search_places("Sagar", client=client, use_cache=True)
        assert len(results) == 1
        loc = results[0]
        assert loc.name == "Sagar"
        assert loc.state == "Karnataka"
        assert loc.district == "Shivamogga"
        assert loc.grid_resolution_min_km == 15.0
        assert loc.grid_resolution_max_km == 28.0

        cached_results = search_places("sagar", client=client, use_cache=True)
        assert cached_results == results


def test_reverse_geocode_nominatim():
    mock_nom_response = {
        "name": "Kasaragod",
        "display_name": "CPCRI, Kasaragod, Kerala, India",
        "address": {
            "town": "Kasaragod",
            "state_district": "Kasaragod",
            "state": "Kerala",
            "postcode": "671124",
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=mock_nom_response)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        loc = reverse_geocode(12.51, 74.98, client=client, use_cache=True)
        assert loc is not None
        assert loc.name == "Kasaragod"
        assert loc.state == "Kerala"
        assert loc.district == "Kasaragod"


# ----------------- Live Smoke Tests (Skipped by default) -----------------

@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="Live network test skipped by default. Run with RUN_LIVE_TESTS=1 to execute.",
)
def test_live_weather_open_meteo_karnataka():
    # Sagar, Shivamogga, Karnataka (Arecanut heartland): 14.167 N, 75.043 E
    forecast = fetch_weather_forecast(14.167, 75.043, use_cache=False)
    assert 13.5 <= forecast.latitude <= 15.0
    assert 74.0 <= forecast.longitude <= 76.0
    assert len(forecast.hourly) >= 168  # 7 days * 24h
    assert 0.0 <= forecast.model_agreement_score <= 1.0
    print(f"\n[LIVE SMOKE TEST OUTPUT]")
    print(f"Location: ({forecast.latitude}, {forecast.longitude}), Elevation: {forecast.elevation}m")
    print(f"Timezone: {forecast.timezone}, Current timestamp: {forecast.current_time}")
    print(f"Past 24h rain: {forecast.past_24h_rain_mm} mm, Past 7d rain: {forecast.past_7d_rain_mm} mm")
    print(f"Model agreement score: {forecast.model_agreement_score}")
    print(f"First 3 forecast hours:")
    for pt in forecast.hourly[168:171]:
        print(f"  {pt.time} -> Rain: {pt.rain_mm}mm ({pt.precipitation_probability}%), Temp: {pt.temperature_c}C, Wind: {pt.wind_speed_kmh}km/h")

