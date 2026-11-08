from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from api.spray_engine import (
    SprayDecision,
    evaluate_spray_decision,
    log_spray_decision,
    load_engine_config,
)
from api.weather import HourlyDataPoint, WeatherForecast


def make_dummy_forecast(hourly_points: list[HourlyDataPoint], agreement: float = 1.0) -> WeatherForecast:
    return WeatherForecast(
        latitude=14.17,
        longitude=75.04,
        elevation=580.0,
        timezone="Asia/Kolkata",
        grid_resolution_desc="15-28 km",
        grid_resolution_min_km=15.0,
        grid_resolution_max_km=28.0,
        current_time=hourly_points[0].time if hourly_points else "",
        hourly=hourly_points,
        model_agreement_score=agreement,
    )


def test_spray_now_immediate_dry_window():
    points = []
    # 24 hours of dry, calm weather
    for i in range(24):
        points.append(
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
        )
    forecast = make_dummy_forecast(points, agreement=0.95)
    res = evaluate_spray_decision("arecanut", "koleroga", forecast)

    assert res.decision == SprayDecision.SPRAY_NOW
    assert res.next_dry_window is not None
    assert res.next_dry_window.lead_time_hours <= 2.0
    assert "Favorable dry window available NOW" in res.decision_reason_en
    assert "1% ತಟಸ್ಥ ಬೋರ್ಡೋ ದ್ರಾವಣ" in res.decision_reason_kn
    assert res.confidence_score > 0.8
    assert "Advisory defaults, not ICAR rules" in res.disclaimer


def test_wait_when_near_term_rainy_but_dry_window_later():
    points = []
    # Hours 0-11: Heavy rain
    for i in range(12):
        points.append(
            HourlyDataPoint(
                time=f"2026-06-15T{i:02d}:00",
                precipitation_mm=5.0,
                precipitation_probability=90.0,
                rain_mm=5.0,
                temperature_c=23.0,
                relative_humidity=95.0,
                wind_speed_kmh=20.0,
                wind_gusts_kmh=30.0,
            )
        )
    # Hours 12-24: Dry break (12 hours)
    for i in range(12, 24):
        points.append(
            HourlyDataPoint(
                time=f"2026-06-15T{i:02d}:00",
                precipitation_mm=0.0,
                precipitation_probability=15.0,
                rain_mm=0.0,
                temperature_c=26.0,
                relative_humidity=65.0,
                wind_speed_kmh=7.0,
                wind_gusts_kmh=10.0,
            )
        )
    forecast = make_dummy_forecast(points, agreement=0.9)
    res = evaluate_spray_decision("arecanut", "koleroga", forecast)

    assert res.decision == SprayDecision.WAIT
    assert res.next_dry_window is not None
    assert res.next_dry_window.lead_time_hours == 12.0
    assert "Next dry window expected" in res.decision_reason_en
    assert "ಮುಂದಿನ ಒಣ ಸಮಯವು" in res.decision_reason_kn


def test_do_not_spray_persistent_rain():
    points = []
    # 48 hours of continuous rain
    for i in range(48):
        points.append(
            HourlyDataPoint(
                time=f"2026-06-15T{i:02d}:00",
                precipitation_mm=4.0,
                precipitation_probability=85.0,
                rain_mm=4.0,
                temperature_c=22.0,
                relative_humidity=95.0,
                wind_speed_kmh=18.0,
                wind_gusts_kmh=25.0,
            )
        )
    forecast = make_dummy_forecast(points, agreement=0.8)
    res = evaluate_spray_decision("arecanut", "koleroga", forecast)

    assert res.decision == SprayDecision.DO_NOT_SPRAY
    assert res.next_dry_window is None
    assert "No suitable dry window" in res.decision_reason_en
    assert "ಯಾವುದೇ ಒಣ ವಾತಾವರಣ ಕಂಡುಬರುತ್ತಿಲ್ಲ" in res.decision_reason_kn


def test_repeat_interval_wait():
    # Previous spray only 10 days ago (ICAR requires 40-45 days)
    ten_days_ago = (datetime.now(timezone.utc).date() - timedelta(days=10)).strftime("%Y-%m-%d")
    points = [HourlyDataPoint(time="2026-06-15T00:00", rain_mm=0.0, precipitation_probability=5.0)]
    forecast = make_dummy_forecast(points)

    res = evaluate_spray_decision("arecanut", "koleroga", forecast, last_spray_date=ten_days_ago)
    assert res.decision == SprayDecision.WAIT
    assert "40-45 day repeat interval" in res.decision_reason_en
    assert "40-45 ದಿನಗಳ ಅಂತರವಿರಬೇಕು" in res.decision_reason_kn


def test_fail_closed_unsupported_crop():
    forecast = make_dummy_forecast([])
    with pytest.raises(ValueError, match="not supported"):
        evaluate_spray_decision("sugarcane", "red_rot", forecast)


def test_audit_logging_to_csv(tmp_path: Path):
    audit_file = tmp_path / "test_audit.csv"
    points = [
        HourlyDataPoint(
            time="2026-06-15T00:00",
            precipitation_mm=0.0,
            precipitation_probability=10.0,
            rain_mm=0.0,
            temperature_c=25.0,
            relative_humidity=70.0,
            wind_speed_kmh=5.0,
            wind_gusts_kmh=8.0,
        )
        for _ in range(10)
    ]
    forecast = make_dummy_forecast(points, agreement=0.92)
    res = evaluate_spray_decision("arecanut", "koleroga", forecast)

    log_spray_decision(res, lat=14.167, lon=75.043, audit_path=audit_file)
    assert audit_file.exists()

    with open(audit_file, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        row = reader[0]
        assert row["crop"] == "arecanut"
        assert row["disease"] == "koleroga"
        assert row["decision"] == "SPRAY_NOW"
        assert float(row["latitude"]) == 14.167
        assert float(row["longitude"]) == 75.043
        assert row["model_agreement"] == "0.92"

