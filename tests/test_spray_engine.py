from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from api.spray_engine import (
    SprayDecision,
    SprayIntent,
    evaluate_spray_decision,
    log_spray_decision,
    load_engine_config,
)
from api.weather import HourlyDataPoint, WeatherForecast


def make_dummy_forecast(
    hourly_points: list[HourlyDataPoint],
    agreement: float = 1.0,
    past_24h_rain: float = 0.0,
    past_7d_rain: float = 0.0,
) -> WeatherForecast:
    return WeatherForecast(
        latitude=14.1678,
        longitude=75.0432,
        elevation=580.0,
        timezone="Asia/Kolkata",
        grid_resolution_desc="15-28 km",
        grid_resolution_min_km=15.0,
        grid_resolution_max_km=28.0,
        current_time=hourly_points[0].time if hourly_points else "",
        hourly=hourly_points,
        model_agreement_score=agreement,
        past_24h_rain_mm=past_24h_rain,
        past_7d_rain_mm=past_7d_rain,
    )


def test_spray_now_immediate_daylight_window():
    points = []
    # 12 hours of dry, calm weather during daylight (06:00 to 18:00)
    for i in range(6, 18):
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
    forecast = make_dummy_forecast(points, agreement=0.95, past_7d_rain=15.0)
    res = evaluate_spray_decision("arecanut", "koleroga", forecast, spray_intent=SprayIntent.FIRST)

    assert res.decision == SprayDecision.SPRAY_NOW
    assert res.next_dry_window is not None
    assert res.next_dry_window.lead_time_hours <= 2.0
    assert "Favorable daylight dry window available NOW" in res.decision_reason_en
    assert "1% ತಟಸ್ಥ ಬೋರ್ಡೋ ದ್ರಾವಣ" in res.decision_reason_kn
    assert res.confidence_score > 0.8
    assert "Advisory defaults, not ICAR rules" in res.disclaimer


def test_daylight_filter_rejects_night():
    points = []
    # Night hours (00:00 to 05:00) are dry, but daylight begins at 06:00
    for i in range(0, 12):
        points.append(
            HourlyDataPoint(
                time=f"2026-06-15T{i:02d}:00",
                precipitation_mm=0.0,
                precipitation_probability=5.0,
                rain_mm=0.0,
                temperature_c=22.0,
                relative_humidity=80.0,
                wind_speed_kmh=5.0,
                wind_gusts_kmh=8.0,
            )
        )
    forecast = make_dummy_forecast(points, past_7d_rain=10.0)
    res = evaluate_spray_decision("arecanut", "koleroga", forecast)

    # First dry window MUST start during daylight (06:00), lead_time = 6 hours -> WAIT
    assert res.decision == SprayDecision.WAIT
    assert res.next_dry_window is not None
    assert "T06:00" in res.next_dry_window.start_time
    assert res.next_dry_window.lead_time_hours == 6.0


def test_wait_when_near_term_rainy_but_dry_window_later():
    points = []
    # Morning (06:00-11:00): Heavy rain
    for i in range(6, 12):
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
    # Afternoon (12:00-18:00): Dry break (6 hours)
    for i in range(12, 18):
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
    forecast = make_dummy_forecast(points, agreement=0.9, past_7d_rain=25.0)
    res = evaluate_spray_decision("arecanut", "koleroga", forecast)

    assert res.decision == SprayDecision.WAIT
    assert res.next_dry_window is not None
    assert res.next_dry_window.lead_time_hours == 6.0
    assert "T12:00" in res.next_dry_window.start_time
    assert "Next daylight dry window expected" in res.decision_reason_en


def test_do_not_spray_persistent_rain():
    points = []
    # 48 hours of continuous rain
    for i in range(48):
        h = i % 24
        points.append(
            HourlyDataPoint(
                time=f"2026-06-15T{h:02d}:00",
                precipitation_mm=4.0,
                precipitation_probability=85.0,
                rain_mm=4.0,
                temperature_c=22.0,
                relative_humidity=95.0,
                wind_speed_kmh=18.0,
                wind_gusts_kmh=25.0,
            )
        )
    forecast = make_dummy_forecast(points, agreement=0.8, past_7d_rain=50.0)
    res = evaluate_spray_decision("arecanut", "koleroga", forecast)

    assert res.decision == SprayDecision.DO_NOT_SPRAY
    assert res.next_dry_window is None
    assert "No suitable daylight dry window" in res.decision_reason_en
    assert "ಯಾವುದೇ ಒಣ ವಾತಾವರಣ ಕಂಡುಬರುತ್ತಿಲ್ಲ" in res.decision_reason_kn


def test_wash_off_warning_when_heavy_rain_follows_recent_spray():
    # Spray 1 day ago, followed by 12mm past rain
    one_day_ago = (datetime.now(timezone.utc).date() - timedelta(days=1)).strftime("%Y-%m-%d")
    points = [HourlyDataPoint(time="2026-06-15T06:00", rain_mm=0.0, precipitation_probability=5.0)]
    forecast = make_dummy_forecast(points, past_24h_rain=12.0)

    res = evaluate_spray_decision("arecanut", "koleroga", forecast, last_spray_date=one_day_ago)
    assert res.wash_off_warning_en is not None
    assert "Wash-off warning" in res.wash_off_warning_en
    assert "12.0 mm of rainfall occurred" in res.wash_off_warning_en
    assert "ಔಷಧ ತೊಳೆದುಹೋಗಿರುವ ಎಚ್ಚರಿಕೆ" in res.wash_off_warning_kn


def test_seasonal_context_note_when_completely_dry():
    # In dry season (0mm past rain, 0mm forecast rain)
    points = [HourlyDataPoint(time="2026-11-08T06:00", rain_mm=0.0, precipitation_probability=0.0)]
    forecast = make_dummy_forecast(points, past_7d_rain=0.0)

    res = evaluate_spray_decision("arecanut", "koleroga", forecast, spray_intent=SprayIntent.FIRST)
    assert res.seasonal_context_en is not None
    assert "premonsoon showers" in res.seasonal_context_en
    assert "ಮುಂಗಾರು ಪೂರ್ವ ಮಳೆ" in res.seasonal_context_kn


def test_repeat_interval_wait():
    # Previous spray only 10 days ago (ICAR requires 40-45 days), without wash-off rain
    ten_days_ago = (datetime.now(timezone.utc).date() - timedelta(days=10)).strftime("%Y-%m-%d")
    points = [HourlyDataPoint(time="2026-06-15T06:00", rain_mm=0.0, precipitation_probability=5.0)]
    forecast = make_dummy_forecast(points, past_24h_rain=0.0)

    res = evaluate_spray_decision("arecanut", "koleroga", forecast, last_spray_date=ten_days_ago)
    assert res.decision == SprayDecision.WAIT
    assert "40-45 day repeat interval" in res.decision_reason_en
    assert "40-45 ದಿನಗಳ ಅಂತರವಿರಬೇಕು" in res.decision_reason_kn


def test_fail_closed_unsupported_crop():
    forecast = make_dummy_forecast([])
    with pytest.raises(ValueError, match="not supported"):
        evaluate_spray_decision("sugarcane", "red_rot", forecast)


def test_audit_logging_privacy_and_no_pii(tmp_path: Path):
    audit_file = tmp_path / "test_audit.csv"
    points = [
        HourlyDataPoint(
            time=f"2026-06-15T{i:02d}:00",
            precipitation_mm=0.0,
            precipitation_probability=10.0,
            rain_mm=0.0,
            temperature_c=25.0,
            relative_humidity=70.0,
            wind_speed_kmh=5.0,
            wind_gusts_kmh=8.0,
        )
        for i in range(6, 12)
    ]
    forecast = make_dummy_forecast(points, agreement=0.92)
    res = evaluate_spray_decision("arecanut", "koleroga", forecast)

    # Input lat/lon with high precision
    log_spray_decision(res, lat=14.1678912, lon=75.0432198, audit_path=audit_file)
    assert audit_file.exists()

    with open(audit_file, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        row = reader[0]
        # Coordinates must be rounded to exactly 2 decimals (~1.1 km grid)
        assert row["latitude"] == "14.17"
        assert row["longitude"] == "75.04"
        # Zero farmer PII keys
        for forbidden in ["farmer_id", "name", "phone", "aadhar", "survey_no"]:
            assert forbidden not in row
