"""
Krishi Sahayak - Spray Decision Engine and Audit Logging.

Determines spray timing (SPRAY_NOW, WAIT, DO_NOT_SPRAY) by evaluating
live 7-day weather forecast against hand-curated, ICAR-grounded spray rules
and configurable safety thresholds (labeled advisory defaults, not ICAR rules).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from api.weather import HourlyDataPoint, WeatherForecast
from genai.spray_rules import AgronomicSprayRule, get_spray_rule

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "spray_advisor_config.json"
AUDIT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "spray_audit.csv"


class SprayDecision(str, Enum):
    SPRAY_NOW = "SPRAY_NOW"
    WAIT = "WAIT"
    DO_NOT_SPRAY = "DO_NOT_SPRAY"


class SprayIntent(str, Enum):
    FIRST = "first"
    REPEAT = "repeat"
    UNSURE = "unsure"


@dataclass
class SprayWindow:
    start_time: str
    end_time: str
    duration_hours: float
    lead_time_hours: float
    avg_temperature_c: float
    max_wind_kmh: float
    max_precipitation_probability: float
    total_rain_mm: float


@dataclass
class SprayAdvisorResult:
    crop: str
    disease: str
    spray_intent: SprayIntent
    decision: SprayDecision
    decision_reason_en: str
    decision_reason_kn: str
    confidence_score: float
    confidence_factors: Dict[str, Any]
    next_dry_window: Optional[SprayWindow]
    upcoming_windows: List[SprayWindow]
    agronomic_rule: Optional[AgronomicSprayRule]
    thresholds_used: Dict[str, Any]
    disclaimer: str
    kvk_fallback_en: str
    kvk_fallback_kn: str
    seasonal_context_en: Optional[str] = None
    seasonal_context_kn: Optional[str] = None
    wash_off_warning_en: Optional[str] = None
    wash_off_warning_kn: Optional[str] = None


def load_engine_config() -> Dict[str, Any]:
    """Load engine safety thresholds and disclaimers from config file."""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to read engine config from {CONFIG_PATH}: {exc}. Using fallback defaults.")
    return {
        "rain_threshold_mm": 0.5,
        "precipitation_probability_threshold_pct": 35.0,
        "wind_speed_max_kmh": 15.0,
        "temperature_min_c": 18.0,
        "temperature_max_c": 32.0,
        "default_min_dry_hours": 4.0,
        "daylight_start_hour": 6,
        "daylight_end_hour": 18,
        "wash_off_rain_threshold_mm": 5.0,
        "immediate_window_max_lead_hours": 2,
        "disclaimer_en": "Advisory defaults, not ICAR rules.",
        "disclaimer_kn": "ಇವು ಸಲಹಾತ್ಮಕ ಡೀಫಾಲ್ಟ್ ಮಿತಿಗಳು, ಐಸಿಎಆರ್ ನಿಯಮಗಳಲ್ಲ.",
    }


def is_hour_sprayable(
    point: HourlyDataPoint,
    config: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Check if a single hourly datapoint meets advisory weather safety conditions.
    Restricts candidate spray hours strictly to daylight hours (default 06:00 to 18:00 local time).
    Preserves missing values without guessing.
    """
    reasons = []
    rain_limit = config.get("rain_threshold_mm", 0.5)
    prob_limit = config.get("precipitation_probability_threshold_pct", 35.0)
    wind_limit = config.get("wind_speed_max_kmh", 15.0)
    temp_min = config.get("temperature_min_c", 18.0)
    temp_max = config.get("temperature_max_c", 32.0)
    daylight_start = config.get("daylight_start_hour", 6)
    daylight_end = config.get("daylight_end_hour", 18)

    # Restrict to daylight hours (farmers spray during daylight, not at 2 AM)
    if "T" in point.time:
        try:
            hour_val = int(point.time.split("T")[1].split(":")[0])
            if not (daylight_start <= hour_val < daylight_end):
                reasons.append(f"non-daylight hour ({hour_val}:00 outside {daylight_start}:00-{daylight_end}:00)")
        except (IndexError, ValueError):
            pass

    if point.rain_mm is not None and point.rain_mm >= rain_limit:
        reasons.append(f"rain {point.rain_mm:.1f}mm >= {rain_limit}mm")

    if point.precipitation_probability is not None and point.precipitation_probability >= prob_limit:
        reasons.append(f"rain probability {point.precipitation_probability:.0f}% >= {prob_limit:.0f}%")

    if point.wind_speed_kmh is not None and point.wind_speed_kmh > wind_limit:
        reasons.append(f"wind {point.wind_speed_kmh:.1f}km/h > {wind_limit}km/h")

    if point.temperature_c is not None:
        if point.temperature_c < temp_min or point.temperature_c > temp_max:
            reasons.append(f"temp {point.temperature_c:.1f}C outside {temp_min}-{temp_max}C")

    return (len(reasons) == 0), reasons


def find_dry_windows(
    forecast: WeatherForecast,
    config: Dict[str, Any],
    min_duration_hours: float,
) -> List[SprayWindow]:
    """
    Scan future hourly forecast for contiguous blocks of sprayable daylight dry weather.
    """
    hourly = forecast.hourly
    start_idx = 0
    if forecast.current_time:
        for idx, pt in enumerate(hourly):
            if pt.time == forecast.current_time:
                start_idx = idx
                break

    future_points = hourly[start_idx:]
    windows: List[SprayWindow] = []

    current_block: List[HourlyDataPoint] = []
    block_start_lead = 0.0

    for i, pt in enumerate(future_points):
        sprayable, _ = is_hour_sprayable(pt, config)
        if sprayable:
            if not current_block:
                block_start_lead = float(i)
            current_block.append(pt)
        else:
            if len(current_block) >= min_duration_hours:
                windows.append(build_window_record(current_block, block_start_lead))
            current_block = []

    if len(current_block) >= min_duration_hours:
        windows.append(build_window_record(current_block, block_start_lead))

    return windows


def build_window_record(block: List[HourlyDataPoint], lead_time_hours: float) -> SprayWindow:
    """Compute aggregate metrics for a contiguous dry daylight block."""
    temps = [pt.temperature_c for pt in block if pt.temperature_c is not None]
    winds = [pt.wind_speed_kmh for pt in block if pt.wind_speed_kmh is not None]
    probs = [pt.precipitation_probability for pt in block if pt.precipitation_probability is not None]
    rains = [pt.rain_mm for pt in block if pt.rain_mm is not None]

    return SprayWindow(
        start_time=block[0].time,
        end_time=block[-1].time,
        duration_hours=float(len(block)),
        lead_time_hours=lead_time_hours,
        avg_temperature_c=round(sum(temps) / len(temps), 1) if temps else 25.0,
        max_wind_kmh=round(max(winds), 1) if winds else 0.0,
        max_precipitation_probability=round(max(probs), 1) if probs else 0.0,
        total_rain_mm=round(sum(rains), 2) if rains else 0.0,
    )


def calculate_decision_confidence(
    window: Optional[SprayWindow],
    model_agreement: Optional[float],
) -> Tuple[float, Dict[str, Any]]:
    """
    Weigh forecast lead time (hour 6 is more reliable than hour 60)
    and Open-Meteo multi-model agreement score.
    """
    if window is None:
        return 0.85, {"reason": "no_dry_window_in_horizon"}

    lead = window.lead_time_hours
    lead_factor = max(0.5, 1.0 - (lead / 168.0) * 0.5)

    factors: Dict[str, Any] = {
        "lead_time_hours": lead,
        "lead_time_reliability_factor": round(lead_factor, 2),
    }

    if model_agreement is not None:
        agreement_factor = 0.7 + (0.3 * model_agreement)
        factors["model_agreement_score"] = model_agreement
        factors["model_agreement_factor"] = round(agreement_factor, 2)
        score = lead_factor * agreement_factor
    else:
        factors["model_agreement_score"] = None
        factors["model_agreement_factor"] = 0.85
        score = lead_factor * 0.85

    return round(score, 2), factors


def evaluate_spray_decision(
    crop: str,
    disease: str,
    forecast: WeatherForecast,
    spray_intent: SprayIntent | str = SprayIntent.FIRST,
    last_spray_date: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> SprayAdvisorResult:
    """
    Evaluate weather and agronomy rules to issue SPRAY_NOW, WAIT, or DO_NOT_SPRAY.
    Includes wash-off warning, daylight window restrictions, and seasonal context notes.
    """
    cfg = config or load_engine_config()
    rule = get_spray_rule(crop, disease)
    if rule is None:
        raise ValueError(f"Crop '{crop}' and disease '{disease}' are not supported in grounded corpus.")

    if isinstance(spray_intent, str):
        try:
            spray_intent = SprayIntent(spray_intent.lower())
        except ValueError:
            spray_intent = SprayIntent.UNSURE

    min_dry_hours = cfg.get("default_min_dry_hours", 4.0)
    immediate_lead_max = cfg.get("immediate_window_max_lead_hours", 2)
    wash_off_threshold = cfg.get("wash_off_rain_threshold_mm", 5.0)
    disclaimer_en = cfg.get("disclaimer_en", "Advisory defaults, not ICAR rules.")
    disclaimer_kn = cfg.get("disclaimer_kn", "ಇವು ಸಲಹಾತ್ಮಕ ಡೀಫಾಲ್ಟ್ ಮಿತಿಗಳು, ಐಸಿಎಆರ್ ನಿಯಮಗಳಲ್ಲ.")

    wash_off_en: Optional[str] = None
    wash_off_kn: Optional[str] = None
    seasonal_en: Optional[str] = None
    seasonal_kn: Optional[str] = None

    # 1. Wash-off check: heavy rain soon after a recent spray (within past 1-3 days)
    if last_spray_date:
        try:
            last_dt = datetime.strptime(last_spray_date, "%Y-%m-%d").date()
            today = datetime.now(timezone.utc).date()
            days_since = (today - last_dt).days
            if days_since <= 3 and (forecast.past_24h_rain_mm >= wash_off_threshold or forecast.past_48h_rain_mm >= wash_off_threshold):
                recent_rain = max(forecast.past_24h_rain_mm, forecast.past_48h_rain_mm)
                wash_off_en = (
                    f"Wash-off warning: Spray was applied on {last_spray_date} ({days_since} days ago), "
                    f"and {recent_rain:.1f} mm of rainfall occurred shortly thereafter. "
                    "The fungicide deposit may have washed off before drying. Check bunches for protective blue residue "
                    "and consult your local KVK for a re-spray during the next dry window."
                )
                wash_off_kn = (
                    f"ಔಷಧ ತೊಳೆದುಹೋಗಿರುವ ಎಚ್ಚರಿಕೆ: ಹಿಂದಿನ ಸಿಂಪರಣೆಯನ್ನು {last_spray_date} ರಂದು ({days_since} ದಿನಗಳ ಹಿಂದೆ) ಮಾಡಲಾಗಿದ್ದು, "
                    f"ನಂತರ {recent_rain:.1f} ಮಿ.ಮೀ ಮಳೆಯಾಗಿದೆ. ಸಿಂಪಡಿಸಿದ ಔಷಧವು ತೊಳೆದುಹೋಗಿರುವ ಸಾಧ್ಯತೆಯಿದೆ. "
                    "ಗೊಂಚಲುಗಳನ್ನು ಪರಿಶೀಲಿಸಿ ಮುಂದಿನ ಒಣ ಸಮಯದಲ್ಲಿ ಮರು-ಸಿಂಪರಣೆ ಮಾಡಲು ಸ್ಥಳೀಯ ಕೆವಿಕೆಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
                )
        except ValueError:
            logger.warning(f"Invalid last_spray_date format '{last_spray_date}'; ignoring wash-off check.")

    # 2. Check repeat interval if spray_intent is repeat or last_spray_date provided
    if last_spray_date and not wash_off_en:
        try:
            last_dt = datetime.strptime(last_spray_date, "%Y-%m-%d").date()
            today = datetime.now(timezone.utc).date()
            days_since = (today - last_dt).days
            if rule.repeat_interval_days_min and days_since < rule.repeat_interval_days_min:
                wait_days = rule.repeat_interval_days_min - days_since
                return SprayAdvisorResult(
                    crop=crop,
                    disease=disease,
                    spray_intent=spray_intent,
                    decision=SprayDecision.WAIT,
                    decision_reason_en=(
                        f"Previous spray was applied {days_since} days ago. "
                        f"ICAR-CPCRI recommends a 40-45 day repeat interval ({rule.preparation['source']}). "
                        f"Wait approximately {wait_days} more days before the next application."
                    ),
                    decision_reason_kn=(
                        f"ಹಿಂದಿನ ಸಿಂಪರಣೆಯನ್ನು {days_since} ದಿನಗಳ ಹಿಂದೆ ಮಾಡಲಾಗಿದೆ. "
                        f"ಐಸಿಎಆರ್-ಸಿಪಿಸಿಆರ್‌ಐ ಶಿಫಾರಸಿನಂತೆ ಮುಂದಿನ ಸಿಂಪರಣೆಗೆ 40-45 ದಿನಗಳ ಅಂತರವಿರಬೇಕು. "
                        f"ಇನ್ನೂ ಸುಮಾರು {wait_days} ದಿನಗಳ ನಂತರ ಸಿಂಪರಣೆ ಕೈಗೊಳ್ಳಿ."
                    ),
                    confidence_score=0.95,
                    confidence_factors={"interval_rule": "repeat_interval_not_elapsed", "days_since": days_since},
                    next_dry_window=None,
                    upcoming_windows=[],
                    agronomic_rule=rule,
                    thresholds_used=cfg,
                    disclaimer=disclaimer_en,
                    kvk_fallback_en=rule.kvk_fallback_message_en,
                    kvk_fallback_kn=rule.kvk_fallback_message_kn,
                    wash_off_warning_en=wash_off_en,
                    wash_off_warning_kn=wash_off_kn,
                )
        except ValueError:
            logger.warning(f"Invalid last_spray_date format '{last_spray_date}'; ignoring interval check.")

    # 3. Seasonal context note: Sources tie Koleroga sprays strictly to pre-monsoon showers or monsoon rain breaks
    current_idx = 0
    if forecast.current_time:
        for idx, pt in enumerate(forecast.hourly):
            if pt.time == forecast.current_time:
                current_idx = idx
                break
    future_rain = sum((pt.rain_mm or 0.0) for pt in forecast.hourly[current_idx:] if pt.rain_mm is not None)

    if forecast.past_7d_rain_mm == 0.0 and future_rain == 0.0:
        if spray_intent == SprayIntent.FIRST:
            seasonal_en = (
                "Seasonal context: ICAR-CPCRI specifies the first prophylactic spray 'after the premonsoon showers'. "
                "Current weather indicates completely dry conditions with zero recent rainfall. "
                "If pre-monsoon showers have not commenced in your locality, consult your local KVK or Raitha Samparka Kendra before spraying."
            )
            seasonal_kn = (
                "ಋತುಮಾನದ ಮಾಹಿತಿ: ಐಸಿಎಆರ್-ಸಿಪಿಸಿಆರ್‌ಐ ಪ್ರಕಾರ ಮೊದಲ ಮುನ್ನೆಚ್ಚರಿಕೆ ಸಿಂಪರಣೆಯನ್ನು 'ಮುಂಗಾರು ಪೂರ್ವ ಮಳೆಯ ನಂತರ' ಮಾಡಬೇಕು. "
                "ಪ್ರಸ್ತುತ ಸಂಪೂರ್ಣ ಒಣ ಹವೆ ಇದ್ದು ಮಳೆಯಾಗಿಲ್ಲ. ನಿಮ್ಮ ಪ್ರದೇಶದಲ್ಲಿ ಮುಂಗಾರು ಪೂರ್ವ ಮಳೆ ಆರಂಭವಾಗದಿದ್ದರೆ, "
                "ಸಿಂಪಡಿಸುವ ಮುನ್ನ ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಅಥವಾ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರವನ್ನು ಸಂಪರ್ಕಿಸಿ."
            )
        elif spray_intent in {SprayIntent.REPEAT, SprayIntent.UNSURE}:
            seasonal_en = (
                "Seasonal context: Koleroga sprays are recommended during monsoon breaks. "
                "Conditions are currently completely dry. Verify local disease symptoms with your KVK."
            )
            seasonal_kn = (
                "ಋತುಮಾನದ ಮಾಹಿತಿ: ಕೊಳೆರೋಗದ ಸಿಂಪರಣೆಯನ್ನು ಮುಂಗಾರು ಮಳೆಯ ಬಿಡುವಿನಲ್ಲಿ ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ. "
                "ಪ್ರಸ್ತುತ ಒಣ ಹವೆ ಇದ್ದು, ರೋಗದ ಲಕ್ಷಣಗಳನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಲು ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಂಪರ್ಕಿಸಿ."
            )

    windows = find_dry_windows(forecast, cfg, min_duration_hours=min_dry_hours)

    if not windows:
        conf_score, conf_factors = calculate_decision_confidence(None, forecast.model_agreement_score)
        return SprayAdvisorResult(
            crop=crop,
            disease=disease,
            spray_intent=spray_intent,
            decision=SprayDecision.DO_NOT_SPRAY,
            decision_reason_en=(
                f"No suitable daylight dry window of at least {min_dry_hours:.0f} hours is forecasted within the next 7 days. "
                "Incessant rain or high rain probability will wash off the spray. "
                "Wait for a clear break in rainfall before applying 1% Bordeaux mixture or Mandipropamid."
            ),
            decision_reason_kn=(
                f"ಮುಂದಿನ 7 ದಿನಗಳಲ್ಲಿ ಹಗಲಿನ ಸಮಯದಲ್ಲಿ ಕನಿಷ್ಠ {min_dry_hours:.0f} ಗಂಟೆಗಳ ಕಾಲ ಯಾವುದೇ ಒಣ ವಾತಾವರಣ ಕಂಡುಬರುತ್ತಿಲ್ಲ. "
                "ನಿರಂತರ ಮಳೆಯಿಂದಾಗಿ ಸಿಂಪಡಿಸಿದ ಔಷಧವು ತೊಳೆದುಹೋಗುತ್ತದೆ. "
                "ಮಳೆ ಬಿಡುವು ನೀಡುವವರೆಗೆ ಕಾದು ನಂತರ 1% ಬೋರ್ಡೋ ದ್ರಾವಣ ಅಥವಾ ಮ್ಯಾಂಡಿಪ್ರೊಪಮಿಡ್ ಸಿಂಪಡಿಸಿ."
            ),
            confidence_score=conf_score,
            confidence_factors=conf_factors,
            next_dry_window=None,
            upcoming_windows=[],
            agronomic_rule=rule,
            thresholds_used=cfg,
            disclaimer=disclaimer_en,
            kvk_fallback_en=rule.kvk_fallback_message_en,
            kvk_fallback_kn=rule.kvk_fallback_message_kn,
            seasonal_context_en=seasonal_en,
            seasonal_context_kn=seasonal_kn,
            wash_off_warning_en=wash_off_en,
            wash_off_warning_kn=wash_off_kn,
        )

    first_win = windows[0]
    conf_score, conf_factors = calculate_decision_confidence(first_win, forecast.model_agreement_score)

    if first_win.lead_time_hours <= immediate_lead_max:
        decision = SprayDecision.SPRAY_NOW
        reason_en = (
            f"Favorable daylight dry window available NOW from {first_win.start_time} to {first_win.end_time} "
            f"({first_win.duration_hours:.0f} hours dry, rain prob <= {first_win.max_precipitation_probability:.0f}%, "
            f"wind <= {first_win.max_wind_kmh:.1f} km/h). Apply 1% neutral Bordeaux mixture or Mandipropamid 23.4% SC (1 ml/L) thoroughly."
        )
        reason_kn = (
            f"ಈಗಲೇ ಸಿಂಪಡಿಸಲು ಸೂಕ್ತ ಹಗಲಿನ ಒಣ ವಾತಾವರಣವಿದೆ: {first_win.start_time} ರಿಂದ {first_win.end_time} ರವರೆಗೆ "
            f"({first_win.duration_hours:.0f} ಗಂಟೆಗಳ ಒಣ ಸಮಯ, ಮಳೆ ಸಂಭವನೀಯತೆ <= {first_win.max_precipitation_probability:.0f}%). "
            "1% ತಟಸ್ಥ ಬೋರ್ಡೋ ದ್ರಾವಣ ಅಥವಾ ಮ್ಯಾಂಡಿಪ್ರೊಪಮಿಡ್ 23.4% SC (1 ಮಿ.ಲೀ/ಲೀ) ಸಿಂಪಡಿಸಿ."
        )
    else:
        decision = SprayDecision.WAIT
        reason_en = (
            f"Conditions are currently unfavorable. Next daylight dry window expected on {first_win.start_time} "
            f"to {first_win.end_time} ({first_win.duration_hours:.0f} hours dry, {first_win.lead_time_hours:.0f} hours ahead). "
            "Wait for this break in rainfall before spraying."
        )
        reason_kn = (
            f"ಪ್ರಸ್ತುತ ಹವಾಮಾನವು ಸಿಂಪರಣೆಗೆ ಸೂಕ್ತವಾಗಿಲ್ಲ. ಮುಂದಿನ ಹಗಲಿನ ಒಣ ಸಮಯವು {first_win.start_time} ರಿಂದ {first_win.end_time} "
            f"ರವರೆಗೆ ({first_win.duration_hours:.0f} ಗಂಟೆಗಳ ಒಣ ವಾತಾವರಣ) ನಿರೀಕ್ಷಿಸಲಾಗಿದೆ. "
            "ಈ ಮಳೆ ಬಿಡುವಿನವರೆಗೆ ಕಾಯಿರಿ."
        )

    return SprayAdvisorResult(
        crop=crop,
        disease=disease,
        spray_intent=spray_intent,
        decision=decision,
        decision_reason_en=reason_en,
        decision_reason_kn=reason_kn,
        confidence_score=conf_score,
        confidence_factors=conf_factors,
        next_dry_window=first_win,
        upcoming_windows=windows,
        agronomic_rule=rule,
        thresholds_used=cfg,
        disclaimer=disclaimer_en,
        kvk_fallback_en=rule.kvk_fallback_message_en,
        kvk_fallback_kn=rule.kvk_fallback_message_kn,
        seasonal_context_en=seasonal_en,
        seasonal_context_kn=seasonal_kn,
        wash_off_warning_en=wash_off_en,
        wash_off_warning_kn=wash_off_kn,
    )


def log_spray_decision(
    result: SprayAdvisorResult,
    lat: float,
    lon: float,
    audit_path: Optional[Path] = None,
) -> None:
    """
    Append an evaluation record to the CSV audit log for transparency and governance.
    Strict privacy rule: coordinates are rounded to 2 decimal places (~1.1 km grid)
    and NO farmer identity or PII is recorded.
    """
    path = audit_path or AUDIT_LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "crop": result.crop,
        "disease": result.disease,
        "spray_intent": result.spray_intent.value,
        "latitude": round(lat, 2),
        "longitude": round(lon, 2),
        "decision": result.decision.value,
        "confidence_score": result.confidence_score,
        "window_start": result.next_dry_window.start_time if result.next_dry_window else "",
        "window_end": result.next_dry_window.end_time if result.next_dry_window else "",
        "lead_time_hours": result.next_dry_window.lead_time_hours if result.next_dry_window else "",
        "duration_hours": result.next_dry_window.duration_hours if result.next_dry_window else "",
        "model_agreement": result.confidence_factors.get("model_agreement_score", ""),
        "wash_off_detected": bool(result.wash_off_warning_en),
        "reason_en": result.decision_reason_en,
    }

    try:
        with open(path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:
        logger.error(f"Failed to write spray audit log to {path}: {exc}")
