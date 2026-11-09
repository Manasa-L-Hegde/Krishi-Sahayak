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
    disclaimer_en: str
    disclaimer_kn: str
    kvk_fallback_en: str
    kvk_fallback_kn: str
    safety_footer_en: str
    safety_footer_kn: str
    seasonal_context_en: Optional[str] = None
    seasonal_context_kn: Optional[str] = None
    wash_off_warning_en: Optional[str] = None
    wash_off_warning_kn: Optional[str] = None

    @property
    def disclaimer(self) -> str:
        return self.disclaimer_en

    @property
    def safety_footer(self) -> str:
        return self.safety_footer_en


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
        "disclaimer_en": "Advisory defaults, not ICAR rules. These parameters are heuristic weather safety thresholds for spray application.",
        "disclaimer_kn": "ಇವು ಸಲಹಾತ್ಮಕ ಡೀಫಾಲ್ಟ್ ಮಿತಿಗಳು, ಐಸಿಎಆರ್ ನಿಯಮಗಳಲ್ಲ. ಇವು ಸಿಂಪರಣೆ ಸುರಕ್ಷತೆಯ ಅಂದಾಜು ಹವಾಮಾನ ಮಿತಿಗಳಾಗಿವೆ.",
        "safety_footer_en": "Safety advisory: Always wear personal protective equipment (PPE: gloves, face mask, eye protection), follow manufacturer instructions on the chemical label, strictly observe the pre-harvest interval (PHI) before harvest, and consult your local Krishi Vigyan Kendra (KVK) or Raitha Samparka Kendra (RSK) for localized guidance.",
        "safety_footer_kn": "ಸುರಕ್ಷತಾ ಸಲಹೆ: ಸಿಂಪಡಿಸುವಾಗ ಯಾವಾಗಲೂ ವೈಯಕ್ತಿಕ ಸುರಕ್ಷತಾ ಸಾಧನಗಳನ್ನು (ಪಿಪಿಇ: ಕೈಗವಸು, ಮುಖಗವಸು, ಕಣ್ಣಿನ ರಕ್ಷಣೆ) ಧರಿಸಿ, ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಮೇಲಿನ ಸೂಚನೆಗಳನ್ನು ಕಟ್ಟುನಿಟ್ಟಾಗಿ ಪಾಲಿಸಿ, ಕಟಾವಿಗೆ ಮುನ್ನ ಕಾಯುವ ಅವಧಿಯನ್ನು (PHI) ಗಮನಿಸಿ, ಮತ್ತು ಸ್ಥಳೀಯ ಮಾರ್ಗದರ್ಶನಕ್ಕಾಗಿ ಹತ್ತಿರದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ (ಕೆವಿಕೆ) ಅಥವಾ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರವನ್ನು (ಆರ್‌ಎಸ್‌ಕೆ) ಸಂಪರ್ಕಿಸಿ.",
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

    # Calculate end of the window: end of the last safe hour (+1 hour from block[-1].time)
    last_time_str = block[-1].time
    end_time_str = last_time_str
    if "T" in last_time_str:
        try:
            from datetime import timedelta
            last_dt = datetime.strptime(last_time_str, "%Y-%m-%dT%H:00")
            end_dt = last_dt + timedelta(hours=1)
            end_time_str = end_dt.strftime("%Y-%m-%dT%H:00")
        except ValueError:
            pass

    return SprayWindow(
        start_time=block[0].time,
        end_time=end_time_str,
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
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Weigh forecast lead time (hour 6 is more reliable than hour 60),
    Open-Meteo multi-model agreement score, and threshold safety margin
    (lowering confidence when a critical parameter such as wind is near its boundary).
    """
    if window is None:
        return 0.85, {"reason": "no_dry_window_in_horizon"}

    cfg = config or load_engine_config()
    wind_limit = cfg.get("wind_speed_max_kmh", 15.0)
    prob_limit = cfg.get("precipitation_probability_threshold_pct", 35.0)

    # 1. Lead time reliability factor
    lead = window.lead_time_hours
    lead_factor = max(0.5, 1.0 - (lead / 168.0) * 0.5)

    # 2. Model agreement factor
    if model_agreement is not None:
        agreement_factor = 0.7 + (0.3 * model_agreement)
    else:
        agreement_factor = 0.85

    # 3. Threshold safety margin factor (heuristic advisory calibration):
    # If the deciding variable (e.g. wind at 13.6 km/h vs 15.0 km/h limit) is within 10-20%
    # of the boundary, scale down confidence since slight microclimate/forecast variance could breach limits.
    # Note: confidence_margin_band_pct (default 20%) and confidence_margin_floor (default 0.75) are
    # heuristic advisory settings, not ICAR biological limits.
    band_pct = cfg.get("confidence_margin_band_pct", 20.0) / 100.0
    floor_factor = cfg.get("confidence_margin_floor", 0.75)

    wind_margin = max(0.0, (wind_limit - window.max_wind_kmh) / wind_limit) if wind_limit > 0 else 1.0
    prob_margin = max(0.0, (prob_limit - window.max_precipitation_probability) / prob_limit) if prob_limit > 0 else 1.0
    limiting_margin = min(wind_margin, prob_margin)

    # Linear interpolation between floor_factor and 1.0 across the margin band:
    # formula: margin_factor = floor_factor + (1.0 - floor_factor) * (limiting_margin / band_pct)
    if limiting_margin < band_pct and band_pct > 0:
        margin_factor = max(floor_factor, floor_factor + (limiting_margin / band_pct) * (1.0 - floor_factor))
    else:
        margin_factor = 1.0

    score = lead_factor * agreement_factor * margin_factor

    factors: Dict[str, Any] = {
        "lead_time_hours": lead,
        "lead_time_reliability_factor": round(lead_factor, 2),
        "model_agreement_score": model_agreement,
        "model_agreement_factor": round(agreement_factor, 2),
        "threshold_margin_pct": round(limiting_margin * 100, 1),
        "threshold_margin_factor": round(margin_factor, 2),
    }

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
    disclaimer_en = cfg.get("disclaimer_en", "Advisory defaults, not ICAR rules. These parameters are heuristic weather safety thresholds for spray application.")
    disclaimer_kn = cfg.get("disclaimer_kn", "ಇವು ಸಲಹಾತ್ಮಕ ಡೀಫಾಲ್ಟ್ ಮಿತಿಗಳು, ಐಸಿಎಆರ್ ನಿಯಮಗಳಲ್ಲ. ಇವು ಸಿಂಪರಣೆ ಸುರಕ್ಷತೆಯ ಅಂದಾಜು ಹವಾಮಾನ ಮಿತಿಗಳಾಗಿವೆ.")
    safety_footer_en = cfg.get(
        "safety_footer_en",
        "Safety advisory: Always wear personal protective equipment (PPE: gloves, face mask, eye protection), follow manufacturer instructions on the chemical label, strictly observe the pre-harvest interval (PHI) before harvest, and consult your local Krishi Vigyan Kendra (KVK) or Raitha Samparka Kendra (RSK) for localized guidance."
    )
    safety_footer_kn = cfg.get(
        "safety_footer_kn",
        "ಸುರಕ್ಷತಾ ಸಲಹೆ: ಸಿಂಪಡಿಸುವಾಗ ಯಾವಾಗಲೂ ವೈಯಕ್ತಿಕ ಸುರಕ್ಷತಾ ಸಾಧನಗಳನ್ನು (ಪಿಪಿಇ: ಕೈಗವಸು, ಮುಖಗವಸು, ಕಣ್ಣಿನ ರಕ್ಷಣೆ) ಧರಿಸಿ, ಕೀಟನಾಶಕ ಲೇಬಲ್ ಮೇಲಿನ ಸೂಚನೆಗಳನ್ನು ಕಟ್ಟುನಿಟ್ಟಾಗಿ ಪಾಲಿಸಿ, ಕಟಾವಿಗೆ ಮುನ್ನ ಕಾಯುವ ಅವಧಿಯನ್ನು (PHI) ಗಮನಿಸಿ, ಮತ್ತು ಸ್ಥಳೀಯ ಮಾರ್ಗದರ್ಶನಕ್ಕಾಗಿ ಹತ್ತಿರದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ (ಕೆವಿಕೆ) ಅಥವಾ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರವನ್ನು (ಆರ್‌ಎಸ್‌ಕೆ) ಸಂಪರ್ಕಿಸಿ."
    )

    wash_off_en: Optional[str] = None
    wash_off_kn: Optional[str] = None
    seasonal_en: Optional[str] = None
    seasonal_kn: Optional[str] = None

    # 1. Wash-off check: rain falling strictly AFTER spray application up to current_time
    if last_spray_date:
        try:
            # App assumption / convention: If date-only (YYYY-MM-DD), anchor application to 10:00 AM local time
            # (default morning window assumption; not an ICAR-prescribed hour). If ISO with time, parse directly.
            spray_time_str = f"{last_spray_date}T10:00" if "T" not in last_spray_date else last_spray_date
            last_date_part = last_spray_date.split("T")[0]
            last_dt = datetime.strptime(last_date_part, "%Y-%m-%d").date()
            ref_date = (
                datetime.strptime(forecast.current_time.split("T")[0], "%Y-%m-%d").date()
                if forecast.current_time and "T" in forecast.current_time
                else datetime.now(timezone.utc).date()
            )
            days_since = (ref_date - last_dt).days

            # Calculate rainfall occurring strictly after spray application up to current_time
            rain_after_spray = 0.0
            for pt in forecast.hourly:
                if pt.time > spray_time_str:
                    if forecast.current_time and pt.time > forecast.current_time:
                        break
                    if pt.rain_mm is not None:
                        rain_after_spray += pt.rain_mm

            if 0 <= days_since <= 3 and rain_after_spray >= wash_off_threshold:
                wash_off_en = (
                    f"[App advisory, not an ICAR rule] Possible wash-off detected: Spray was applied on {last_spray_date} "
                    f"({days_since} day{'s' if days_since != 1 else ''} ago), and {rain_after_spray:.1f} mm of rainfall occurred after application. "
                    "Heavy rainfall shortly after application can wash off fungicide deposits before proper drying. "
                    "Inspect bunches for visible fungicide coverage and consult your local KVK or Raitha Samparka Kendra before re-spraying."
                )
                wash_off_kn = (
                    f"[ಆ್ಯಪ್ ಸಲಹೆ, ಐಸಿಎಆರ್ ನಿಯಮವಲ್ಲ] ಔಷಧ ತೊಳೆದುಹೋಗಿರುವ ಸಾಧ್ಯತೆ: {last_spray_date} ರಂದು ({days_since} ದಿನಗಳ ಹಿಂದೆ) "
                    f"ಸಿಂಪರಣೆ ಮಾಡಿದ ನಂತರ {rain_after_spray:.1f} ಮಿ.ಮೀ ಮಳೆಯಾಗಿದೆ. ಸಿಂಪಡಿಸಿದ ನಂತರ ಸುರಿದ ಮಳೆಯಿಂದಾಗಿ ಔಷಧದ ಲೇಪನವು ತೊಳೆದುಹೋಗಿರುವ ಸಾಧ್ಯತೆಯಿದೆ. "
                    "ಗೊಂಚಲುಗಳಲ್ಲಿ ಔಷಧದ ಲೇಪನ ಉಳಿದಿದೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸಿ, ಮತ್ತು ಮರು-ಸಿಂಪರಣೆ ಮಾಡುವ ಮುನ್ನ ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಅಥವಾ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರವನ್ನು ಸಂಪರ್ಕಿಸಿ."
                )
        except ValueError:
            logger.warning(f"Invalid last_spray_date format '{last_spray_date}'; ignoring wash-off check.")

    # 2. Check repeat interval if spray_intent is repeat or last_spray_date provided
    if last_spray_date and not wash_off_en:
        try:
            last_date_part = last_spray_date.split("T")[0]
            last_dt = datetime.strptime(last_date_part, "%Y-%m-%d").date()
            ref_date = (
                datetime.strptime(forecast.current_time.split("T")[0], "%Y-%m-%d").date()
                if forecast.current_time and "T" in forecast.current_time
                else datetime.now(timezone.utc).date()
            )
            days_since = (ref_date - last_dt).days
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
                    disclaimer_en=disclaimer_en,
                    disclaimer_kn=disclaimer_kn,
                    kvk_fallback_en=rule.kvk_fallback_message_en,
                    kvk_fallback_kn=rule.kvk_fallback_message_kn,
                    safety_footer_en=safety_footer_en,
                    safety_footer_kn=safety_footer_kn,
                    wash_off_warning_en=wash_off_en,
                    wash_off_warning_kn=wash_off_kn,
                )
        except ValueError:
            logger.warning(f"Invalid last_spray_date format '{last_spray_date}'; ignoring interval check.")

    # 3. Seasonal context note: Sources tie Koleroga sprays strictly to pre-monsoon showers or monsoon rain breaks
    if spray_intent in {SprayIntent.FIRST, SprayIntent.UNSURE}:
        seasonal_en = (
            "Seasonal context: ICAR-CPCRI recommends applying the first prophylactic spray after the premonsoon showers (Ext. Folder No. 17, 1989) "
            "or as a pre-monsoon application (Ext. Pub. No. 69, 1999). "
            "If pre-monsoon showers have not commenced in your locality, consult your local KVK or Raitha Samparka Kendra before spraying."
        )
        seasonal_kn = (
            "ಋತುಮಾನದ ಮಾಹಿತಿ: ಐಸಿಎಆರ್-ಸಿಪಿಸಿಆರ್‌ಐ ಪ್ರಕಾರ ಮೊದಲ ಮುನ್ನೆಚ್ಚರಿಕೆ ಸಿಂಪರಣೆಯನ್ನು ಮುಂಗಾರು ಪೂರ್ವ ಮಳೆಯ ನಂತರ (ವಿಸ್ತರಣಾ ಪತ್ರಿಕೆ 17, 1989) "
            "ಅಥವಾ ಮುಂಗಾರು ಪೂರ್ವದಲ್ಲಿ (ವಿಸ್ತರಣಾ ಪ್ರಕಟಣೆ 69, 1999) ಕೈಗೊಳ್ಳಬೇಕು. "
            "ನಿಮ್ಮ ಭಾಗದಲ್ಲಿ ಮುಂಗಾರು ಪೂರ್ವ ಮಳೆಯಾಗಿರದಿದ್ದರೆ ಸಿಂಪಡಿಸುವ ಮುನ್ನ ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಅಥವಾ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರವನ್ನು ಸಂಪರ್ಕಿಸಿ."
        )
    elif spray_intent == SprayIntent.REPEAT:
        seasonal_en = (
            "Seasonal context: Koleroga repeat sprays (40-45 days after the first spray) are recommended during breaks in monsoon rainfall (Ext. Pub. No. 69, 1999; The Hindu / CPCRI, Aug 2024)."
        )
        seasonal_kn = (
            "ಋತುಮಾನದ ಮಾಹಿತಿ: ಕೊಳೆರೋಗದ ಎರಡನೇ ಸಿಂಪರಣೆಯನ್ನು (ಮೊದಲ ಸಿಂಪರಣೆಯ 40-45 ದಿನಗಳ ನಂತರ) ಮುಂಗಾರು ಮಳೆಯ ಬಿಡುವಿನ ಸಮಯದಲ್ಲಿ ಕೈಗೊಳ್ಳಲು ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ (ವಿಸ್ತರಣಾ ಪ್ರಕಟಣೆ 69, 1999; ದಿ ಹಿಂದೂ / ಸಿಪಿಸಿಆರ್‌ಐ, ಆಗಸ್ಟ್ 2024)."
        )

    windows = find_dry_windows(forecast, cfg, min_duration_hours=min_dry_hours)

    if not windows:
        conf_score, conf_factors = calculate_decision_confidence(None, forecast.model_agreement_score, cfg)
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
            disclaimer_en=disclaimer_en,
            disclaimer_kn=disclaimer_kn,
            kvk_fallback_en=rule.kvk_fallback_message_en,
            kvk_fallback_kn=rule.kvk_fallback_message_kn,
            safety_footer_en=safety_footer_en,
            safety_footer_kn=safety_footer_kn,
            seasonal_context_en=seasonal_en,
            seasonal_context_kn=seasonal_kn,
            wash_off_warning_en=wash_off_en,
            wash_off_warning_kn=wash_off_kn,
        )

    first_win = windows[0]
    conf_score, conf_factors = calculate_decision_confidence(first_win, forecast.model_agreement_score, cfg)

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
        # Diagnose specific causes for current unsuitability
        unfavorable_causes_en = []
        unfavorable_causes_kn = []

        curr_pt = None
        if forecast.current_time:
            for pt in forecast.hourly:
                if pt.time == forecast.current_time:
                    curr_pt = pt
                    break

        if curr_pt:
            _, fail_reasons = is_hour_sprayable(curr_pt, cfg)
            for r in fail_reasons:
                if "non-daylight" in r:
                    unfavorable_causes_en.append("Currently nighttime (spraying is restricted to daylight hours 06:00-18:00)")
                    unfavorable_causes_kn.append("ಪ್ರಸ್ತುತ ರಾತ್ರಿ ಸಮಯ (ಸಿಂಪರಣೆಯನ್ನು ಹಗಲಿನ ಸಮಯ 06:00-18:00 ಕ್ಕೆ ಮಾತ್ರ ಸೀಮಿತಗೊಳಿಸಲಾಗಿದೆ)")
                elif "rain " in r and "probability" not in r:
                    unfavorable_causes_en.append(f"Currently raining ({curr_pt.rain_mm:.1f} mm)")
                    unfavorable_causes_kn.append(f"ಪ್ರಸ್ತುತ ಮಳೆಯಾಗುತ್ತಿದೆ ({curr_pt.rain_mm:.1f} ಮಿ.ಮೀ)")
                elif "rain probability" in r:
                    unfavorable_causes_en.append(f"Rain probability is currently elevated ({curr_pt.precipitation_probability:.0f}%)")
                    unfavorable_causes_kn.append(f"ಮಳೆ ಸಂಭವನೀಯತೆ ಹೆಚ್ಚಾಗಿದೆ ({curr_pt.precipitation_probability:.0f}%)")
                elif "wind" in r:
                    unfavorable_causes_en.append(f"Wind speed is currently high ({curr_pt.wind_speed_kmh:.1f} km/h)")
                    unfavorable_causes_kn.append(f"ಗಾಳಿಯ ವೇಗ ಹೆಚ್ಚಾಗಿದೆ ({curr_pt.wind_speed_kmh:.1f} ಕಿ.ಮೀ/ಗಂ)")
                elif "temp" in r:
                    unfavorable_causes_en.append(f"Temperature is outside safe limits ({curr_pt.temperature_c:.1f}°C)")
                    unfavorable_causes_kn.append(f"ತಾಪಮಾನವು ಸುರಕ್ಷಿತ ಮಿತಿಯ ಹೊರಗಿದೆ ({curr_pt.temperature_c:.1f}°C)")

        cause_str_en = "; ".join(unfavorable_causes_en) if unfavorable_causes_en else "Conditions are currently unfavorable"
        cause_str_kn = "; ".join(unfavorable_causes_kn) if unfavorable_causes_kn else "ಪ್ರಸ್ತುತ ಹವಾಮಾನವು ಸಿಂಪರಣೆಗೆ ಸೂಕ್ತವಾಗಿಲ್ಲ"

        reason_en = (
            f"{cause_str_en}. Next daylight dry window expected on {first_win.start_time} "
            f"to {first_win.end_time} ({first_win.duration_hours:.0f} hours dry, {first_win.lead_time_hours:.0f} hours ahead). "
            "Wait for safe spray conditions before applying."
        )
        reason_kn = (
            f"{cause_str_kn}. ಮುಂದಿನ ಹಗಲಿನ ಒಣ ಸಮಯವು {first_win.start_time} ರಿಂದ {first_win.end_time} "
            f"ರವರೆಗೆ ({first_win.duration_hours:.0f} ಗಂಟೆಗಳ ಒಣ ವಾತಾವರಣ) ನಿರೀಕ್ಷಿಸಲಾಗಿದೆ. "
            "ಸಿಂಪಡಿಸಲು ಸೂಕ್ತ ವಾತಾವರಣ ಬರುವವರೆಗೆ ಕಾಯಿರಿ."
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
        disclaimer_en=disclaimer_en,
        disclaimer_kn=disclaimer_kn,
        kvk_fallback_en=rule.kvk_fallback_message_en,
        kvk_fallback_kn=rule.kvk_fallback_message_kn,
        safety_footer_en=safety_footer_en,
        safety_footer_kn=safety_footer_kn,
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
