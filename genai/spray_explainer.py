"""
Krishi Sahayak - Spray Advice Explainer.

Generates bilingual (English and Kannada) explanations grounded strictly
in ICAR-CPCRI publications and deterministic decision engine outputs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from api.spray_engine import SprayAdvisorResult, SprayDecision
from genai.spray_rules import AgronomicSprayRule


def build_spray_explanation(
    result: SprayAdvisorResult,
    language: str = "en",
) -> Dict[str, Any]:
    """
    Format a complete, user-facing advisory payload with step-by-step
    agronomic preparation and safety guidance in English or Kannada.
    """
    lang = language.lower()
    is_kn = lang == "kn"
    rule = result.agronomic_rule

    # Decision banner
    if is_kn:
        decision_label = {
            SprayDecision.SPRAY_NOW: "ಈಗಲೇ ಸಿಂಪಡಿಸಿ (SPRAY NOW)",
            SprayDecision.WAIT: "ಕಾದುನೋಡಿ (WAIT)",
            SprayDecision.DO_NOT_SPRAY: "ಸಿಂಪಡಿಸಬೇಡಿ (DO NOT SPRAY)",
        }[result.decision]
        reason_text = result.decision_reason_kn
    else:
        decision_label = {
            SprayDecision.SPRAY_NOW: "SPRAY NOW",
            SprayDecision.WAIT: "WAIT",
            SprayDecision.DO_NOT_SPRAY: "DO NOT SPRAY",
        }[result.decision]
        reason_text = result.decision_reason_en

    # Agronomic guidance
    agronomy_details: Dict[str, Any] = {}
    if rule:
        if is_kn:
            agronomy_details = {
                "fungicide_name": "1% ತಟಸ್ಥ ಬೋರ್ಡೋ ದ್ರಾವಣ",
                "preparation_steps": rule.preparation["instruction_kn"],
                "quality_test": rule.neutrality_test["instruction_kn"],
                "adjuvant": rule.adjuvant["instruction_kn"],
                "repeat_interval": f"{rule.repeat_interval_days_min}-{rule.repeat_interval_days_max} ದಿನಗಳ ನಂತರ",
                "bunch_covering": rule.bunch_covering["instruction_kn"],
                "alternative_fungicide": (
                    rule.alternative_fungicide["instruction_kn"]
                    if rule.alternative_fungicide
                    else None
                ),
            }
        else:
            agronomy_details = {
                "fungicide_name": rule.fungicide_name,
                "preparation_steps": rule.preparation["instruction_en"],
                "quality_test": rule.neutrality_test["instruction_en"],
                "adjuvant": rule.adjuvant["instruction_en"],
                "repeat_interval": f"{rule.repeat_interval_days_min}-{rule.repeat_interval_days_max} days",
                "bunch_covering": rule.bunch_covering["instruction_en"],
                "alternative_fungicide": (
                    rule.alternative_fungicide["instruction_en"]
                    if rule.alternative_fungicide
                    else None
                ),
            }

    # Window details
    window_data = None
    if result.next_dry_window:
        win = result.next_dry_window
        window_data = {
            "start_time": win.start_time,
            "end_time": win.end_time,
            "duration_hours": win.duration_hours,
            "lead_time_hours": win.lead_time_hours,
            "avg_temperature_c": win.avg_temperature_c,
            "max_wind_kmh": win.max_wind_kmh,
            "max_rain_probability_pct": win.max_precipitation_probability,
        }

    # Wash-off alert
    wash_off_alert = result.wash_off_warning_kn if is_kn else result.wash_off_warning_en
    # Seasonal context note
    seasonal_note = result.seasonal_context_kn if is_kn else result.seasonal_context_en
    # KVK fallback
    kvk_fallback = result.kvk_fallback_kn if is_kn else result.kvk_fallback_en

    # Citations
    citations = []
    if rule:
        for p in rule.provenance.values():
            if p.verification_status == "VERIFIED":
                citations.append({
                    "title": p.title,
                    "publication_year": p.publication_year,
                    "page": p.page,
                    "url": p.url,
                    "source_type": p.source_type,
                })

    return {
        "crop": result.crop,
        "disease": result.disease,
        "spray_intent": result.spray_intent.value,
        "decision": result.decision.value,
        "decision_label": decision_label,
        "reason": reason_text,
        "confidence_score": result.confidence_score,
        "confidence_factors": result.confidence_factors,
        "best_window": window_data,
        "agronomy": agronomy_details,
        "wash_off_alert": wash_off_alert,
        "seasonal_context_note": seasonal_note,
        "disclaimer": result.disclaimer,
        "kvk_guidance": kvk_fallback,
        "citations": citations,
    }

