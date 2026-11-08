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
        prep_inst = rule.preparation["instruction_kn"] if is_kn else rule.preparation["instruction_en"]
        prep_src = {
            "title": rule.preparation.get("source_title", "Koleroga (Mahali) of Arecanut (Extension Folder No. 17)"),
            "publication_year": rule.preparation.get("publication_year", 1989),
            "page": rule.preparation.get("page", 2),
            "url": rule.preparation.get("url", "https://cpcri.gov.in/filemgr/webfs/publication/ext17.pdf"),
        }

        test_inst = rule.neutrality_test["instruction_kn"] if is_kn else rule.neutrality_test["instruction_en"]
        test_src = {
            "title": rule.neutrality_test.get("source_title", "Koleroga (Mahali) of Arecanut (Extension Folder No. 17)"),
            "publication_year": rule.neutrality_test.get("publication_year", 1989),
            "page": rule.neutrality_test.get("page", 2),
            "url": rule.neutrality_test.get("url", "https://cpcri.gov.in/filemgr/webfs/publication/ext17.pdf"),
        }

        adj_inst = rule.adjuvant["instruction_kn"] if is_kn else rule.adjuvant["instruction_en"]
        adj_src = {
            "title": rule.adjuvant.get("source_title", "CPCRI issues advisory to control koleroga disease in arecanut plantations (The Hindu, Aug 9, 2024)"),
            "publication_year": rule.adjuvant.get("publication_year", 2024),
            "page": rule.adjuvant.get("page"),
            "url": rule.adjuvant.get("url", "https://www.thehindu.com/news/cities/Mangalore/cpcri-issues-advisory-to-control-koleroga-disease-in-arecanut-plantations/article68506079.ece"),
        }

        repeat_inst = f"{rule.repeat_interval_days_min}-{rule.repeat_interval_days_max} ದಿನಗಳ ನಂತರ" if is_kn else f"{rule.repeat_interval_days_min}-{rule.repeat_interval_days_max} days"
        repeat_src = {
            "title": "Disease Management Strategies in Arecanut and Cocoa (Extension Publication No. 69)",
            "publication_year": 1999,
            "page": 2,
            "url": "https://cpcri.gov.in/filemgr/webfs/publication/disease_management_strategies_in_arecanut_and_cocoa(eng)_1.pdf",
        }

        bunch_inst = rule.bunch_covering["instruction_kn"] if is_kn else rule.bunch_covering["instruction_en"]
        bunch_src = {
            "title": rule.bunch_covering.get("source_title", "Disease Management Strategies in Arecanut and Cocoa (Extension Publication No. 69)"),
            "publication_year": rule.bunch_covering.get("publication_year", 1999),
            "page": rule.bunch_covering.get("page", 2),
            "url": rule.bunch_covering.get("url", "https://cpcri.gov.in/filemgr/webfs/publication/disease_management_strategies_in_arecanut_and_cocoa(eng)_1.pdf"),
        }

        alt_inst = (rule.alternative_fungicide["instruction_kn"] if is_kn else rule.alternative_fungicide["instruction_en"]) if rule.alternative_fungicide else None
        alt_src = {
            "title": rule.alternative_fungicide.get("source_title", "CPCRI issues advisory to control koleroga disease in arecanut plantations (The Hindu, Aug 9, 2024)"),
            "publication_year": rule.alternative_fungicide.get("publication_year", 2024),
            "page": rule.alternative_fungicide.get("page"),
            "url": rule.alternative_fungicide.get("url", "https://www.thehindu.com/news/cities/Mangalore/cpcri-issues-advisory-to-control-koleroga-disease-in-arecanut-plantations/article68506079.ece"),
        } if rule.alternative_fungicide else None

        agronomy_details = {
            "fungicide_name": "1% ತಟಸ್ಥ ಬೋರ್ಡೋ ದ್ರಾವಣ" if is_kn else rule.fungicide_name,
            "preparation_steps": prep_inst,
            "preparation_source": prep_src,
            "quality_test": test_inst,
            "quality_test_source": test_src,
            "adjuvant": adj_inst,
            "adjuvant_source": adj_src,
            "repeat_interval": repeat_inst,
            "repeat_interval_source": repeat_src,
            "bunch_covering": bunch_inst,
            "bunch_covering_source": bunch_src,
            "alternative_fungicide": alt_inst,
            "alternative_fungicide_source": alt_src,
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
    # Disclaimers and Safety Footers
    disclaimer_text = result.disclaimer_kn if is_kn else result.disclaimer_en
    safety_footer_text = result.safety_footer_kn if is_kn else result.safety_footer_en

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
        "seasonal_context_note": seasonal_note,  # TOP of the advisory card
        "decision": result.decision.value,
        "decision_label": decision_label,
        "reason": reason_text,
        "confidence_score": result.confidence_score,
        "confidence_factors": result.confidence_factors,
        "best_window": window_data,
        "agronomy": agronomy_details,
        "wash_off_alert": wash_off_alert,
        "disclaimer": disclaimer_text,
        "kvk_guidance": kvk_fallback,
        "safety_footer": safety_footer_text,  # ALWAYS VISIBLE SAFETY FOOTER
        "citations": citations,
    }

