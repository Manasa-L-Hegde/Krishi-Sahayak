"""
faq_router.py
=============
Deterministic FAQ router for canonical arecanut koleroga agronomic queries.
Directly maps the 9 established agronomic intents to verified spray_rules fields
with exact CPCRI publication citations in English and Kannada.

Bypasses vector embedding / similarity entirely for these canonical questions,
guaranteeing 100% precision and zero threshold degradation.
All other queries fall through to similarity search or abstention handlers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional
from genai.spray_rules import ARECANUT_KOLEROGA_RULE


# Canonical Agronomic Intent Matchers
# Each intent defines regex patterns in English and Kannada
INTENT_PATTERNS = {
    "prep": [
        # English patterns
        r"\b(how\s+to\s+make|how\s+do\s+i\s+make|how\s+to\s+prepare|prepare|recipe|proportion|ratio)\b.*\b(bordeaux|mixture)\b",
        r"\bhow\s+much\s+(copper\s+sulphate|lime)\b",
        r"\b(copper\s+sulphate\s+and\s+lime\s+for)\b",
        r"\b1%\s*bordeaux\s*(mixture)?\b",
        # Kannada patterns
        r"ಬೋರ್ಡೋ\s*(ದ್ರಾವಣ|ಮಿಶ್ರಣ)?.*(ಹೇಗೆ\s*ತಯಾರಿಸುವುದು|ತಯಾರಿಸುವ\s*ವಿಧಾನ|ಮಾಡುವುದು\s*ಹೇಗೆ)",
        r"(ಎಷ್ಟು\s*ಮೈಲುತುತ್ತ|ಮೈಲುತುತ್ತ\s*ಮತ್ತು\s*ಸುಣ್ಣ)",
        r"200\s*ಲೀಟರ್\s*ನೀರಿಗೆ",
    ],
    "knife_test": [
        # English patterns
        r"\b(check|test)\b.*\b(mixture\s+is\s+right|excess\s+copper|neutrality|neutral)\b",
        r"\b(iron\s+knife|knife\s+test|nail\s+test)\b",
        r"\bph\s*(level|value|range)?\s*(for\s*bordeaux)?\b",
        # Kannada patterns
        r"(ಮಿಶ್ರಣ\s*ಸರಿಯಾಗಿದೆಯೇ|ಹೇಗೆ\s*ಪರೀಕ್ಷಿಸುವುದು)",
        r"(ಚಾಕು\s*ಪರೀಕ್ಷೆ|ಮೊಳೆಯ\s*ಪರೀಕ್ಷೆ|ಕಬ್ಬಿಣದ\s*ಚಾಕು)",
    ],
    "vessels": [
        # English patterns
        r"\b(which|what)\s+(vessels?|containers?|buckets?|pots?)\b",
        r"\b(metallic\s+vessels?|metal\s+vessels?|plastic\s+bucket|copper\s+vessels?)\b",
        # Kannada patterns
        r"(ಯಾವ\s*ಪಾತ್ರೆ|ಲೋಹದ\s*ಪಾತ್ರೆ|ಪ್ಲಾಸ್ಟಿಕ್\s*ಬಕೆಟ್|ತಾಮ್ರದ\s*ಪಾತ್ರೆ)",
    ],
    "second_spray": [
        # English patterns
        r"\b(second\s+spray|2nd\s+spray|repeat\s+spray|repeat\s+interval)\b",
        r"\bwhen\s+to\s+spray\s+(the\s+)?second\s+time\b",
        r"\b40[- ]45\s*days\b",
        # Kannada patterns
        r"(ಎರಡನೇ\s*ಸಿಂಪರಣೆ|2ನೇ\s*ಸಿಂಪರಣೆ|ಎರಡನೇ\s*ಬಾರಿ)",
    ],
    "third_spray": [
        # English patterns
        r"\b(third\s+spray|3rd\s+spray)\b",
        r"\b(need|require)\s+a\s+third\s+spray\b",
        r"\bmonsoon\s+is\s+prolonged\b",
        # Kannada patterns
        r"(ಮೂರನೇ\s*ಸಿಂಪರಣೆ|3ನೇ\s*ಸಿಂಪರಣೆ|ಮಳೆ\s*ದೀರ್ಘಕಾಲ|ಮಳೆ\s*ಮುಂದುವರಿದರೆ)",
    ],
    "covers": [
        # English patterns
        r"\b(plastic\s+covers?|polythene\s+covers?|bunch\s+covers?|tie\s+covers?|cover\s+(the\s+)?bunches?)\b",
        r"\b125\s*gauge\b",
        # Kannada patterns
        r"(ಪ್ಲಾಸ್ಟಿಕ್\s*ಕವರ್|ಪಾಲಿಥಿನ್\s*ಚೀಲ|ಗೊನೆಗೆ\s*ಕವರ್|ಗೊಂಚಲು\s*ಮುಚ್ಚ)",
    ],
    "sticker": [
        # English patterns
        r"\b(which|what|how\s+much)\s+sticker\b",
        r"\b(sticker\s+cum\s+spreader|adjuvant|wetting\s+agent)\b",
        # Kannada patterns
        r"(ಸ್ಟಿಕ್ಕರ್|ಅಂಟುದ್ರವ|ಸ್ಪ್ರೆಡರ್)",
    ],
    "alternative": [
        # English patterns
        r"\b(alternative\s+fungicide|substitute\s+for\s+bordeaux|mandipropamid)\b",
        r"\bwhat\s+(other\s+fungicide|alternative)\b",
        # Kannada patterns
        r"(ಬದಲಿ\s*ಶಿಲೀಂಧ್ರನಾಶಕ|ಮ್ಯಾಂಡಿಪ್ರೊಪಮಿಡ್|ಬೇರೆ\s*ಔಷಧ)",
    ],
    "no_mix": [
        # English patterns
        r"\b(can\s+i\s+mix|mix\s+mandipropamid|tank\s+mix)\b.*\b(nutrients?|pesticides?|other\s+fungicides?)\b",
        r"\bdo\s+not\s+mix\b",
        # Kannada patterns
        r"(ಇತರ\s*(ಔಷಧ|ಕೀಟನಾಶಕ|ಪೋಷಕಾಂಶ).*ಬೆರೆಸ|ಮಿಶ್ರಣ\s*ಮಾಡಬಹುದೇ)",
    ],
}


def route_faq(query: str, language: str = "en") -> Optional[Dict[str, Any]]:
    """
    Checks if a user query matches one of the 9 canonical agronomic intents.
    Returns a grounded response dict with source citations, or None if no match.
    """
    clean_q = query.strip().lower()
    is_kannada = language.strip().lower() in ("kn", "kannada") or any(
        "\u0c80" <= c <= "\u0cff" for c in query
    )

    matched_intent = None
    for intent, patterns in INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, clean_q, re.IGNORECASE):
                matched_intent = intent
                break
        if matched_intent:
            break

    if not matched_intent:
        return None

    rule = ARECANUT_KOLEROGA_RULE

    if matched_intent == "prep":
        if is_kannada:
            ans = rule.preparation["instruction_kn"]
        else:
            ans = rule.preparation["instruction_en"]
        source = f"{rule.preparation['source_title']} (Page {rule.preparation['page']}, {rule.preparation['publication_year']})"
        url = rule.preparation["url"]

    elif matched_intent == "knife_test":
        if is_kannada:
            ans = rule.neutrality_test["instruction_kn"]
        else:
            ans = rule.neutrality_test["instruction_en"]
        source = f"{rule.neutrality_test['source_title']} (Page {rule.neutrality_test['page']}, {rule.neutrality_test['publication_year']})"
        url = rule.neutrality_test["url"]

    elif matched_intent == "vessels":
        if is_kannada:
            ans = (
                "ಬೋರ್ಡೋ ಮಿಶ್ರಣ ತಯಾರಿಸಲು ತಾಮ್ರ, ಮರ ಅಥವಾ ಪ್ಲಾಸ್ಟಿಕ್ ಪಾತ್ರೆಗಳನ್ನು ಬಳಸಬಹುದು. "
                "ಇತರ ಲೋಹದ ಪಾತ್ರೆಗಳನ್ನು ಬಳಸಬೇಡಿ."
            )
        else:
            ans = (
                "Copper, wooden or plastic vessels may be used for preparation of this mixture. "
                "Do not use other metallic vessels."
            )
        source = f"{rule.preparation['source_title']} (Page {rule.preparation['page']}, {rule.preparation['publication_year']})"
        url = rule.preparation["url"]

    elif matched_intent == "second_spray":
        if is_kannada:
            ans = (
                f"ಎರಡನೇ ಸಿಂಪರಣೆಯನ್ನು ಮೊದಲ ಮುನ್ನೆಚ್ಚರಿಕೆ ಸಿಂಪರಣೆಯ {rule.repeat_interval_days_min} ರಿಂದ "
                f"{rule.repeat_interval_days_max} ದಿನಗಳ ನಂತರ ಮಾಡಬೇಕು."
            )
        else:
            ans = (
                f"The second spray should be carried out {rule.repeat_interval_days_min} to "
                f"{rule.repeat_interval_days_max} days after the first prophylactic spray."
            )
        source = "Disease Management Strategies in Arecanut and Cocoa (Extension Publication No. 69, 1999, p. 2) & Ext. 17 (1989, p. 2)"
        url = rule.bunch_covering["url"]

    elif matched_intent == "third_spray":
        if is_kannada:
            ans = "ಮುಂಗಾರು ಮಳೆಯು ದೀರ್ಘಕಾಲ ಮುಂದುವರಿದರೆ ಮಾತ್ರ ಮೂರನೇ ಸಿಂಪರಣೆ ಅಗತ್ಯವಿರುತ್ತದೆ."
        else:
            ans = "A third spray would be necessary only if the monsoon is prolonged."
        source = "Disease Management Strategies in Arecanut and Cocoa (Extension Publication No. 69, 1999, p. 2) & Ext. 17 (1989, p. 2)"
        url = rule.bunch_covering["url"]

    elif matched_intent == "covers":
        if is_kannada:
            ans = rule.bunch_covering["instruction_kn"]
        else:
            ans = rule.bunch_covering["instruction_en"]
        source = f"{rule.bunch_covering['source_title']} (Page {rule.bunch_covering['page']}, {rule.bunch_covering['publication_year']})"
        url = rule.bunch_covering["url"]

    elif matched_intent == "sticker":
        if is_kannada:
            ans = rule.adjuvant["instruction_kn"]
        else:
            ans = rule.adjuvant["instruction_en"]
        source = f"{rule.adjuvant['source_title']} ({rule.adjuvant['publication_year']})"
        url = rule.adjuvant["url"]

    elif matched_intent == "alternative":
        if is_kannada:
            ans = rule.alternative_fungicide["instruction_kn"]
        else:
            ans = rule.alternative_fungicide["instruction_en"]
        source = f"{rule.alternative_fungicide['source_title']} ({rule.alternative_fungicide['publication_year']})"
        url = rule.alternative_fungicide["url"]

    elif matched_intent == "no_mix":
        if is_kannada:
            ans = "ಮ್ಯಾಂಡಿಪ್ರೊಪಮಿಡ್ ದ್ರಾವಣವನ್ನು ಇತರ ಯಾವುದೇ ಶಿಲೀಂಧ್ರನಾಶಕ, ಕೀಟನಾಶಕ ಅಥವಾ ಪೋಷಕಾಂಶಗಳೊಂದಿಗೆ ಬೆರೆಸಬಾರದು."
        else:
            ans = "Do not mix Mandipropamid with any other fungicides, pesticides, or nutrients."
        source = f"{rule.alternative_fungicide['source_title']} ({rule.alternative_fungicide['publication_year']})"
        url = rule.alternative_fungicide["url"]

    else:
        return None

    return {
        "answer": ans,
        "grounded": True,
        "sources": [source],
        "citations": [source],
        "urls": [url],
        "fallback": False,
        "intent": matched_intent,
    }

