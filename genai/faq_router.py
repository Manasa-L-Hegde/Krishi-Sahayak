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


UNMENTIONED_FOREIGN_ENTITIES = [
    "mancozeb", "copper oxychloride", "coc", "carbendazim", "chlorpyrifos",
    "quicklime", "slaked lime", "urea", "dap", "potash", "neem oil", "pseudomonas", "trichoderma",
    "ಮ್ಯಾಂಕೋಜೆಬ್", "ಕಾಪರ್ ಆಕ್ಸಿಕ್ಲೋರೈಡ್", "ಕ್ವಿಕ್‌ಲೈಮ್", "ಅರಳಿದ ಸುಣ್ಣ", "ಕ್ಲೋರ್‌ಪೈರಿಫಾಸ್", "ಕಾರ್ಬಂಡೈಸಿಮ್", "ಯೂರಿಯಾ"
]


def check_quantity_guard(clean_q: str, is_kannada: bool) -> Optional[Dict[str, Any]]:
    """Detects requests for non-standard volumes or per-acre/per-palm scalings."""
    quantity_patterns = [
        r"\b50\s*(l|litres?|liters?)\b",
        r"\b200\s*(l|litres?|liters?)\b",
        r"\b500\s*(l|litres?|liters?)\b",
        r"\b1000\s*(l|litres?|liters?)\b",
        r"\bper\s*acre\b",
        r"\bfor\s*(an?|1|one)\s*acre\b",
        r"\bper\s*palm\b",
        r"\bper\s*tree\b",
        r"50\s*ಲೀಟರ್",
        r"200\s*ಲೀಟರ್",
        r"500\s*ಲೀಟರ್",
        r"ಪ್ರತಿ\s*ಎಕರೆ",
        r"ಒಂದು\s*ಎಕರೆಗೆ",
        r"ಪ್ರತಿ\s*ಮರಕ್ಕೆ",
    ]
    has_quantity = any(re.search(pat, clean_q) for pat in quantity_patterns)
    has_spray_context = any(w in clean_q for w in [
        "copper", "lime", "bordeaux", "mixture", "how much", "dose", "water",
        "ಮೈಲುತುತ್ತು", "ಸುಣ್ಣ", "ಬೋರ್ಡೋ", "ದ್ರಾವಣ", "ಎಷ್ಟು", "ನೀರಿಗೆ"
    ])
    if has_quantity and has_spray_context:
        if is_kannada:
            ans = "ಮೂಲ ವಿಧಾನವು ಒಟ್ಟು 100 ಲೀಟರ್‌ಗಳಿಗೆ ನಿಗದಿಯಾಗಿದೆ (50 ಲೀಟರ್‌ಗೆ 1 ಕೆ.ಜಿ ಮೈಲುತುತ್ತು + 50 ಲೀಟರ್‌ಗೆ 1 ಕೆ.ಜಿ ಸುಣ್ಣ). ಇತರ ಪ್ರಮಾಣಗಳಿಗಾಗಿ ನಿಮ್ಮ ಕೆವಿಕೆ / ಆರ್‌ಎಸ್‌ಕೆ ಸಂಪರ್ಕಿಸಿ."
        else:
            ans = "The source recipe makes 100 litres in total (1 kg copper sulphate in 50 L + 1 kg lime in 50 L). For other quantities consult your KVK/RSK."
        return {
            "answer": ans,
            "sources": ["Koleroga (Mahali) of Arecanut (Extension Folder No. 17, 1989, p. 2)"],
            "url": "https://cpcri.gov.in/filemgr/webfs/publication/ext17.pdf",
            "intent": "quantity_scaling",
        }
    return None


def route_faq(query: str, language: str = "en") -> Optional[Dict[str, Any]]:
    """
    Checks if a user query matches one of the 9 canonical agronomic intents.
    Returns a grounded response dict with source citations, or None if no match.
    """
    clean_q = query.strip().lower()
    is_kannada = language.strip().lower() in ("kn", "kannada") or any(
        "\u0c80" <= c <= "\u0cff" for c in query
    )

    # 1. Quantity guard check (50 L, per acre, per palm scaling)
    q_guard = check_quantity_guard(clean_q, is_kannada)
    if q_guard:
        return q_guard

    # 2. Entity guard check: if query mentions foreign entities (e.g. mancozeb, quicklime),
    # do NOT route to standard canonical intents unless query specifically targets Mandipropamid tank mixing
    has_foreign_entity = any(ent in clean_q for ent in UNMENTIONED_FOREIGN_ENTITIES)
    is_mandipropamid = any(m in clean_q for m in ["mandipropamid", "ಮ್ಯಾಂಡಿಪ್ರೊಪಮಿಡ್"])

    if has_foreign_entity and not is_mandipropamid:
        # Fall through to similarity / entity guard for clean refusal
        return None

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

