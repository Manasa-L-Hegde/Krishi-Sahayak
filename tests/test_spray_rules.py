from __future__ import annotations

import pytest
from genai.spray_rules import (
    ARECANUT_KOLEROGA_RULE,
    get_spray_rule,
    AgronomicSprayRule,
)


def test_get_spray_rule_arecanut_koleroga():
    rule = get_spray_rule("arecanut", "koleroga")
    assert rule is not None
    assert rule.crop == "arecanut"
    assert rule.target_disease == "koleroga"
    assert rule.causal_organism == "Phytophthora meadii"
    assert rule.fungicide_name == "Bordeaux mixture (1%)"

    # Verify preparation 1kg/50L + 1kg/50L
    assert rule.preparation["copper_sulphate_kg"] == 1.0
    assert rule.preparation["copper_sulphate_water_l"] == 50.0
    assert rule.preparation["lime_kg"] == 1.0
    assert rule.preparation["lime_water_l"] == 50.0
    assert "ext17_pdf" in rule.preparation["source"]

    # Verify 40-45 day repeat interval
    assert rule.repeat_interval_days_min == 40
    assert rule.repeat_interval_days_max == 45

    # Verify unverified items are strictly None
    assert rule.min_dry_hours is None
    assert rule.calendar_start_dates is None
    assert "Krishi Vigyan Kendra" in rule.kvk_fallback_message_en
    assert "ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ" in rule.kvk_fallback_message_kn

    # Verify Mandipropamid dosage 1 ml/L
    assert rule.alternative_fungicide is not None
    assert rule.alternative_fungicide["fungicide_name"] == "Mandipropamid 23.4% SC"
    assert rule.alternative_fungicide["dose_per_litre"] == "1 ml / L"

    # Verify polythene cover specs
    assert "125 gauge, 75 x 60 cm" in rule.bunch_covering["material"]


def test_get_spray_rule_aliases():
    assert get_spray_rule("Arecanut", "Mahali") is not None
    assert get_spray_rule("arecanut", "fruit rot") is not None
    assert get_spray_rule("arecanut", "fruit_rot") is not None


def test_get_spray_rule_fail_closed_for_unknown():
    assert get_spray_rule("paddy", "blast") is None
    assert get_spray_rule("cotton", "bollworm") is None
    assert get_spray_rule("arecanut", "unknown_disease") is None


def test_provenance_integrity():
    rule = ARECANUT_KOLEROGA_RULE
    prov = rule.provenance
    assert "ext17" in prov
    assert "leaflet_p2" in prov
    assert "the_hindu_2024" in prov

    # Ext17 checks
    assert prov["ext17"].page == 2
    assert "cpcri.gov.in" in prov["ext17"].url
    assert "1 % Bordeaux mixture" in prov["ext17"].verbatim_text

    # Leaflet checks
    assert prov["leaflet_p2"].page == 2
    assert "125 gauge, 75 x 60 cm" in prov["leaflet_p2"].verbatim_text

    # The Hindu checks
    assert prov["the_hindu_2024"].source_type == "News report of a CPCRI release"
    assert "Mandipropamid 23.4% SC (1 ml per litre)" in prov["the_hindu_2024"].verbatim_text

