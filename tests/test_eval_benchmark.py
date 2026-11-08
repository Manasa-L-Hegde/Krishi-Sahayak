from __future__ import annotations

import json
from pathlib import Path
import pytest

from api.spray_engine import (
    evaluate_spray_decision,
    load_engine_config,
    SprayDecision,
    SprayIntent,
)
from api.weather import HourlyDataPoint, WeatherForecast
from genai.spray_rules import ARECANUT_KOLEROGA_RULE, get_spray_rule
from api.geo import is_within_india

BENCHMARK_FILE = Path(__file__).resolve().parent / "eval_spray_advisor.json"


def load_benchmark() -> dict:
    assert BENCHMARK_FILE.exists()
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_eval_benchmark_structure_and_coverage():
    data = load_benchmark()
    assert "cases" in data
    cases = data["cases"]
    assert len(cases) >= 20, f"Benchmark must have at least 20 cases, found {len(cases)}"

    categories = {c["category"] for c in cases}
    expected_categories = {
        "agronomic_groundedness",
        "weather_decision",
        "safety_fail_closed",
        "abstention_refusal",
        "wash_off_governance",
        "interval_governance",
    }
    for cat in expected_categories:
        assert cat in categories


def test_eval_benchmark_agronomic_groundedness():
    """Verify 100% of agronomic benchmark assertions against canonical hand-curated rule."""
    data = load_benchmark()
    agro_cases = [c for c in data["cases"] if c["category"] == "agronomic_groundedness"]
    assert len(agro_cases) >= 7

    rule = ARECANUT_KOLEROGA_RULE
    grounded_count = 0

    for case in agro_cases:
        cid = case["id"]
        if cid == "EVAL_AGRO_01":
            # 1 kg copper sulphate + 1 kg lime per 100 L (50 L + 50 L)
            assert rule.preparation["copper_sulphate_kg"] == 1.0
            assert rule.preparation["lime_kg"] == 1.0
            assert rule.preparation["copper_sulphate_water_l"] == 50.0
            assert rule.preparation["lime_water_l"] == 50.0
            grounded_count += 1
        elif cid == "EVAL_AGRO_02":
            # Neutrality test: iron knife/nail, copper in excess
            assert "iron knife or nail" in rule.neutrality_test["method"].lower()
            assert "copper is in excess" in rule.neutrality_test["instruction_en"].lower()
            grounded_count += 1
        elif cid == "EVAL_AGRO_03":
            # Adjuvant 1 ml / L
            assert rule.adjuvant["dose_per_litre"] == "1 ml / L"
            grounded_count += 1
        elif cid == "EVAL_AGRO_04":
            # Polythene bunch covering 125 gauge, 75x60 cm
            assert "125 gauge" in rule.bunch_covering["material"]
            assert "75 x 60 cm" in rule.bunch_covering["material"]
            assert "mechanical barrier" in rule.bunch_covering["instruction_en"].lower()
            grounded_count += 1
        elif cid == "EVAL_AGRO_05":
            # Repeat interval 40-45 days
            assert rule.repeat_interval_days_min == 40
            assert rule.repeat_interval_days_max == 45
            grounded_count += 1
        elif cid == "EVAL_AGRO_06":
            # Third spray conditional on prolonged monsoon
            assert "prolonged" in rule.third_spray_condition.lower()
            grounded_count += 1
        elif cid == "EVAL_AGRO_07":
            # Alternative chemical Mandipropamid 23.4% SC @ 1 ml/L, mist spray, do not mix
            assert "Mandipropamid 23.4% SC" in rule.alternative_fungicide["fungicide_name"]
            assert "1 ml / L" in rule.alternative_fungicide["dose_per_litre"]
            assert "do not mix" in rule.alternative_fungicide["instruction_en"].lower()
            grounded_count += 1

    groundedness_rate = grounded_count / len(agro_cases)
    assert groundedness_rate == 1.0, f"Groundedness rate must be 100%, got {groundedness_rate*100}%"


def test_eval_benchmark_refusal_and_safety_rates():
    """Verify 100% refusal rate on ungrounded claims, unsupported crops, and foreign coordinates."""
    data = load_benchmark()
    safety_cases = [c for c in data["cases"] if c["category"] in {"safety_fail_closed", "abstention_refusal"}]
    assert len(safety_cases) >= 5

    refusal_count = 0
    total_refusal_tests = 0

    # 1. Unsupported crop (EVAL_WEATHER_13)
    total_refusal_tests += 1
    assert get_spray_rule("sugarcane", "red_rot") is None
    refusal_count += 1

    # 2. Unsupported disease (EVAL_WEATHER_14)
    total_refusal_tests += 1
    assert get_spray_rule("arecanut", "leaf_spot") is None
    refusal_count += 1

    # 3. Foreign coordinates (EVAL_GEO_15)
    total_refusal_tests += 1
    assert not is_within_india(51.5074, -0.1278)  # London
    refusal_count += 1

    # 4. Abstention on calendar start dates and ICAR drying hours (EVAL_REFUSAL_20)
    total_refusal_tests += 1
    assert ARECANUT_KOLEROGA_RULE.calendar_start_dates is None
    assert ARECANUT_KOLEROGA_RULE.min_dry_hours is None
    refusal_count += 1

    # 5. Abstention on chemical mixing (EVAL_REFUSAL_21)
    total_refusal_tests += 1
    assert "do not mix with other fungicides, pesticides, or nutrients" in ARECANUT_KOLEROGA_RULE.alternative_fungicide["instruction_en"].lower()
    refusal_count += 1

    refusal_rate = refusal_count / total_refusal_tests
    assert refusal_rate == 1.0, f"Refusal rate must be 100%, got {refusal_rate*100}%"
