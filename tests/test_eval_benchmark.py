from __future__ import annotations

import json
from pathlib import Path
import pytest

BENCHMARK_FILE = Path(__file__).resolve().parent / "eval_spray_advisor.json"


def test_eval_benchmark_structure_and_coverage():
    assert BENCHMARK_FILE.exists()
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "cases" in data
    cases = data["cases"]
    assert len(cases) >= 20, f"Benchmark must have at least 20 cases, found {len(cases)}"

    categories = {c["category"] for c in cases}
    assert "agronomic_groundedness" in categories
    assert "weather_decision" in categories
    assert "safety_fail_closed" in categories
    assert "abstention_refusal" in categories
    assert "wash_off_governance" in categories
    assert "interval_governance" in categories

    for c in cases:
        assert "id" in c
        assert "category" in c
        assert "description" in c

