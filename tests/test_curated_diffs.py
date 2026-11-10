import pytest
from pathlib import Path
from scripts.verify_curated_chunks import verify_and_report_diffs, ROOT

def test_curated_chunks_match_raw_sources_with_approved_edits():
    """
    Verifies that every hand-curated English agronomy chunk matches its raw source page text,
    and every single difference is an explicit, approved edit (OCR repair, label prefix, or punctuation).
    Fails if any unapproved edits, inventions, or paraphrases are introduced.
    Gracefully skips if raw downloaded source files are not present in the environment.
    """
    downloads = ROOT / "genai" / "corpus" / "downloads"
    required = [
        downloads / "ext17.pdf",
        downloads / "disease_management_strategies_in_arecanut_and_cocoa(eng)_1.pdf",
        downloads / "thehindu_cpcri_koleroga_advisory_aug2024.html",
    ]
    missing = [f.name for f in required if not f.exists()]
    if missing:
        pytest.skip(f"Raw source files missing ({', '.join(missing)}); run scripts/download_sources.py to fetch them")

    reports = verify_and_report_diffs(verbose=False)
    assert len(reports) == 3
    for rep in reports:
        assert rep["status"] == "APPROVED"

