import pytest
from scripts.verify_curated_chunks import verify_and_report_diffs

def test_curated_chunks_match_raw_sources_with_approved_edits():
    """
    Verifies that every hand-curated English agronomy chunk matches its raw source page text,
    and every single difference is an explicit, approved edit (OCR repair, label prefix, or punctuation).
    Fails if any unapproved edits, inventions, or paraphrases are introduced.
    """
    reports = verify_and_report_diffs(verbose=False)
    assert len(reports) == 3
    for rep in reports:
        assert rep["status"] == "APPROVED"
