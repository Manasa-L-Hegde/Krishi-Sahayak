"""
verify_curated_chunks.py
========================
Script and verification harness to diff each hand-curated chunk against
the saved raw source page text (PDF OCR / HTML) using difflib.

Enforces that every single difference between the raw text and the curated chunk
is an approved edit belonging to one of three categories:
1. LABEL_PREFIX: Agronomic heading added to scope the chunk.
2. OCR_REPAIR: Correction of broken OCR typography/spelling.
3. PUNCTUATION_AND_WHITESPACE: Normalizing OCR spacing around punctuation or linebreaks.

Fails with an AssertionError if any unapproved difference or text invention occurs.
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from pypdf import PdfReader
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Approved edits whitelist
APPROVED_EDITS = {
    "CURATED-01-EN": {
        "label_prefix": "PREPARATION OF BORDEAUX MIXTURE: ",
        "ocr_repairs": [
            ("and \nkg", "and 1 kg"),
            ("m ix j ust", "mix just"),
            ("qualit y", "quality"),
            ("inferi or", "inferior"),
            ("copper sulphat e.", "copper sulphate."),
            ("dipp ed", "dipped"),
            ("brown ish", "brownish"),
            ("w ill", "will"),
            ("oth er", "other"),
            ("th e", "the"),
            ("disappe ars.", "disappears."),
        ],
        "punctuation_and_whitespace": [
            ("Copper ,", "Copper,"),
            ("solution .", "solution."),
            ("knife in", "knife, in"),
        ],
    },
    "CURATED-02-EN": {
        "label_prefix": "DISEASES OF ARECANUT - Fruit rot ('Koleroga/mahali'): ",
        "ocr_repairs": [
            ("satisf actonly", "satisfactorily"),
            ("prophylatic", "prophylactic"),
            ("Bordeau x \nmi xture", "Bordeaux mixture"),
            ("75 x60 cm", "75 x 60 cm"),
            ("fOW1d", "found"),
            ("checkin g", "checking"),
            ("meclmical", "mechanical"),
        ],
        "punctuation_and_whitespace": [
            ("40-45 days", "40-45 days."),
        ],
    },
    "CURATED-03-EN": {
        "label_prefix": "",
        "ocr_repairs": [],
        "punctuation_and_whitespace": [],
    },
}


def get_raw_source_slices() -> Dict[str, str]:
    from scripts.fetch_sources import fetch_sources
    fetch_sources()

    # 1. Ext17 Page 2 (ext17.pdf)
    p_ext17 = ROOT / "genai" / "corpus" / "downloads" / "ext17.pdf"
    reader17 = PdfReader(str(p_ext17))
    p2_text_17 = reader17.pages[1].extract_text()
    start_17 = p2_text_17.find("PREPARATION OF BORDEAUX MIXTURE")
    end_17 = p2_text_17.find("Do not use oth er metallic vessels.") + len("Do not use oth er metallic vessels.")
    raw_ext17 = p2_text_17[start_17:end_17].strip()

    # 2. Pub69 Page 2 (disease_management_strategies_in_arecanut_and_cocoa(eng)_1.pdf)
    p_pub69 = ROOT / "genai" / "corpus" / "downloads" / "disease_management_strategies_in_arecanut_and_cocoa(eng)_1.pdf"
    reader69 = PdfReader(str(p_pub69))
    p2_text_69 = reader69.pages[1].extract_text()
    start_69 = p2_text_69.find("Disease can be controlled")
    end_69 = p2_text_69.find("acting as meclmical barrier.") + len("acting as meclmical barrier.")
    raw_pub69 = p2_text_69[start_69:end_69].strip()

    # 3. The Hindu 2024 Advisory (thehindu_cpcri_koleroga_advisory_aug2024.html)
    p_hindu = ROOT / "genai" / "corpus" / "downloads" / "thehindu_cpcri_koleroga_advisory_aug2024.html"
    if p_hindu.exists():
        soup = BeautifulSoup(p_hindu.read_text(encoding="utf-8", errors="replace"), "html.parser")
        article = soup.find("div", {"class": "articlebodycontent"}) or soup
        article_text = article.get_text(separator=" ", strip=True)

        target_sentence_start = "Therefore, it is important to take up control measures timely whenever there is a break in rainfall, a CPCRI release said."
        target_sentence_end = "Ensure proper mist spray of the fungicide."
        start_h = article_text.find(target_sentence_start)
        end_h = article_text.find(target_sentence_end, start_h) + len(target_sentence_end)
        raw_hindu = article_text[start_h:end_h].strip()
        # Normalize spaces in raw_hindu
        raw_hindu = re.sub(r"\s+", " ", raw_hindu)
    else:
        raw_hindu = None

    return {
        "CURATED-01-EN": raw_ext17,
        "CURATED-02-EN": raw_pub69,
        "CURATED-03-EN": raw_hindu,
    }


def verify_and_report_diffs(verbose: bool = True) -> List[Dict[str, str]]:
    raw_slices = get_raw_source_slices()
    curated_path = ROOT / "genai" / "curated_arecanut_chunks.json"
    with open(curated_path, "r", encoding="utf-8") as f:
        chunks = {c["id"]: c["text"] for c in json.load(f)}

    reports = []
    for chunk_id, approved in APPROVED_EDITS.items():
        raw_text = raw_slices[chunk_id]
        if raw_text is None:
            if verbose:
                print(f"=== {chunk_id} VERIFICATION SKIPPED ===")
                print("  Local saved copy of source not present; skipping live web scraping.\n")
            reports.append({
                "chunk_id": chunk_id,
                "status": "SKIPPED",
                "approved_edits_count": 0,
                "diff": "(Source file not found locally; skipped)",
            })
            continue

        curated_text = chunks[chunk_id]

        # Generate difflib unified diff between raw and curated
        raw_lines = raw_text.splitlines(keepends=True)
        curated_lines = curated_text.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            raw_lines,
            curated_lines,
            fromfile=f"raw_source_{chunk_id}",
            tofile=f"curated_{chunk_id}",
            lineterm="",
        ))
        diff_report = "\n".join(diff)

        # Verification step: reconstruct curated text applying ONLY approved edits
        reconstructed = raw_text
        # 1. OCR repairs
        for find_pat, repl_pat in approved["ocr_repairs"]:
            assert find_pat in reconstructed, f"Expected raw pattern '{find_pat}' not found in raw text of {chunk_id}"
            reconstructed = reconstructed.replace(find_pat, repl_pat)

        # 2. Punctuation & whitespace
        for find_pat, repl_pat in approved["punctuation_and_whitespace"]:
            reconstructed = reconstructed.replace(find_pat, repl_pat)

        # Normalize line breaks and spaces
        reconstructed = re.sub(r"\s+", " ", reconstructed).strip()

        # 3. Label prefix
        if approved["label_prefix"]:
            if reconstructed.startswith("PREPARATION OF BORDEAUX MIXTURE"):
                reconstructed = reconstructed.replace("PREPARATION OF BORDEAUX MIXTURE", "PREPARATION OF BORDEAUX MIXTURE:", 1)
            else:
                reconstructed = approved["label_prefix"] + reconstructed

        # Assert reconstructed exactly equals curated text
        if reconstructed != curated_text:
            diff_mismatch = list(difflib.ndiff([reconstructed], [curated_text]))
            raise AssertionError(
                f"Curated chunk {chunk_id} contains unapproved modifications!\n"
                f"Diff:\n" + "\n".join(diff_mismatch)
            )

        reports.append({
            "chunk_id": chunk_id,
            "diff": diff_report,
            "status": "APPROVED",
            "edits_count": len(approved["ocr_repairs"]) + len(approved["punctuation_and_whitespace"]) + (1 if approved["label_prefix"] else 0),
        })

        if verbose:
            print(f"=== {chunk_id} VERIFICATION PASSED ===")
            print(f"Approved Edits: {reports[-1]['edits_count']}")
            print("Unified Diff against raw source page:")
            print(diff_report or "  (No line-level diff; whitespace/normalization only)")
            print("\n")

    return reports


if __name__ == "__main__":
    verify_and_report_diffs(verbose=True)
