from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from genai.rag import FALLBACK, LocalRAG

CASES = [
    {"id": "A01", "question": "What is the Agriculture Infrastructure Fund?", "language": "English", "source": "Agriculture Infrastructure Fund Operational Guidelines", "terms": ["Agriculture Infrastructure"]},
    {"id": "A02", "question": "What are the PM-RKVY Operational Guidelines?", "language": "English", "source": "PM-RKVY Operational Guidelines", "terms": ["PM-RKVY"]},
    {"id": "A03", "question": "What does the ICAR Crop Science Division cover?", "language": "English", "source": "ICAR Crop Science Division", "terms": ["crop"]},
    {"id": "A04", "question": "What is agricultural extension?", "language": "English", "source": "ICAR Agricultural Extension Division", "terms": ["extension"]},
    {"id": "A05", "question": "What is the School Soil Health Programme?", "language": "English", "source": "School Soil Health Programme Letter", "terms": ["soil"]},
    {"id": "A06", "question": "What is the purpose of the school soil health programme?", "language": "English", "source": "School Soil Health Programme Letter", "terms": ["soil"]},
    {"id": "A07", "question": "ಕೃಷಿ ವಿಜ್ಞಾನಕ್ಕೆ ಸಂಬಂಧಿಸಿದ ICAR ವಿಭಾಗ ಯಾವುದು?", "language": "Kannada", "source": "ICAR Crop Science Division", "terms": ["crop"]},
    {"id": "A08", "question": "ಶಾಲಾ ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಯಕ್ರಮ", "language": "Kannada", "source": "School Soil Health Programme Letter", "terms": ["soil"]},
    {"id": "A09", "question": "PM-RKVY ಮಾರ್ಗಸೂಚಿಗಳು", "language": "Kannada", "source": "PM-RKVY Operational Guidelines", "terms": ["PM-RKVY"]},
    {"id": "A10", "question": "What does ICAR natural resource management cover?", "language": "English", "source": "ICAR Natural Resource Management Division", "terms": ["resource"]},
    {"id": "O01", "question": "What pesticide dose should I spray on my cotton tomorrow?", "language": "English", "source": None, "terms": []},
    {"id": "O02", "question": "What will onion prices be in my village next month?", "language": "English", "source": None, "terms": []},
    {"id": "O03", "question": "ನನ್ನ ಜಮೀನಿನ ನಾಳೆಯ ಹವಾಮಾನ ಹೇಗಿರುತ್ತದೆ?", "language": "Kannada", "source": None, "terms": []},
    {"id": "O04", "question": "Which tractor should I buy and what is its financing rate?", "language": "English", "source": None, "terms": []},
    {"id": "O05", "question": "ನನ್ನ ಬೆಳೆಗೆ ರೋಗ ಬಂದಿದೆ, ನಿಖರ ಔಷಧಿ ಯಾವುದು?", "language": "Kannada", "source": None, "terms": []},
]


def main() -> None:
    rag = LocalRAG(Path("genai/index"))
    rows = []
    for case in CASES:
        hits = rag.retrieve(case["question"])
        top = hits[0] if hits else None
        retrieved_right = bool(top and case["source"] and top["source"] == case["source"])
        grounded = bool(case["source"] and any(term.casefold() in " ".join(hit["text"] for hit in hits).casefold() for term in case["terms"]))
        fallback_correct = case["source"] is None and (not hits or top["similarity"] < 0.59)
        rows.append({"id": case["id"], "question": case["question"], "language": case["language"],
                     "expected_source": case["source"] or "OUT_OF_CORPUS", "top_source": top["source"] if top else "NONE",
                     "top_similarity": top["similarity"] if top else 0, "retrieval_right_chunk": retrieved_right,
                     "grounded_context": grounded, "fallback_correct": fallback_correct})
    Path("diagnostics").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("diagnostics/rag_evaluation.csv", index=False)
    print(json.dumps(rows, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
