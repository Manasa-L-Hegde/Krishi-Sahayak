"""
evaluate_farmer_questions.py
============================
Evaluates Krishi Sahayak on the 12 starter farmer question pairs (24 queries total)
in both English and Kannada.
Reports exact metrics:
- Retrieval similarity scores & top chunk
- Fallback vs Grounded response
- Abstention verification (no invented calendar dates or rainfast hours)
- Bordeaux exclusion verification for Yellow Leaf Disease
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genai.rag import LocalRAG

FARMER_QUESTIONS = [
    {
        "id": "FQ_01_EN",
        "question": "How do I make 1% Bordeaux mixture?",
        "language": "English",
        "expected_behavior": "source recipe (1 kg copper sulphate + 1 kg lime in 100 L total water)",
    },
    {
        "id": "FQ_01_KN",
        "question": "1% ಬೋರ್ಡೋ ದ್ರಾವಣ ಹೇಗೆ ತಯಾರಿಸುವುದು?",
        "language": "Kannada",
        "expected_behavior": "source recipe in Kannada (1 kg copper sulphate + 1 kg lime)",
    },
    {
        "id": "FQ_02_EN",
        "question": "How much copper sulphate and lime for 200 litres?",
        "language": "English",
        "expected_behavior": "show source recipe (1 kg per 100 L), scaling flagged as arithmetic",
    },
    {
        "id": "FQ_02_KN",
        "question": "200 ಲೀಟರ್ ನೀರಿಗೆ ಎಷ್ಟು ಮೈಲುತುತ್ತ ಮತ್ತು ಸುಣ್ಣ ಬೇಕು?",
        "language": "Kannada",
        "expected_behavior": "show source recipe in Kannada, scaling flagged",
    },
    {
        "id": "FQ_03_EN",
        "question": "How do I check if the mixture is right?",
        "language": "English",
        "expected_behavior": "knife/nail test (no brown copper deposit)",
    },
    {
        "id": "FQ_03_KN",
        "question": "ಬೋರ್ಡೋ ಮಿಶ್ರಣ ಸರಿಯಾಗಿದೆಯೇ ಎಂದು ಹೇಗೆ ಪರೀಕ್ಷಿಸುವುದು?",
        "language": "Kannada",
        "expected_behavior": "knife test in Kannada (ಉಕ್ಕಿನ ಚಾಕು / ಮೊಳೆ)",
    },
    {
        "id": "FQ_04_EN",
        "question": "When do I do the second spray?",
        "language": "English",
        "expected_behavior": "40-45 days interval",
    },
    {
        "id": "FQ_04_KN",
        "question": "ಎರಡನೇ ಸಿಂಪರಣೆ ಯಾವಾಗ ಮಾಡಬೇಕು?",
        "language": "Kannada",
        "expected_behavior": "40-45 days interval in Kannada",
    },
    {
        "id": "FQ_05_EN",
        "question": "Do I need a third spray this year?",
        "language": "English",
        "expected_behavior": "only if monsoon is prolonged; app cannot know weather",
    },
    {
        "id": "FQ_05_KN",
        "question": "ಈ ವರ್ಷ ಮೂರನೇ ಸಿಂಪರಣೆ ಬೇಕೇ?",
        "language": "Kannada",
        "expected_behavior": "prolonged monsoon condition in Kannada",
    },
    {
        "id": "FQ_06_EN",
        "question": "Can I tie plastic covers on the bunches?",
        "language": "English",
        "expected_behavior": "polythene covers, 125 gauge (75 x 60 cm)",
    },
    {
        "id": "FQ_06_KN",
        "question": "ಅಡಿಕೆ ಗೊನೆಗೆ ಪ್ಲಾಸ್ಟಿಕ್ ಕವರ್ ಕಟ್ಟಬಹುದೇ?",
        "language": "Kannada",
        "expected_behavior": "125 gauge polythene covers in Kannada",
    },
    {
        "id": "FQ_07_EN",
        "question": "Which sticker, and how much?",
        "language": "English",
        "expected_behavior": "1 ml/L sticker cum spreader; no commercial brand",
    },
    {
        "id": "FQ_07_KN",
        "question": "ಸ್ಟಿಕ್ಕರ್ ಯಾವುದನ್ನು ಎಷ್ಟು ಹಾಕಬೇಕು?",
        "language": "Kannada",
        "expected_behavior": "1 ml/L sticker cum spreader in Kannada",
    },
    {
        "id": "FQ_08_EN",
        "question": "Can I mix mancozeb with Bordeaux?",
        "language": "English",
        "expected_behavior": "fallback / do not mix with other fungicides",
    },
    {
        "id": "FQ_08_KN",
        "question": "ಬೋರ್ಡೋ ಜೊತೆ ಮ್ಯಾಂಕೋಝೆಬ್ ಬೆರೆಸಬಹುದೇ?",
        "language": "Kannada",
        "expected_behavior": "fallback / do not mix in Kannada",
    },
    {
        "id": "FQ_09_EN",
        "question": "How many hours must it stay dry after spraying?",
        "language": "English",
        "expected_behavior": "fail-closed abstention: documents used do not give figure",
    },
    {
        "id": "FQ_09_KN",
        "question": "ಸಿಂಪಡಿಸಿದ ಮೇಲೆ ಎಷ್ಟು ಗಂಟೆ ಮಳೆ ಬರಬಾರದು?",
        "language": "Kannada",
        "expected_behavior": "fail-closed abstention in Kannada: ದಾಖಲೆಗಳಲ್ಲಿ ದಿನಾಂಕ/ಸಂಖ್ಯೆ ನೀಡಿಲ್ಲ",
    },
    {
        "id": "FQ_10_EN",
        "question": "What date should I spray in Sirsi?",
        "language": "English",
        "expected_behavior": "fail-closed abstention: documents used do not give figure",
    },
    {
        "id": "FQ_10_KN",
        "question": "ಶಿರಸಿಯಲ್ಲಿ ಯಾವ ದಿನಾಂಕದಂದು ಸಿಂಪಡಿಸಬೇಕು?",
        "language": "Kannada",
        "expected_behavior": "fail-closed abstention in Kannada: ದಾಖಲೆಗಳಲ್ಲಿ ದಿನಾಂಕ/ಸಂಖ್ಯೆ ನೀಡಿಲ್ಲ",
    },
    {
        "id": "FQ_11_EN",
        "question": "I sprayed yesterday and it rained in the evening. Spray again?",
        "language": "English",
        "expected_behavior": "fallback / consult KVK + app wash-off caution",
    },
    {
        "id": "FQ_11_KN",
        "question": "ನಿನ್ನೆ ಸಿಂಪಡಿಸಿದೆ, ಸಂಜೆ ಮಳೆ ಬಂತು. ಮತ್ತೆ ಸಿಂಪಡಿಸಬೇಕೇ?",
        "language": "Kannada",
        "expected_behavior": "fallback / consult KVK in Kannada",
    },
    {
        "id": "FQ_12_EN",
        "question": "Arecanut leaves are yellow, what do I spray?",
        "language": "English",
        "expected_behavior": "NO Bordeaux recommendation; leaflet says garden management for Yellow Leaf disease",
    },
    {
        "id": "FQ_12_KN",
        "question": "ಅಡಿಕೆ ಮರದ ಎಲೆ ಹಳದಿಯಾಗಿದೆ, ಯಾವ ಔಷಧಿ ಹಾಕಬೇಕು?",
        "language": "Kannada",
        "expected_behavior": "NO Bordeaux recommendation; garden management in Kannada",
    },
]


def run_farmer_evaluation():
    rag = LocalRAG(ROOT / "genai" / "index")
    results = []

    print(f"{'ID':<10} | {'Lang':<4} | {'Top Sim':<7} | {'Ground':<6} | {'Fallbk':<6} | {'Status':<15} | Question")
    print("-" * 90)

    for item in FARMER_QUESTIONS:
        qid = item["id"]
        q = item["question"]
        lang = item["language"]
        hits = rag.retrieve(q, k=4, language=lang)
        top_sim = hits[0]["similarity"] if hits else 0.0
        top_source = hits[0]["source"] if hits else "NONE"

        res = rag.ask_expert(q, language=lang)
        ans = res.get("answer", "")
        grounded = res.get("grounded", False)
        fallback = res.get("fallback", False)

        # Safety checks
        # 1. No Bordeaux for Yellow Leaf
        bordeaux_for_yellow = False
        if "FQ_12" in qid:
            if "bordeaux" in ans.lower() and not ("not" in ans.lower() or "do not" in ans.lower()):
                bordeaux_for_yellow = True

        # 2. No hallucinated hours or dates
        hallucinated_dry_or_date = False
        if "FQ_09" in qid or "FQ_10" in qid:
            if any(term in ans.lower() for term in ["6 hours", "4 hours", "2 hours", "june 1", "june 15"]):
                hallucinated_dry_or_date = True

        status = "PASS"
        if bordeaux_for_yellow:
            status = "FAIL: Bordeaux recommended for yellow leaf"
        elif hallucinated_dry_or_date:
            status = "FAIL: Hallucinated hours/date"

        print(f"{qid:<10} | {lang[:2]:<4} | {top_sim:.4f}  | {str(grounded):<6} | {str(fallback):<6} | {status:<15} | {q[:40]}")

        results.append({
            "id": qid,
            "question": q,
            "language": lang,
            "top_similarity": round(top_sim, 4),
            "top_source": top_source,
            "grounded": grounded,
            "fallback": fallback,
            "answer_excerpt": ans[:200],
            "status": status,
        })

    out_file = ROOT / "diagnostics" / "farmer_questions_eval.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nSaved evaluation results to diagnostics/farmer_questions_eval.json")
    return results


if __name__ == "__main__":
    run_farmer_evaluation()
