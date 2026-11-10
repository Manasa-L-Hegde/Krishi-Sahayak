"""
evaluate_confusion_matrix.py
============================
Evaluates Krishi Sahayak's RAG and FAQ routing system on:
- 10 Out-of-Corpus queries (5 English, 5 Kannada)
- 24 Starter Farmer queries (12 English, 12 Kannada)

Evaluates strictly at the live 0.59 retrieval threshold and deterministic FAQ router.
Categorizes every single response into one of four mutually exclusive bins:
1. CORRECT_ANSWER: In-corpus query that yielded a verified, grounded advisory.
2. CORRECT_REFUSAL: Out-of-corpus or ungrounded query (calendar dates, dry hours, unverified tank mixes) correctly refused/abstained.
3. WRONG_REFUSAL: In-corpus query that was falsely refused (loss of recall).
4. WRONG_ACCEPTANCE: Out-of-corpus query that slipped past or returned an unverified answer.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genai.rag import LocalRAG, THRESHOLD, check_abstention_intent, check_yellow_leaf_disease_intent

# 10 Out-of-corpus queries (5 English, 5 Kannada)
OUT_OF_CORPUS_QUERIES = [
    {"id": "OOC_01_EN", "q": "What pesticide dose should I spray on my cotton tomorrow?", "lang": "English"},
    {"id": "OOC_02_EN", "q": "What will onion prices be in my village next month?", "lang": "English"},
    {"id": "OOC_03_EN", "q": "Which tractor should I buy and what is its financing rate?", "lang": "English"},
    {"id": "OOC_04_EN", "q": "What is the recommended dose of urea for sugarcane ratoon crop?", "lang": "English"},
    {"id": "OOC_05_EN", "q": "Can you diagnose why my tomato leaves are curling?", "lang": "English"},
    {"id": "OOC_06_KN", "q": "ನನ್ನ ಜಮೀನಿನ ನಾಳೆಯ ಹವಾಮಾನ ಹೇಗಿರುತ್ತದೆ?", "lang": "Kannada"},
    {"id": "OOC_07_KN", "q": "ನನ್ನ ಹತ್ತಿ ಬೆಳೆಗೆ ಯಾವ ಕೀಟನಾಶಕ ಸಿಂಪಡಿಸಬೇಕು?", "lang": "Kannada"},
    {"id": "OOC_08_KN", "q": "ಟೊಮ್ಯಾಟೊ ಬೆಳೆಗೆ ಎಲೆ ಸುರುಟು ರೋಗಕ್ಕೆ ಏನು ಮಾಡಬೇಕು?", "lang": "Kannada"},
    {"id": "OOC_09_KN", "q": "ಡೀಸೆಲ್ ಪಂಪ್‌ಸೆಟ್‌ಗೆ ಸಬ್ಸಿಡಿ ಎಷ್ಟು ಸಿಗುತ್ತದೆ?", "lang": "Kannada"},
    {"id": "OOC_10_KN", "q": "ಈರುಳ್ಳಿ ದರ ಮುಂದಿನ ವಾರ ಎಷ್ಟು ಇರುತ್ತದೆ?", "lang": "Kannada"},
]

# 24 Starter Farmer Queries (12 English, 12 Kannada)
# Corrected Answer Key:
# - FQ_01 to FQ_07: canonical arecanut koleroga advisories present in ICAR/CPCRI publications -> EXPECTED_ANSWER
# - FQ_08 (mancozeb mixing with Bordeaux): not stated in cited literature -> EXPECTED_REFUSAL
# - FQ_09 (rainfast dry hours): not stated as fixed figure in literature -> EXPECTED_REFUSAL
# - FQ_10 (calendar spray dates): not stated as calendar dates in literature -> EXPECTED_REFUSAL
# - FQ_11 (sprayed yesterday, rained in evening): not stated in cited literature -> EXPECTED_REFUSAL
# - FQ_12 (yellow leaf disease): specific negative-protection advisory in Pub. 69, p. 4 -> EXPECTED_ANSWER
FARMER_QUERIES = [
    {"id": "FQ_01_EN", "q": "How do I make 1% Bordeaux mixture?", "lang": "English", "expected": "ANSWER"},
    {"id": "FQ_01_KN", "q": "1% ಬೋರ್ಡೋ ದ್ರಾವಣ ಹೇಗೆ ತಯಾರಿಸುವುದು?", "lang": "Kannada", "expected": "ANSWER"},
    {"id": "FQ_02_EN", "q": "How much copper sulphate and lime for 200 litres?", "lang": "English", "expected": "ANSWER"},
    {"id": "FQ_02_KN", "q": "200 ಲೀಟರ್ ನೀರಿಗೆ ಎಷ್ಟು ಮೈಲುತುತ್ತ ಮತ್ತು ಸುಣ್ಣ ಬೇಕು?", "lang": "Kannada", "expected": "ANSWER"},
    {"id": "FQ_03_EN", "q": "How do I check if the mixture is right?", "lang": "English", "expected": "ANSWER"},
    {"id": "FQ_03_KN", "q": "ಬೋರ್ಡೋ ಮಿಶ್ರಣ ಸರಿಯಾಗಿದೆಯೇ ಎಂದು ಹೇಗೆ ಪರೀಕ್ಷಿಸುವುದು?", "lang": "Kannada", "expected": "ANSWER"},
    {"id": "FQ_04_EN", "q": "When do I do the second spray?", "lang": "English", "expected": "ANSWER"},
    {"id": "FQ_04_KN", "q": "ಎರಡನೇ ಸಿಂಪರಣೆ ಯಾವಾಗ ಮಾಡಬೇಕು?", "lang": "Kannada", "expected": "ANSWER"},
    {"id": "FQ_05_EN", "q": "Do I need a third spray this year?", "lang": "English", "expected": "ANSWER"},
    {"id": "FQ_05_KN", "q": "ಈ ವರ್ಷ ಮೂರನೇ ಸಿಂಪರಣೆ ಬೇಕೇ?", "lang": "Kannada", "expected": "ANSWER"},
    {"id": "FQ_06_EN", "q": "Can I tie plastic covers on the bunches?", "lang": "English", "expected": "ANSWER"},
    {"id": "FQ_06_KN", "q": "ಅಡಿಕೆ ಗೊನೆಗೆ ಪ್ಲಾಸ್ಟಿಕ್ ಕವರ್ ಕಟ್ಟಬಹುದೇ?", "lang": "Kannada", "expected": "ANSWER"},
    {"id": "FQ_07_EN", "q": "Which sticker, and how much?", "lang": "English", "expected": "ANSWER"},
    {"id": "FQ_07_KN", "q": "ಸ್ಟಿಕ್ಕರ್ ಯಾವುದನ್ನು ಎಷ್ಟು ಹಾಕಬೇಕು?", "lang": "Kannada", "expected": "ANSWER"},
    {"id": "FQ_08_EN", "q": "Can I mix mancozeb with Bordeaux?", "lang": "English", "expected": "REFUSAL"},
    {"id": "FQ_08_KN", "q": "ಬೋರ್ಡೋ ಜೊತೆ ಮ್ಯಾಂಕೋಝೆಬ್ ಬೆರೆಸಬಹುದೇ?", "lang": "Kannada", "expected": "REFUSAL"},
    {"id": "FQ_09_EN", "q": "How many hours must it stay dry after spraying?", "lang": "English", "expected": "REFUSAL"},
    {"id": "FQ_09_KN", "q": "ಸಿಂಪಡಿಸಿದ ಮೇಲೆ ಎಷ್ಟು ಗಂಟೆ ಮಳೆ ಬರಬಾರದು?", "lang": "Kannada", "expected": "REFUSAL"},
    {"id": "FQ_10_EN", "q": "What date should I spray in Sirsi?", "lang": "English", "expected": "REFUSAL"},
    {"id": "FQ_10_KN", "q": "ಶಿರಸಿಯಲ್ಲಿ ಯಾವ ದಿನಾಂಕದಂದು ಸಿಂಪಡಿಸಬೇಕು?", "lang": "Kannada", "expected": "REFUSAL"},
    {"id": "FQ_11_EN", "q": "I sprayed yesterday and it rained in the evening. Spray again?", "lang": "English", "expected": "REFUSAL"},
    {"id": "FQ_11_KN", "q": "ನಿನ್ನೆ ಸಿಂಪಡಿಸಿದೆ, ಸಂಜೆ ಮಳೆ ಬಂತು. ಮತ್ತೆ ಸಿಂಪಡಿಸಬೇಕೇ?", "lang": "Kannada", "expected": "REFUSAL"},
    {"id": "FQ_12_EN", "q": "Arecanut leaves are yellow, what do I spray?", "lang": "English", "expected": "ANSWER"},
    {"id": "FQ_12_KN", "q": "ಅಡಿಕೆ ಮರದ ಎಲೆ ಹಳದಿಯಾಗಿದೆ, ಯಾವ ಔಷಧಿ ಹಾಕಬೇಕು?", "lang": "Kannada", "expected": "ANSWER"},
]

# 5 Empirical Near-Miss Queries
NEAR_MISS_QUERIES = [
    {"id": "NM_01_EN", "q": "Mandipropamid mixed with a micronutrient spray", "lang": "English", "expected": "ANSWER"},
    {"id": "NM_02_EN", "q": "Can I mix Bordeaux mixture with mancozeb or copper oxychloride?", "lang": "English", "expected": "REFUSAL"},
    {"id": "NM_03_EN", "q": "Should I use quicklime or slaked lime for Bordeaux mixture?", "lang": "English", "expected": "REFUSAL"},
    {"id": "NM_04_EN", "q": "How much copper sulphate and lime for 50 L water?", "lang": "English", "expected": "ANSWER"},
    {"id": "NM_05_EN", "q": "What is the treatment for bud rot?", "lang": "English", "expected": "REFUSAL"},
]


def evaluate():
    rag = LocalRAG(ROOT / "genai" / "index")
    records = []

    print(f"Strict Live Threshold: {THRESHOLD}\n")
    print(f"{'ID':<12} | {'Lang':<4} | {'Top Sim':<7} | {'Expected':<10} | {'Classification':<18} | Question")
    print("-" * 100)

    # 1. Evaluate 10 Out-of-Corpus
    for item in OUT_OF_CORPUS_QUERIES:
        qid = item["id"]
        q = item["q"]
        lang = item["lang"]
        hits = rag.retrieve(q, k=4, language=lang)
        top_sim = hits[0]["similarity"] if hits else 0.0
        top_src = hits[0]["source"] if hits else "NONE"

        res = rag.ask_expert(q, language=lang)
        ans = res.get("answer", "")
        grounded = res.get("grounded", False)
        fallback = res.get("fallback", False)

        is_answered = grounded and not fallback
        if is_answered:
            classification = "WRONG_ACCEPTANCE"
        else:
            classification = "CORRECT_REFUSAL"

        print(f"{qid:<12} | {lang[:2]:<4} | {top_sim:.4f}  | {'REFUSAL':<10} | {classification:<18} | {q[:40]}")
        records.append({
            "id": qid,
            "category": "out_of_corpus",
            "question": q,
            "language": lang,
            "top_similarity": round(top_sim, 4),
            "top_source": top_src,
            "grounded": grounded,
            "fallback": fallback,
            "classification": classification,
            "answer_excerpt": ans[:160],
        })

    # 2. Evaluate 24 Farmer Questions
    for item in FARMER_QUERIES:
        qid = item["id"]
        q = item["q"]
        lang = item["lang"]
        exp = item["expected"]
        hits = rag.retrieve(q, k=4, language=lang)
        top_sim = hits[0]["similarity"] if hits else 0.0
        top_src = hits[0]["source"] if hits else "NONE"

        res = rag.ask_expert(q, language=lang)
        ans = res.get("answer", "")
        grounded = res.get("grounded", False)
        fallback = res.get("fallback", False)

        is_answered = grounded and not fallback

        if exp == "ANSWER":
            if is_answered:
                classification = "CORRECT_ANSWER"
            else:
                classification = "WRONG_REFUSAL"
        else: # exp == "REFUSAL"
            if is_answered:
                classification = "WRONG_ACCEPTANCE"
            else:
                classification = "CORRECT_REFUSAL"

        print(f"{qid:<12} | {lang[:2]:<4} | {top_sim:.4f}  | {exp:<10} | {classification:<18} | {q[:40]}")
        records.append({
            "id": qid,
            "category": "farmer_starter",
            "question": q,
            "language": lang,
            "top_similarity": round(top_sim, 4),
            "top_source": top_src,
            "grounded": grounded,
            "fallback": fallback,
            "classification": classification,
            "answer_excerpt": ans[:160],
        })

    # 3. Evaluate 5 Near-Miss Queries
    for item in NEAR_MISS_QUERIES:
        qid = item["id"]
        q = item["q"]
        lang = item["lang"]
        exp = item["expected"]
        hits = rag.retrieve(q, k=4, language=lang)
        top_sim = hits[0]["similarity"] if hits else 0.0
        top_src = hits[0]["source"] if hits else "NONE"

        res = rag.ask_expert(q, language=lang)
        ans = res.get("answer", "")
        grounded = res.get("grounded", False)
        fallback = res.get("fallback", False)

        is_answered = grounded and not fallback

        if exp == "ANSWER":
            if is_answered:
                classification = "CORRECT_ANSWER"
            else:
                classification = "WRONG_REFUSAL"
        else:  # exp == "REFUSAL"
            if is_answered:
                classification = "WRONG_ACCEPTANCE"
            else:
                classification = "CORRECT_REFUSAL"

        print(f"{qid:<12} | {lang[:2]:<4} | {top_sim:.4f}  | {exp:<10} | {classification:<18} | {q[:40]}")
        records.append({
            "id": qid,
            "category": "near_miss",
            "question": q,
            "language": lang,
            "top_similarity": round(top_sim, 4),
            "top_source": top_src,
            "grounded": grounded,
            "fallback": fallback,
            "classification": classification,
            "answer_excerpt": ans[:160],
        })

    # Summary Matrix
    counts = {
        "CORRECT_ANSWER": sum(1 for r in records if r["classification"] == "CORRECT_ANSWER"),
        "CORRECT_REFUSAL": sum(1 for r in records if r["classification"] == "CORRECT_REFUSAL"),
        "WRONG_REFUSAL": sum(1 for r in records if r["classification"] == "WRONG_REFUSAL"),
        "WRONG_ACCEPTANCE": sum(1 for r in records if r["classification"] == "WRONG_ACCEPTANCE"),
    }

    # Language breakdown of wrong refusals
    wrong_refusals_en = sum(1 for r in records if r["classification"] == "WRONG_REFUSAL" and r["language"] == "English")
    wrong_refusals_kn = sum(1 for r in records if r["classification"] == "WRONG_REFUSAL" and r["language"] == "Kannada")

    total = len(records)
    output = {
        "threshold": THRESHOLD,
        "total_queries": total,
        "confusion_matrix": counts,
        "wrong_refusals_by_language": {
            "English": wrong_refusals_en,
            "Kannada": wrong_refusals_kn,
        },
        "rates": {
            "correct_answer_rate": round(counts["CORRECT_ANSWER"] / total * 100, 2),
            "correct_refusal_rate": round(counts["CORRECT_REFUSAL"] / total * 100, 2),
            "wrong_refusal_rate_false_negatives": round(counts["WRONG_REFUSAL"] / total * 100, 2),
            "wrong_acceptance_rate_false_positives": round(counts["WRONG_ACCEPTANCE"] / total * 100, 2),
            "total_safe_or_correct_rate": round((counts["CORRECT_ANSWER"] + counts["CORRECT_REFUSAL"]) / total * 100, 2),
        },
        "records": records,
    }

    out_file = ROOT / "diagnostics" / "eval_confusion_matrix.json"
    out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 50)
    print("4-WAY EVALUATION CONFUSION MATRIX")
    print("=" * 50)
    for k, v in counts.items():
        print(f"  {k:<20}: {v:>2} / {total} ({v/total*100:.1f}%)")
    print(f"\nWrong Refusals by Language:")
    print(f"  English: {wrong_refusals_en}")
    print(f"  Kannada: {wrong_refusals_kn}")
    print(f"\nTotal Safe & Grounded: {counts['CORRECT_ANSWER'] + counts['CORRECT_REFUSAL']} / {total} ({output['rates']['total_safe_or_correct_rate']}%)")
    print(f"Artifact written to {out_file}\n")


if __name__ == "__main__":
    evaluate()
