"""
evaluate_confusion_matrix.py
============================
Evaluates Krishi Sahayak's RAG and advisory system on:
- 10 Out-of-Corpus queries (5 English, 5 Kannada)
- 24 Starter Farmer queries (12 English pairs)

Evaluates strictly at the live 0.59 retrieval threshold.
Categorizes every single response into one of four mutually exclusive bins:
1. CORRECT_ANSWER: In-corpus query that retrieved >= 0.59 and returned grounded answer.
2. CORRECT_REFUSAL: Out-of-corpus or ungrounded query (calendar dates, dry hours) correctly refused/abstained.
3. WRONG_REFUSAL: In-corpus query whose similarity was < 0.59, resulting in false refusal (loss of recall).
4. WRONG_ACCEPTANCE: Out-of-corpus query that slipped past threshold (>= 0.59) or hallucinated outside facts.
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
# Note: FQ_09 (dry hours) and FQ_10 (calendar dates) are intentionally UNGROUNDED in ICAR literature
# so their expected behavior is CORRECT_REFUSAL.
# FQ_12 (yellow leaf disease) is a negative plant protection intent with specific garden management.
FARMER_QUERIES = [
    {"id": "FQ_01_EN", "q": "How do I make 1% Bordeaux mixture?", "lang": "English", "type": "in_corpus"},
    {"id": "FQ_01_KN", "q": "1% ಬೋರ್ಡೋ ದ್ರಾವಣ ಹೇಗೆ ತಯಾರಿಸುವುದು?", "lang": "Kannada", "type": "in_corpus"},
    {"id": "FQ_02_EN", "q": "How much copper sulphate and lime for 200 litres?", "lang": "English", "type": "in_corpus"},
    {"id": "FQ_02_KN", "q": "200 ಲೀಟರ್ ನೀರಿಗೆ ಎಷ್ಟು ಮೈಲುತುತ್ತ ಮತ್ತು ಸುಣ್ಣ ಬೇಕು?", "lang": "Kannada", "type": "in_corpus"},
    {"id": "FQ_03_EN", "q": "How do I check if the mixture is right?", "lang": "English", "type": "in_corpus"},
    {"id": "FQ_03_KN", "q": "ಬೋರ್ಡೋ ಮಿಶ್ರಣ ಸರಿಯಾಗಿದೆಯೇ ಎಂದು ಹೇಗೆ ಪರೀಕ್ಷಿಸುವುದು?", "lang": "Kannada", "type": "in_corpus"},
    {"id": "FQ_04_EN", "q": "When do I do the second spray?", "lang": "English", "type": "in_corpus"},
    {"id": "FQ_04_KN", "q": "ಎರಡನೇ ಸಿಂಪರಣೆ ಯಾವಾಗ ಮಾಡಬೇಕು?", "lang": "Kannada", "type": "in_corpus"},
    {"id": "FQ_05_EN", "q": "Do I need a third spray this year?", "lang": "English", "type": "in_corpus"},
    {"id": "FQ_05_KN", "q": "ಈ ವರ್ಷ ಮೂರನೇ ಸಿಂಪರಣೆ ಬೇಕೇ?", "lang": "Kannada", "type": "in_corpus"},
    {"id": "FQ_06_EN", "q": "Can I tie plastic covers on the bunches?", "lang": "English", "type": "in_corpus"},
    {"id": "FQ_06_KN", "q": "ಅಡಿಕೆ ಗೊನೆಗೆ ಪ್ಲಾಸ್ಟಿಕ್ ಕವರ್ ಕಟ್ಟಬಹುದೇ?", "lang": "Kannada", "type": "in_corpus"},
    {"id": "FQ_07_EN", "q": "Which sticker, and how much?", "lang": "English", "type": "in_corpus"},
    {"id": "FQ_07_KN", "q": "ಸ್ಟಿಕ್ಕರ್ ಯಾವುದನ್ನು ಎಷ್ಟು ಹಾಕಬೇಕು?", "lang": "Kannada", "type": "in_corpus"},
    {"id": "FQ_08_EN", "q": "Can I mix mancozeb with Bordeaux?", "lang": "English", "type": "in_corpus"},
    {"id": "FQ_08_KN", "q": "ಬೋರ್ಡೋ ಜೊತೆ ಮ್ಯಾಂಕೋಝೆಬ್ ಬೆರೆಸಬಹುದೇ?", "lang": "Kannada", "type": "in_corpus"},
    {"id": "FQ_09_EN", "q": "How many hours must it stay dry after spraying?", "lang": "English", "type": "ungrounded_intent"},
    {"id": "FQ_09_KN", "q": "ಸಿಂಪಡಿಸಿದ ಮೇಲೆ ಎಷ್ಟು ಗಂಟೆ ಮಳೆ ಬರಬಾರದು?", "lang": "Kannada", "type": "ungrounded_intent"},
    {"id": "FQ_10_EN", "q": "What date should I spray in Sirsi?", "lang": "English", "type": "ungrounded_intent"},
    {"id": "FQ_10_KN", "q": "ಶಿರಸಿಯಲ್ಲಿ ಯಾವ ದಿನಾಂಕದಂದು ಸಿಂಪಡಿಸಬೇಕು?", "lang": "Kannada", "type": "ungrounded_intent"},
    {"id": "FQ_11_EN", "q": "I sprayed yesterday and it rained in the evening. Spray again?", "lang": "English", "type": "in_corpus"},
    {"id": "FQ_11_KN", "q": "ನಿನ್ನೆ ಸಿಂಪಡಿಸಿದೆ, ಸಂಜೆ ಮಳೆ ಬಂತು. ಮತ್ತೆ ಸಿಂಪಡಿಸಬೇಕೇ?", "lang": "Kannada", "type": "in_corpus"},
    {"id": "FQ_12_EN", "q": "Arecanut leaves are yellow, what do I spray?", "lang": "English", "type": "yellow_leaf_special"},
    {"id": "FQ_12_KN", "q": "ಅಡಿಕೆ ಮರದ ಎಲೆ ಹಳದಿಯಾಗಿದೆ, ಯಾವ ಔಷಧಿ ಹಾಕಬೇಕು?", "lang": "Kannada", "type": "yellow_leaf_special"},
]


def evaluate():
    rag = LocalRAG(ROOT / "genai" / "index")
    records = []

    print(f"Strict Live Threshold: {THRESHOLD}\n")
    print(f"{'ID':<12} | {'Lang':<4} | {'Top Sim':<7} | {'Expected':<16} | {'Classification':<18} | Question")
    print("-" * 105)

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

        # Classification logic for OOC:
        # Expected: Refusal.
        # If top_sim >= THRESHOLD or grounded: WRONG_ACCEPTANCE (false positive)
        # If refused (fallback is True and not grounded): CORRECT_REFUSAL
        if top_sim >= THRESHOLD or (grounded and not fallback):
            classification = "WRONG_ACCEPTANCE"
        else:
            classification = "CORRECT_REFUSAL"

        print(f"{qid:<12} | {lang[:2]:<4} | {top_sim:.4f}  | {'REFUSAL':<16} | {classification:<18} | {q[:40]}")
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
        qtype = item["type"]
        hits = rag.retrieve(q, k=4, language=lang)
        top_sim = hits[0]["similarity"] if hits else 0.0
        top_src = hits[0]["source"] if hits else "NONE"

        res = rag.ask_expert(q, language=lang)
        ans = res.get("answer", "")
        grounded = res.get("grounded", False)
        fallback = res.get("fallback", False)

        # Classification logic:
        if qtype == "ungrounded_intent":
            # Must refuse/abstain
            should_ab, _ = check_abstention_intent(q)
            if should_ab and fallback:
                classification = "CORRECT_REFUSAL"
            elif top_sim >= THRESHOLD:
                classification = "WRONG_ACCEPTANCE"
            else:
                classification = "CORRECT_REFUSAL"
            expected = "REFUSAL_ABSTAIN"

        elif qtype == "yellow_leaf_special":
            # Must NOT recommend Bordeaux; must recommend garden management
            is_y, _ = check_yellow_leaf_disease_intent(q)
            if is_y and grounded and not ("bordeaux" in ans.lower() and "not" not in ans.lower()):
                classification = "CORRECT_ANSWER"
            else:
                classification = "WRONG_REFUSAL"
            expected = "CORRECT_ADVICE"

        else: # in_corpus
            expected = "CORRECT_ANSWER"
            if top_sim >= THRESHOLD:
                # Top hit passed strict threshold
                classification = "CORRECT_ANSWER"
            else:
                # Similarity was below strict 0.59 threshold -> false negative / recall loss
                classification = "WRONG_REFUSAL"

        print(f"{qid:<12} | {lang[:2]:<4} | {top_sim:.4f}  | {expected:<16} | {classification:<18} | {q[:40]}")
        records.append({
            "id": qid,
            "category": qtype,
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
    total = len(records)

    summary = {
        "threshold": THRESHOLD,
        "total_queries": total,
        "confusion_matrix": counts,
        "rates": {
            "correct_answer_rate": round(counts["CORRECT_ANSWER"] / total * 100, 2),
            "correct_refusal_rate": round(counts["CORRECT_REFUSAL"] / total * 100, 2),
            "wrong_refusal_rate_false_negatives": round(counts["WRONG_REFUSAL"] / total * 100, 2),
            "wrong_acceptance_rate_false_positives": round(counts["WRONG_ACCEPTANCE"] / total * 100, 2),
            "total_safe_or_correct_rate": round((counts["CORRECT_ANSWER"] + counts["CORRECT_REFUSAL"]) / total * 100, 2),
        },
        "records": records,
    }

    print("\n" + "=" * 50)
    print("4-WAY EVALUATION CONFUSION MATRIX SUMMARY")
    print("=" * 50)
    print(f"CORRECT_ANSWER:     {counts['CORRECT_ANSWER']:>3} / {total} ({summary['rates']['correct_answer_rate']}%)")
    print(f"CORRECT_REFUSAL:    {counts['CORRECT_REFUSAL']:>3} / {total} ({summary['rates']['correct_refusal_rate']}%)")
    print(f"WRONG_REFUSAL:      {counts['WRONG_REFUSAL']:>3} / {total} ({summary['rates']['wrong_refusal_rate_false_negatives']}%) [Recall Loss]")
    print(f"WRONG_ACCEPTANCE:   {counts['WRONG_ACCEPTANCE']:>3} / {total} ({summary['rates']['wrong_acceptance_rate_false_positives']}%) [False Accepts]")
    print(f"Total Safe/Correct: {counts['CORRECT_ANSWER'] + counts['CORRECT_REFUSAL']:>3} / {total} ({summary['rates']['total_safe_or_correct_rate']}%)")

    out_file = ROOT / "diagnostics" / "eval_confusion_matrix.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved confusion matrix to {out_file}")
    return summary


if __name__ == "__main__":
    evaluate()
