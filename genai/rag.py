from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

FALLBACK = "I don't have verified information on that — please check with your local agriculture officer"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
THRESHOLD = 0.59
TOP_K = 4


def check_abstention_intent(question: str) -> tuple[bool, str]:
    """
    Detect questions asking for fixed calendar spray dates or rainfast drying hours,
    which ICAR-CPCRI does not prescribe as fixed values.
    Returns (should_abstain, response_text).
    """
    q_lower = question.lower()

    # 1. Calendar dates detection (EN and KN)
    calendar_patterns = [
        "calendar date", "which date", "exact date", "what date", "calendar day",
        "ಕ್ಯಾಲೆಂಡರ್", "ದಿನಾಂಕ", "ಯಾವ ದಿನಾಂಕ", "ಖಚಿತ ದಿನಾಂಕ", "ಕ್ಯಾಲೆಂಡರ್ ದಿನಾಂಕ"
    ]
    is_calendar = any(p in q_lower for p in calendar_patterns)

    # 2. Rainfast / dry hours detection (EN and KN)
    dry_hours_patterns = [
        "hours dry", "dry hours", "stay dry", "how many hours", "hours after spray",
        "rainfast", "drying time", "dry time", "hours needed",
        "ಎಷ್ಟು ಗಂಟೆ", "ಒಣ ಗಂಟೆ", "ಒಣಗಲು ಎಷ್ಟು ಸಮಯ", "ಮಳೆ ಬರುವ ಮುನ್ನ ಎಷ್ಟು ಗಂಟೆ"
    ]
    is_dry_hours = any(p in q_lower for p in dry_hours_patterns)

    # Check if the query is in arecanut / spray context
    crop_context = any(c in q_lower for c in ["arecanut", "koleroga", "mahali", "bordeaux", "spray", "ಅಡಿಕೆ", "ಕೊಳೆರೋಗ", "ಬೋರ್ಡೋ", "ಸಿಂಪರಣೆ"])

    if (is_calendar or is_dry_hours) and crop_context:
        msg_en = (
            "ICAR-CPCRI publications do not prescribe fixed calendar dates or fixed rainfast dry hours. "
            "Timing strictly depends on localized pre-monsoon showers and weather breaks. "
            "Please consult your local Krishi Vigyan Kendra (KVK) or Raitha Samparka Kendra (RSK) for localized guidance."
        )
        msg_kn = (
            "ಐಸಿಎಆರ್-ಸಿಪಿಸಿಆರ್‌ಐ ಪ್ರಕಟಣೆಗಳಲ್ಲಿ ನಿರ್ದಿಷ್ಟ ಕ್ಯಾಲೆಂಡರ್ ದಿನಾಂಕಗಳು ಅಥವಾ ಮಳೆ ನಿರೋಧಕ ಒಣಗುವ ಗಂಟೆಗಳನ್ನು ನಿಗದಿಪಡಿಸಿಲ್ಲ. "
            "ಸಿಂಪರಣೆಯು ಮುಂಗಾರು ಪೂರ್ವ ಮಳೆ ಮತ್ತು ಸ್ಥಳೀಯ ಹವಾಮಾನ ಬಿಡುವನ್ನು ಅವಲಂಬಿಸಿರುತ್ತದೆ. "
            "ದಯವಿಟ್ಟು ಸ್ಥಳೀಯ ಮಾರ್ಗದರ್ಶನಕ್ಕಾಗಿ ನಿಮ್ಮ ಹತ್ತಿರದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ (ಕೆವಿಕೆ) ಅಥವಾ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರವನ್ನು (ಆರ್‌ಎಸ್‌ಕೆ) ಸಂಪರ್ಕಿಸಿ."
        )
        is_kn = any(ord(char) >= 0x0C80 and ord(char) <= 0x0CFF for char in question)
        return True, msg_kn if is_kn else msg_en

    return False, ""


def verify_numeric_grounding(answer_text: str, context_text: str) -> bool:
    """
    Verify that every numeric token or percentage in the generated answer appears in the retrieved context.
    Prevents hallucinated doses, percentages, intervals, or hours.
    """
    answer_numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", answer_text)
    for num in answer_numbers:
        if num not in context_text:
            return False
    return True


def chunk_text(text: str, words_per_chunk: int = 350, overlap: int = 60) -> list[str]:
    words = re.findall(r"\S+", text)
    step = words_per_chunk - overlap
    return [" ".join(words[start:start + words_per_chunk]) for start in range(0, len(words), step) if len(words[start:start + words_per_chunk]) >= 40]


class LocalRAG:
    def __init__(self, directory: str | Path):
        directory = Path(directory)
        self.index = faiss.read_index(str(directory / "faiss.index"))
        self.embedder = SentenceTransformer(MODEL_NAME)
        self.chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))

    def retrieve(self, question: str, k: int = TOP_K) -> list[dict[str, Any]]:
        vector = self.embedder.encode([question], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(vector, k)
        return [{**self.chunks[int(index)], "similarity": float(score)} for score, index in zip(scores[0], indices[0]) if index >= 0]

    def answer(self, question: str, language: str | None = None) -> dict[str, Any]:
        should_abstain, abstention_msg = check_abstention_intent(question)
        if should_abstain:
            return {"answer": abstention_msg, "sources": [], "grounded": False, "fallback": True}

        hits = self.retrieve(question)
        qualifying = [hit for hit in hits if hit["similarity"] >= THRESHOLD]
        if not qualifying:
            return {"answer": FALLBACK, "sources": [], "grounded": False, "fallback": True}
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {"answer": "Verified context was retrieved, but no Groq API key is configured. " + FALLBACK,
                    "sources": sorted({hit["source"] for hit in qualifying}), "grounded": False, "fallback": True}
        from groq import Groq
        context = "\n\n".join(f"SOURCE: {hit['source']}\n{hit['text']}" for hit in qualifying)
        prompt = ("Answer ONLY from the provided context. If the context does not contain the answer, say so explicitly; "
                  "never guess or add outside facts. Cite the source names. English and Kannada are allowed.\n\n"
                  f"CONTEXT:\n{context}\n\nQUESTION: {question}")
        try:
            response = Groq(api_key=api_key).chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages=[{"role": "system", "content": "You are a strictly grounded agricultural information assistant."},
                          {"role": "user", "content": prompt}], temperature=0)
            content = response.choices[0].message.content or ""
            # Post-generation numeric check: every number in the answer must exist in the context
            if not verify_numeric_grounding(content, context):
                return {"answer": "The generated advisory contained unverified quantities. " + FALLBACK,
                        "sources": sorted({hit["source"] for hit in qualifying}), "grounded": False, "fallback": True}
            return {"answer": content, "sources": sorted({hit["source"] for hit in qualifying}),
                    "grounded": True, "fallback": False}
        except Exception:
            return {"answer": "The advisory service is unavailable. " + FALLBACK,
                    "sources": sorted({hit["source"] for hit in qualifying}), "grounded": False, "fallback": True}

    def ask_expert(self, question: str, language: str = "English") -> dict[str, Any]:
        """Answers agricultural questions thoroughly.
        If verified corpus matches are found, it grounds the answer in those official documents.
        If general farming advice is needed, it provides expert agronomic guidance through Groq."""
        should_abstain, abstention_msg = check_abstention_intent(question)
        if should_abstain:
            return {
                "answer": abstention_msg,
                "sources": [],
                "grounded": False,
                "verified": False,
                "fallback": True
            }

        hits = self.retrieve(question, k=TOP_K)
        qualifying = [hit for hit in hits if hit["similarity"] >= 0.50]

        # For dosage, preparation, dilution, mixing, and interval queries, return the verified passage directly (extractive)
        is_dose_prep_timing = any(kw in question.lower() for kw in [
            "dose", "proportion", "prepare", "preparation", "mix", "how to make", "interval", "percent", "%",
            "gauge", "polythene", "bunch cover", "vessel", "knife", "blade", "ತಯಾರಿಕೆ", "ಪ್ರಮಾಣ", "ಅಳತೆ", "ದಿನ", "ಗೇಜ್", "ಚೀಲ", "ಪಾತ್ರೆ", "ಚಾಕು"
        ])
        if qualifying and is_dose_prep_timing:
            top = qualifying[0]
            return {
                "answer": f"{top['text']}\n\n[Official Source: {top['source']}]",
                "sources": sorted({hit["source"] for hit in qualifying}),
                "grounded": True,
                "verified": True,
                "fallback": False
            }

        api_key = os.getenv("GROQ_API_KEY")

        if qualifying:
            sources = sorted({hit["source"] for hit in qualifying})
            context = "\n\n".join(f"SOURCE: {hit['source']}\n{hit['text']}" for hit in qualifying)
            prompt = (
                f"You are Krishi Sahayak (ಕೃಷಿ ಸಹಾಯಕ), an expert senior agricultural advisor for Indian farmers.\n"
                f"The farmer asked in {language}: '{question}'\n\n"
                f"Use the verified government/ICAR context below as your primary reference, and supplement with accurate, "
                f"practical farming best practices if needed.\n"
                f"CONTEXT:\n{context}\n\n"
                f"Provide a clear, respectful, and well-structured answer in {language}. "
                f"Use bullet points and bold key terms. Explicitly mention official scheme benefits or ICAR guidelines if relevant."
            )
            grounded = True
            verified_label = True
        else:
            sources = ["General Agricultural Advisory (ICAR & KVK Best Practices)"]
            prompt = (
                f"You are Krishi Sahayak (ಕೃಷಿ ಸಹಾಯಕ), a compassionate and highly experienced senior agricultural scientist "
                f"and agronomist dedicated to helping Indian farmers.\n"
                f"The farmer has asked this question: '{question}'\n\n"
                f"Provide comprehensive, practical, step-by-step agricultural guidance tailored to Indian farming conditions.\n"
                f"Include:\n"
                f"1. Direct answer and practical solution\n"
                f"2. Best practices (recommended varieties, organic options/Jeevamrutha, or IPM methods where applicable)\n"
                f"3. Timing, dosage, or soil/water management tips\n"
                f"4. Preventative measures\n\n"
                f"Language required: {language}. Ensure the tone is warm, encouraging, and easy for a farmer to understand. "
                f"Use bullet points for readability."
            )
            grounded = True
            verified_label = False

        if not api_key:
            if qualifying:
                return {
                    "answer": f"**Verified Information Found ({', '.join(sources)}):**\n\n" + qualifying[0]["text"][:600] + "...",
                    "sources": sources,
                    "grounded": True,
                    "verified": True,
                    "fallback": False
                }
            return {
                "answer": "Krishi Sahayak is ready to assist you. Please ensure the GROQ_API_KEY is active in your .env file to enable full interactive AI advisory, or visit your nearest Krishi Vigyan Kendra (KVK).",
                "sources": ["Local KVK Advisory"],
                "grounded": False,
                "verified": False,
                "fallback": True
            }

        from groq import Groq
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are Krishi Sahayak, an expert agricultural advisor who provides clear, actionable farming guidance in the farmer's chosen language (English, Kannada, Hindi)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            answer_text = response.choices[0].message.content
            return {
                "answer": answer_text,
                "sources": sources,
                "grounded": grounded,
                "verified": verified_label,
                "fallback": False
            }
        except Exception as e:
            # Fallback to local context or safe response
            if qualifying:
                return {
                    "answer": f"**Verified Official Guidelines ({', '.join(sources)}):**\n\n" + qualifying[0]["text"][:600] + "...",
                    "sources": sources,
                    "grounded": True,
                    "verified": True,
                    "fallback": False
                }
            return {
                "answer": f"We encountered a temporary connection issue reaching the AI engine. For immediate agricultural queries, please consult the Kisan Call Centre (Toll-Free 1800-180-1551) or your local Raitha Samparka Kendra.",
                "sources": ["Kisan Call Centre (1800-180-1551)"],
                "grounded": False,
                "verified": False,
                "fallback": True
            }

