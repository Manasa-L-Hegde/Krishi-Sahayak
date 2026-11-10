from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

FALLBACK_EN = "I could not find a matching passage in the verified documents; please rephrase or consult your local KVK or RSK."
FALLBACK_KN = "ಪರಿಶೀಲಿಸಿದ ದಾಖಲೆಗಳಲ್ಲಿ ಸೂಕ್ತ ಮಾಹಿತಿ ದೊರೆತಿಲ್ಲ; ದಯವಿಟ್ಟು ವಾಕ್ಯವನ್ನು ಬದಲಿಸಿ ಅಥವಾ ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೆವಿಕೆ / ಆರ್‌ಎಸ್‌ಕೆ ಸಂಪರ್ಕಿಸಿ."
FALLBACK = FALLBACK_EN


def get_low_retrieval_fallback(language: str | None = None, question: str = "") -> str:
    is_kn = any(ord(char) >= 0x0C80 and ord(char) <= 0x0CFF for char in question)
    if language and (language.lower().startswith("kn") or language.lower().startswith("kann")):
        is_kn = True
    return FALLBACK_KN if is_kn else FALLBACK_EN
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
THRESHOLD = 0.59
TOP_K = 4


def check_abstention_intent(question: str) -> tuple[bool, str]:
    """
    Detect questions asking for fixed calendar spray dates or rainfast drying hours,
    which the official documents used by this app do not prescribe as fixed figures.
    Returns (should_abstain, response_text).
    """
    q_lower = question.lower()

    # 1. Calendar dates detection (EN and KN)
    calendar_patterns = [
        "calendar date", "which date", "exact date", "what date", "calendar day",
        "which month", "what day",
        "ಕ್ಯಾಲೆಂಡರ್", "ದಿನಾಂಕ", "ಯಾವ ದಿನಾಂಕ", "ಖಚಿತ ದಿನಾಂಕ", "ಕ್ಯಾಲೆಂಡರ್ ದಿನಾಂಕ", "ಯಾವ ತಾರೀಖು", "ಯಾವ ತಿಂಗಳು"
    ]
    is_calendar = any(p in q_lower for p in calendar_patterns)

    # 2. Rainfast / dry hours detection (EN and KN) including paraphrases
    dry_hours_patterns = [
        "hours dry", "dry hours", "stay dry", "how many hours", "hours after spray",
        "rainfast", "drying time", "dry time", "hours needed",
        "how long should it not rain", "how long must it not rain", "how long before it rains",
        "how long before rain", "rain free hours", "rain-free hours", "dry period",
        "drying hours", "how long dry",
        "ಎಷ್ಟು ಗಂಟೆ", "ಒಣ ಗಂಟೆ", "ಒಣಗಲು ಎಷ್ಟು ಸಮಯ", "ಮಳೆ ಬರುವ ಮುನ್ನ", "ಮಳೆ ಬರಬಾರದು", "ಎಷ್ಟು ಹೊತ್ತು ಮಳೆ", "ಎಷ್ಟು ಕಾಲ ಮಳೆ"
    ]
    is_dry_hours = any(p in q_lower for p in dry_hours_patterns)

    # Check if the query is in arecanut / spray context
    crop_context = any(c in q_lower for c in [
        "arecanut", "koleroga", "mahali", "bordeaux", "spray", "spraying",
        "ಅಡಿಕೆ", "ಕೊಳೆರೋಗ", "ಬೋರ್ಡೋ", "ಸಿಂಪರಣೆ", "ಸಿಂಪಡ"
    ])

    if (is_calendar or is_dry_hours) and crop_context:
        msg_en = "The documents used by this app do not give a figure for this. Please consult your KVK or RSK."
        msg_kn = "ಈ ಆ್ಯಪ್‌ನಲ್ಲಿ ಬಳಸಲಾದ ದಾಖಲೆಗಳಲ್ಲಿ ಇದಕ್ಕೆ ನಿರ್ದಿಷ್ಟ ಸಂಖ್ಯೆ ಅಥವಾ ದಿನಾಂಕ ನೀಡಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಕೆವಿಕೆ ಅಥವಾ ಆರ್‌ಎಸ್‌ಕೆ ಸಂಪರ್ಕಿಸಿ."
        is_kn = any(ord(char) >= 0x0C80 and ord(char) <= 0x0CFF for char in question)
        return True, msg_kn if is_kn else msg_en

    return False, ""


def check_yellow_leaf_disease_intent(question: str) -> tuple[bool, str]:
    """
    According to ICAR-CPCRI (Extension Pub. No. 69), Yellow Leaf Disease of arecanut
    is caused by a phytoplasma and is not amenable to control by conventional plant
    protection measures. Chemical sprays like Bordeaux mixture must NEVER be recommended for it.
    Instead, garden management (NPK fertilizers, organic manure, drainage, cover crops) is recommended.
    """
    q_lower = question.lower()
    is_yellow_leaf = any(k in q_lower for k in [
        "yellow leaf", "leaves are yellow", "leaf yellowing", "yellowing of leaves",
        "leaves yellow", "leaf is yellow",
        "ಎಲೆ ಹಳದಿ", "ಹಳದಿ ಎಲೆ", "ಹಳದಿಯಾಗಿದೆ"
    ])
    is_arecanut = any(k in q_lower for k in ["arecanut", "areca", "palm", "ಮರ", "ಅಡಿಕೆ"])
    if is_yellow_leaf and is_arecanut:
        msg_en = (
            "According to ICAR-CPCRI Extension Publication No. 69, Yellow Leaf Disease of arecanut is "
            "caused by a phytoplasma and is NOT amenable to control by conventional chemical spraying (do NOT spray Bordeaux mixture). "
            "The recommended measure is proper garden management: regular application of balanced fertilizers (100g N, 40g P, 140g K per palm/year in two splits), "
            "additional Mussoorie phos (800g/palm), organic manures and green leaf (12kg each/palm/year), and ensuring good drainage. "
            "Please consult your local Krishi Vigyan Kendra (KVK) for localized diagnosis."
        )
        msg_kn = (
            "ಐಸಿಎಆರ್-ಸಿಪಿಸಿಆರ್‌ಐ ವಿಸ್ತರಣಾ ಪ್ರಕಟಣೆ 69 ರ ಪ್ರಕಾರ, ಅಡಿಕೆ ಮರದ ಹಳದಿ ಎಲೆ ರೋಗವು ಫೈಟೋಪ್ಲಾಸ್ಮಾದಿಂದ ಬರುತ್ತದೆ ಮತ್ತು "
            "ಇದಕ್ಕೆ ಸಾಂಪ್ರದಾಯಿಕ ರಾಸಾಯನಿಕ ಸಿಂಪರಣೆ ಪರಿಣಾಮಕಾರಿಯಲ್ಲ (ಬೋರ್ಡೋ ದ್ರಾವಣವನ್ನು ಸಿಂಪಡಿಸಬಾರದು). "
            "ಇದಕ್ಕೆ ತೋಟದ ನಿರ್ವಹಣೆ ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ: ಸಮತೋಲಿತ ಗೊಬ್ಬರ (ಪ್ರತಿ ಮರಕ್ಕೆ 100 ಗ್ರಾಂ ಸಾರಜನಕ, 40 ಗ್ರಾಂ ರಂಜಕ, 140 ಗ್ರಾಂ ಪೊಟ್ಯಾಷ್), "
            "ಹೆಚ್ಚುವರಿ ಮಸ್ಸೋರಿ ಫಾಸ್ (800 ಗ್ರಾಂ), ಸಾವಯವ ಗೊಬ್ಬರ ಮತ್ತು ಹಸಿರೆಲೆ ಗೊಬ್ಬರ (ತಲಾ 12 ಕೆ.ಜಿ) ನೀಡುವುದು ಹಾಗೂ ಉತ್ತಮ ನೀರು ಬಸಿದುಹೋಗುವ ವ್ಯವಸ್ಥೆ ಮಾಡುವುದು. "
            "ದಯವಿಟ್ಟು ಹೆಚ್ಚಿನ ಮಾಹಿತಿಗಾಗಿ ಸ್ಥಳೀಯ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರವನ್ನು (ಕೆವಿಕೆ) ಸಂಪರ್ಕಿಸಿ."
        )
        is_kn = any(ord(char) >= 0x0C80 and ord(char) <= 0x0CFF for char in question)
        return True, msg_kn if is_kn else msg_en
    return False, ""


def verify_numeric_grounding(answer_text: str, context_text: str) -> bool:
    """
    Verify that every numeric token or percentage in the generated answer exists as an exact
    token in the retrieved context. Prevents hallucinated doses, percentages, intervals, or hours.
    Comparing whole tokens ensures that e.g. '4' does not falsely match inside '40-45'.
    """
    context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", context_text))
    answer_numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", answer_text)
    for num in answer_numbers:
        if num not in context_numbers:
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

    def retrieve(self, question: str, k: int = TOP_K, language: str | None = None) -> list[dict[str, Any]]:
        is_kn = any(ord(char) >= 0x0C80 and ord(char) <= 0x0CFF for char in question)
        target_lang = "kn" if is_kn else "en"
        if language:
            target_lang = "kn" if language.lower().startswith("kn") or language.lower().startswith("kann") else "en"

        vector = self.embedder.encode([question], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(vector, k * 3)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index >= 0:
                chunk = self.chunks[int(index)]
                chunk_id = chunk.get("id", "")
                chunk_text_sample = chunk.get("text", "")
                is_chunk_kn = "-KN" in chunk_id or any(ord(char) >= 0x0C80 and ord(char) <= 0x0CFF for char in chunk_text_sample)
                if is_chunk_kn and target_lang == "en":
                    continue
                if not is_chunk_kn and target_lang == "kn" and "-EN" in chunk_id:
                    continue
                results.append({**chunk, "similarity": float(score)})
                if len(results) >= k:
                    break
        return results

    def answer(self, question: str, language: str | None = None) -> dict[str, Any]:
        should_abstain, abstention_msg = check_abstention_intent(question)
        if should_abstain:
            return {"answer": abstention_msg, "sources": [], "grounded": False, "fallback": True}

        is_yellow, yellow_msg = check_yellow_leaf_disease_intent(question)
        if is_yellow:
            return {
                "answer": yellow_msg,
                "sources": ["Disease Management Strategies in Arecanut and Cocoa (Extension Publication No. 69, 1999, p. 4)"],
                "grounded": True,
                "fallback": False
            }

        hits = self.retrieve(question, language=language)
        qualifying = [hit for hit in hits if hit["similarity"] >= THRESHOLD]
        low_fb = get_low_retrieval_fallback(language, question)
        if not qualifying:
            return {"answer": low_fb, "sources": [], "grounded": False, "fallback": True}

        # Guard Kannada free-text generation until recall improves
        is_kn = any(ord(char) >= 0x0C80 and ord(char) <= 0x0CFF for char in question)
        if language and (language.lower().startswith("kn") or language.lower().startswith("kann")):
            is_kn = True
        if is_kn:
            return {
                "answer": "ಕನ್ನಡ ಮುಕ್ತ ಪ್ರಶ್ನೋತ್ತರ ವ್ಯವಸ್ಥೆಯನ್ನು ನಿಖರತೆ ಸುಧಾರಣೆಗಾಗಿ ತಾತ್ಕಾಲಿಕವಾಗಿ ಸ್ಥಗಿತಗೊಳಿಸಲಾಗಿದೆ. ದಯವಿಟ್ಟು /spray ಪುಟದಲ್ಲಿ ಅಧಿಕೃತ ಸಿಂಪರಣಾ ಸಲಹೆ ಪಡೆಯಿರಿ ಅಥವಾ ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಂಪರ್ಕಿಸಿ.",
                "sources": sorted({hit["source"] for hit in qualifying}),
                "grounded": False,
                "fallback": True,
            }

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {"answer": "Verified context was retrieved, but no Groq API key is configured. " + low_fb,
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
            lower_content = content.lower()
            if any(phrase in lower_content for phrase in [
                "does not contain", "do not contain", "does not provide", "do not provide",
                "does not mention", "do not mention", "no information", "not found in the context"
            ]):
                return {"answer": low_fb, "sources": sorted({hit["source"] for hit in qualifying}),
                        "grounded": False, "fallback": True}
            # Post-generation numeric check: every number in the answer must exist in the context
            if not verify_numeric_grounding(content, context):
                return {"answer": "The generated advisory contained unverified quantities. " + low_fb,
                        "sources": sorted({hit["source"] for hit in qualifying}), "grounded": False, "fallback": True}
            return {"answer": content, "sources": sorted({hit["source"] for hit in qualifying}),
                    "grounded": True, "fallback": False}
        except Exception:
            return {"answer": "The advisory service is unavailable. " + low_fb,
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

        is_yellow, yellow_msg = check_yellow_leaf_disease_intent(question)
        if is_yellow:
            return {
                "answer": yellow_msg,
                "sources": ["Disease Management Strategies in Arecanut and Cocoa (Extension Publication No. 69, 1999, p. 4)"],
                "grounded": True,
                "verified": True,
                "fallback": False
            }

        hits = self.retrieve(question, k=TOP_K, language=language)
        qualifying = [hit for hit in hits if hit["similarity"] >= THRESHOLD]

        if not qualifying:
            fb = get_low_retrieval_fallback(language, question)
            return {
                "answer": fb,
                "sources": [],
                "grounded": False,
                "verified": False,
                "fallback": True,
            }

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

        is_kn = any(ord(char) >= 0x0C80 and ord(char) <= 0x0CFF for char in question)
        if language and (language.lower().startswith("kn") or language.lower().startswith("kann")):
            is_kn = True
        if is_kn:
            return {
                "answer": "ಕನ್ನಡ ಮುಕ್ತ ಪ್ರಶ್ನೋತ್ತರ ವ್ಯವಸ್ಥೆಯನ್ನು ನಿಖರತೆ ಸುಧಾರಣೆಗಾಗಿ ತಾತ್ಕಾಲಿಕವಾಗಿ ಸ್ಥಗಿತಗೊಳಿಸಲಾಗಿದೆ. ದಯವಿಟ್ಟು /spray ಪುಟದಲ್ಲಿ ಅಧಿಕೃತ ಸಿಂಪರಣಾ ಸಲಹೆ ಪಡೆಯಿರಿ ಅಥವಾ ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಂಪರ್ಕಿಸಿ.",
                "sources": sorted({hit["source"] for hit in qualifying}),
                "grounded": False,
                "verified": False,
                "fallback": True,
            }

        api_key = os.getenv("GROQ_API_KEY")
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

        if not api_key:
            return {
                "answer": f"**Verified Information Found ({', '.join(sources)}):**\n\n" + qualifying[0]["text"][:600] + "...",
                "sources": sources,
                "grounded": True,
                "verified": True,
                "fallback": False
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

