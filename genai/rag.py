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
        except Exception:
            return {"answer": "The advisory service is unavailable. " + FALLBACK,
                    "sources": sorted({hit["source"] for hit in qualifying}), "grounded": False, "fallback": True}
        return {"answer": response.choices[0].message.content, "sources": sorted({hit["source"] for hit in qualifying}),
                "grounded": True, "fallback": False}
