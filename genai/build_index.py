from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from genai.rag import MODEL_NAME, chunk_text

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
OUT = ROOT / "index"


def extract(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for node in soup(["script", "style", "noscript"]):
        node.decompose()
    return soup.get_text(" ", strip=True)


def main() -> None:
    CORPUS.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    records, chunks = [], []
    for number, item in enumerate(manifest, 1):
        suffix = ".pdf" if item["url"].lower().split("?")[0].endswith(".pdf") else ".html"
        path = CORPUS / f"{number:02d}{suffix}"
        response = requests.get(item["url"], timeout=60, headers={"User-Agent": "Krishi-Sahayak-research/1.0"})
        response.raise_for_status()
        path.write_bytes(response.content)
        text = extract(path)
        if len(text) < 200:
            raise ValueError(f"Downloaded source has too little extractable text: {item['url']}")
        records.append({**item, "local_file": path.name, "characters": len(text)})
        for index, text_chunk in enumerate(chunk_text(text)):
            chunks.append({"id": f"{number:02d}-{index:04d}", "source": item["title"], "url": item["url"], "text": text_chunk})
    embedder = SentenceTransformer(MODEL_NAME)
    vectors = embedder.encode([item["text"] for item in chunks], normalize_embeddings=True, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    OUT.mkdir(exist_ok=True)
    faiss.write_index(index, str(OUT / "faiss.index"))
    (OUT / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "download_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"indexed_sources={len(records)} chunks={len(chunks)}")


if __name__ == "__main__":
    main()
