import json
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup

def extract_pdf(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": idx + 1, "text": text})
    return pages

def extract_html(html_path: Path):
    html = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("div", {"class": "articlebodycontent"}) or soup.find("article") or soup
    return article.get_text(separator="\n", strip=True)

def main():
    print("--- 1. Ext17.pdf ---")
    p1 = Path("genai/corpus/downloads/ext17.pdf")
    pages1 = extract_pdf(p1)
    print(f"Ext17 pages: {len(pages1)}")
    for p in pages1:
        print(f"--- Ext17 Page {p['page']} ---")
        print(p["text"][:1000])

    print("\n--- 2. Disease Management Strategies PDF ---")
    p2 = Path("genai/corpus/downloads/disease_management_strategies_in_arecanut_and_cocoa(eng)_1.pdf")
    pages2 = extract_pdf(p2)
    print(f"Leaflet pages: {len(pages2)}")
    for p in pages2:
        print(f"--- Leaflet Page {p['page']} ---")
        for line in p["text"].split("\n"):
            if any(k in line.lower() for k in ["bordeaux", "koleroga", "rot", "spray", "cover", "polythene", "lime", "copper", "mandipropamid"]):
                print("  ", line)

    print("\n--- 3. The Hindu 2024 Advisory ---")
    p3 = Path("genai/corpus/downloads/thehindu_cpcri_koleroga_advisory_aug2024.html")
    article_text = extract_html(p3)
    print(article_text[:3000])

if __name__ == "__main__":
    main()

