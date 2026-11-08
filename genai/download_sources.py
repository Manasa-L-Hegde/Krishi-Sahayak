import hashlib
import json
import os
import sys
from pathlib import Path
import urllib.request
import ssl

SOURCES = [
    {
        "id": "ext17_pdf",
        "url": "https://cpcri.gov.in/filemgr/webfs/publication/ext17.pdf",
        "filename": "ext17.pdf",
        "title": "Koleroga of Arecanut (Extension Folder No. 17)",
        "publisher": "ICAR - Central Plantation Crops Research Institute (CPCRI)",
    },
    {
        "id": "cpcri_vittal_leaflet_pdf",
        "url": "https://cpcri.gov.in/filemgr/webfs/publication/disease_management_strategies_in_arecanut_and_cocoa(eng)_1.pdf",
        "filename": "disease_management_strategies_in_arecanut_and_cocoa(eng)_1.pdf",
        "title": "Disease Management Strategies in Arecanut and Cocoa",
        "publisher": "ICAR - Central Plantation Crops Research Institute (CPCRI), Regional Station Vittal",
    },
    {
        "id": "cpcri_advisory_2024_thehindu",
        "url": "https://www.thehindu.com/news/cities/Mangalore/cpcri-issues-advisory-to-control-koleroga-disease-in-arecanut-plantations/article68506079.ece",
        "filename": "thehindu_cpcri_koleroga_advisory_aug2024.html",
        "title": "CPCRI issues advisory to control koleroga disease in arecanut plantations (The Hindu, August 9, 2024)",
        "publisher": "The Hindu / ICAR-CPCRI",
    }
]

def download_all():
    dest_dir = Path("genai/corpus/downloads")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    manifest = []

    for src in SOURCES:
        dest_path = dest_dir / src["filename"]
        print(f"Downloading {src['url']} -> {dest_path}...")
        req = urllib.request.Request(
            src["url"],
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                content = resp.read()
                dest_path.write_bytes(content)
                sha256 = hashlib.sha256(content).hexdigest()
                size = len(content)
                print(f"  Success: {size} bytes, SHA-256: {sha256}")
                manifest.append({
                    "id": src["id"],
                    "title": src["title"],
                    "url": src["url"],
                    "filename": src["filename"],
                    "local_path": str(dest_path.as_posix()),
                    "sha256": sha256,
                    "bytes": size,
                    "publisher": src["publisher"],
                })
        except Exception as e:
            print(f"  Error downloading {src['url']}: {e}", file=sys.stderr)
            manifest.append({
                "id": src["id"],
                "url": src["url"],
                "error": str(e)
            })

    manifest_path = Path("genai/download_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")

if __name__ == "__main__":
    download_all()
