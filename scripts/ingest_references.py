import os
import httpx
from pathlib import Path

API_URL = "http://localhost:8001/ingest/pdf"
REF_DIR = Path("/Users/macos/Downloads/1. KLTN/RAG Production/tài liệu tham khảo")

def ingest_all():
    print(f"Checking {REF_DIR}...")
    files = list(REF_DIR.glob("*.pdf")) + list(REF_DIR.glob("*.docx"))
    for fpath in files:
        print(f"Ingesting: {fpath.name}...")
        try:
            with open(fpath, "rb") as f:
                resp = httpx.post(API_URL, files={"file": (fpath.name, f, "application/octet-stream")}, timeout=300)
                print(f"  Response: {resp.status_code} | {resp.text[:100]}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    ingest_all()
