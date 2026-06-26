import os
import uuid
import sys
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path để import các module local
sys.path.append(str(Path(__file__).parent.parent))

from vector_db import get_storage
from data_loader import load_and_chunk_pdf, embed_texts

def main():
    store = get_storage()
    ref_dir = Path("/Users/macos/Downloads/1. KLTN/RAG Production/tài liệu tham khảo")
    
    files = list(ref_dir.glob("*.pdf")) + list(ref_dir.glob("*.docx"))
    print(f"Bắt đầu ingest {len(files)} tài liệu vào local Qdrant...")

    for fpath in files:
        print(f"Processing: {fpath.name}...")
        try:
            chunks_data = load_and_chunk_pdf(str(fpath))
            chunks = [c["text"] for c in chunks_data]
            
            if not chunks:
                print(f"  Warning: No chunks for {fpath.name}")
                continue
                
            vecs = embed_texts(chunks)
            ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{fpath.name}:{i}")) for i in range(len(chunks))]
            payloads = [
                {
                    "source": fpath.name,
                    "text": c["text"],
                    "page": c["page"],
                    "type": c["type"],
                    "source_tag": "pdf" if fpath.suffix == ".pdf" else "docx"
                }
                for c in chunks_data
            ]
            
            store.upsert(ids, vecs, payloads)
            print(f"  Successfully ingested {len(chunks)} chunks.")
            
        except Exception as e:
            print(f"  Error processing {fpath.name}: {e}")

    print("Hoàn tất ingestion!")

if __name__ == "__main__":
    main()
