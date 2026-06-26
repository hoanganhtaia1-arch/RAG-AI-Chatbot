import os
import glob
import uuid
import sys
from pathlib import Path

# Add project root to path so we can import internal modules safely
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from tqdm import tqdm

def ingest_all_cvs():
    cv_dir = '/Users/macos/Downloads/1. KLTN/RAG Production/CV'
    cv_files = glob.glob(os.path.join(cv_dir, "*.*"))
    cv_files = [f for f in cv_files if f.endswith(('.pdf', '.docx'))]
    
    store = QdrantStorage()
    
    success_count = 0
    fail_count = 0
    
    print(f"Bắt đầu quá trình nhúng (Ingestion) {len(cv_files)} tệp CV...")
    
    for file_path in tqdm(cv_files):
        source_id = Path(file_path).name
        try:
            # Bước 1: Gọi OCR và chia nhỏ văn bản từ PDF
            chunks_data = load_and_chunk_pdf(file_path)
            if not chunks_data:
                fail_count += 1
                continue
                
            chunks = [item["text"] for item in chunks_data]
            meta = [{"page": item.get("page", 1), "type": item.get("type", "pdf")} for item in chunks_data]
            
            # Bước 2: Nhúng văn bản (Embedding)
            vecs = embed_texts(chunks)
            
            # Bước 3: Tạo payload và ghi xuống Qdrant
            ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"cv:{source_id}:{i}")) for i in range(len(chunks))]
            payloads = [
                {
                    "source": source_id,
                    "source_tag": "admission_profile", # Tag để tách biệt với PDF do user tải lên
                    "role_model_trait": "unknown", # Chúng ta có thể dùng AI chạy lại để thêm meta tag Holland sau này
                    "text": chunks[i],
                    "page": meta[i]["page"],
                    "type": meta[i]["type"],
                }
                for i in range(len(chunks))
            ]
            
            store.upsert(ids, vecs, payloads)
            success_count += 1
            
        except Exception as e:
            print(f"\n[Lỗi] Không thể nạp tệp '{source_id}': {str(e)}")
            fail_count += 1
            
    print(f"\n=== Hoàn thành Ingestion ===")
    print(f"Thành công: {success_count} tài liệu.")
    print(f"Thất bại: {fail_count} tài liệu.")

if __name__ == "__main__":
    ingest_all_cvs()
