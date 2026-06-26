import os
import sys
from collections import Counter

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_db import QdrantStorage
from qdrant_client.http import models

def run_riasec_stats():
    storage = QdrantStorage()
    client = storage.client
    collection_name = storage.collection
    
    tags = ["riasec_knowledge", "admission_profile"]
    
    stats = {}
    
    for tag in tags:
        print(f"Đang phân tích tag: {tag}...")
        
        counts = Counter()
        next_page_offset = None
        
        while True:
            records, next_page_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_tag",
                            match=models.MatchValue(value=tag)
                        )
                    ]
                ),
                limit=1000,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )
            
            for rec in records:
                code = rec.payload.get("holland_code", "Unknown")
                # Chuẩn hóa code (ví dụ: 'R', 'RI', v.v.)
                if isinstance(code, str):
                    code = code.upper().strip()
                counts[code] += 1
                
            if next_page_offset is None:
                break
        
        stats[tag] = counts

    print("\n" + "="*50)
    print("THỐNG KÊ CHI TIẾT BỘ DỮ LIỆU RIASEC & ADMISSIONS")
    print("="*50)
    
    for tag, counts in stats.items():
        total = sum(counts.values())
        print(f"\n--- Nguồn: {tag} (Tổng cộng: {total} bản ghi) ---")
        # Sắp xếp theo số lượng giảm dần
        for code, count in counts.most_common():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"- Nhóm {code:8}: {count:6} học sinh ({percentage:5.2f}%)")

if __name__ == "__main__":
    run_riasec_stats()
