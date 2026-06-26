import os
import sys
from collections import Counter

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_db import QdrantStorage
from qdrant_client.http import models

def run_stats():
    storage = QdrantStorage()
    client = storage.client
    collection_name = storage.collection
    
    # 1. Tổng số chunks (points) có tag internal_cv
    count_result = client.count(
        collection_name=collection_name,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source_tag",
                    match=models.MatchValue(value="internal_cv")
                )
            ]
        )
    )
    total_chunks = count_result.count
    
    # 2. Lấy thông tin chi tiết để đếm số lượng CV duy nhất
    all_sources = []
    next_page_offset = None
    
    while True:
        records, next_page_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_tag",
                        match=models.MatchValue(value="internal_cv")
                    )
                ]
            ),
            limit=100,
            offset=next_page_offset,
            with_payload=True,
            with_vectors=False
        )
        
        for rec in records:
            source = rec.payload.get("source", "Unknown")
            all_sources.append(source)
            
        if next_page_offset is None:
            break
            
    unique_cvs = set(all_sources)
    num_unique_cvs = len(unique_cvs)
    
    # 3. Thống kê số chunk trên mỗi CV
    chunks_per_cv = Counter(all_sources)
    avg_chunks = total_chunks / num_unique_cvs if num_unique_cvs > 0 else 0
    
    print("=== THỐNG KÊ MÔ TẢ BỘ DỮ LIỆU INTERNAL_CV ===")
    print(f"1. Tổng số hồ sơ CV duy nhất : {num_unique_cvs}")
    print(f"2. Tổng số đoạn văn bản (Chunks): {total_chunks}")
    print(f"3. Trung bình số đoạn/CV     : {avg_chunks:.2f}")
    print("\n--- Danh sách một số hồ sơ tiêu biểu ---")
    sorted_cvs = sorted(list(unique_cvs))
    for cv in sorted_cvs[:15]:
        print(f"- {cv} ({chunks_per_cv[cv]} chunks)")
    if num_unique_cvs > 15:
        print(f"... và {num_unique_cvs - 15} hồ sơ khác.")

if __name__ == "__main__":
    run_stats()
