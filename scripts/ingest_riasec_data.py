#!/usr/bin/env python3
"""
scripts/ingest_riasec_data.py

Script mẫu để chuyển đổi 145,828 bài test RIASEC và 10 năm hồ sơ nhập học 
thành định dạng RAG (Hành văn hóa - Textualization) và đẩy vào Qdrant.

Cần cài đặt: uv add pandas tqdm
Sử dụng: python scripts/ingest_riasec_data.py
"""

import sys
import os
import uuid
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_db import QdrantStorage
from data_loader import embed_texts

def ingest_145k_riasec_tests(csv_path: str):
    """
    Xử lý file dữ liệu 145,828 bài kiểm tra mã Holland.
    Dữ liệu đầu vào giả định có cột: [id, answers_summary, holland_code, traits]
    """
    print(f"Đang đọc file {csv_path}...")
    try:
        # chunkSize để xử lý file lớn mà không bị hết RAM
        df_chunks = pd.read_csv(csv_path, chunksize=1000)
    except Exception as e:
        print(f"Lỗi đọc file: {e}\n(Đây là script mẫu, vui lòng sửa lại tên cột cho khớp với file thật của bạn)")
        return

    store = QdrantStorage()
    total_ingested = 0

    for chunk in tqdm(df_chunks, desc="Ingesting RIASEC Tests"):
        texts = []
        ids = []
        payloads = []
        
        for _, row in chunk.iterrows():
            # Bước cực kỳ quan trọng: Hành văn hóa (Textualization)
            # Biến dữ liệu bảng thành một case-study hoặc định nghĩa để LLM đọc dễ hiểu.
            holland_code = str(row.get('holland_code', 'Unknown'))
            traits = str(row.get('traits', ''))
            answers = str(row.get('answers_summary', ''))
            
            text_block = (
                f"Đặc điểm bài test đại diện cho mã Holland [{holland_code}]:\n"
                f"Các câu trả lời nổi bật: {answers}.\n"
                f"Tính cách cốt lõi: {traits}."
            )
            
            texts.append(text_block)
            # Dùng UUID chuẩn để track
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"riasec_test:{row.get('id', uuid.uuid4())}"))
            ids.append(doc_id)
            
            payloads.append({
                "source": "riasec_145k_dataset",
                "source_tag": "riasec_knowledge",
                "holland_code": holland_code,
                "text": text_block
            })
            
        # Véc tơ hóa (Embedding)
        vectors = embed_texts(texts)
        # Bắn vào Vector Database Qdrant
        store.upsert(ids, vectors, payloads)
        total_ingested += len(texts)
        
    print(f"✅ Hoàn tất nạp {total_ingested} bài test RIASEC vào Vector DB.")


def ingest_10_year_admission_records(csv_path: str):
    """
    Xử lý file dữ liệu 10 năm hồ sơ nhập học.
    Đầu vào giả lập: [student_id, riasec_code, foundation_ec, professional_ec, personal_ec, admission_result]
    """
    print(f"\nĐang đọc file hồ sơ {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return

    store = QdrantStorage()
    texts, ids, payloads = [], [], []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting Admissions"):
        code = str(row.get('riasec_code', ''))
        result = str(row.get('admission_result', ''))
        
        # Văn bản hóa hồ sơ học sinh
        profile_text = (
            f"Case study hồ sơ thành công: Học sinh hệ {code}, kết quả xét tuyển: {result}.\n"
            f"1. Hoạt động ngoại khóa nền tảng đã tham gia: {row.get('foundation_ec', 'Không có')}.\n"
            f"2. Hoạt động chuyên môn định hướng: {row.get('professional_ec', 'Không có')}.\n"
            f"3. Hoạt động phát triển cá nhân: {row.get('personal_ec', 'Không có')}."
        )
        
        texts.append(profile_text)
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"admission_record:{row.get('student_id', i)}"))
        ids.append(doc_id)
        
        payloads.append({
            "source": "10_year_admissions",
            "source_tag": "admission_profile",
            "holland_code": code,
            "text": profile_text
        })
        
        # Upsert theo batch nhỏ 500 records
        if len(texts) >= 500:
            vectors = embed_texts(texts)
            store.upsert(ids, vectors, payloads)
            texts, ids, payloads = [], [], []

    # Quét tàn dư block cuối
    if texts:
        vectors = embed_texts(texts)
        store.upsert(ids, vectors, payloads)
        
    print(f"✅ Hoàn tất nạp {len(df)} hồ sơ nhập học 10 năm qua.")

if __name__ == "__main__":
    print("="*60)
    print("TỰ ĐỘNG CHUYỂN HOÁ DỮ LIỆU ĐỊNH DẠNG BẢNG SANG VECTOR DB")
    print("="*60)
    
    # Ở đây do tôi chưa có đường dẫn file CSV thực tế của bạn, tôi để tên giả định.
    # Bạn hãy thay đổi tên file tương ứng nhé.
    ingest_145k_riasec_tests("riasec_145828_samples.csv")
    ingest_10_year_admission_records("admissions_10_years.csv")
    
    print("\nHoàn tất! Hệ thống RAG đã được trang bị tri thức mới.")
