import sys
import os
import uuid
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_db import get_storage
from data_loader import embed_texts

DATA_PATH = "/Users/macos/Downloads/1. KLTN/Holland/data.csv"

def textualize_row(row):
    """Biến 1 dòng dữ liệu numeric từ dataset Holland thành văn bản mô tả tính cách."""
    # Các cột R1-R8, I1-I8, ..., C1-C8
    try:
        r_cols = [f'R{i}' for i in range(1, 9)]
        i_cols = [f'I{i}' for i in range(1, 9)]
        a_cols = [f'A{i}' for i in range(1, 9)]
        s_cols = [f'S{i}' for i in range(1, 9)]
        e_cols = [f'E{i}' for i in range(1, 9)]
        c_cols = [f'C{i}' for i in range(1, 9)]
        
        r_score = sum(int(row[c]) for c in r_cols if str(row[c]).isdigit())
        i_score = sum(int(row[c]) for c in i_cols if str(row[c]).isdigit())
        a_score = sum(int(row[c]) for c in a_cols if str(row[c]).isdigit())
        s_score = sum(int(row[c]) for c in s_cols if str(row[c]).isdigit())
        e_score = sum(int(row[c]) for c in e_cols if str(row[c]).isdigit())
        c_score = sum(int(row[c]) for c in c_cols if str(row[c]).isdigit())
        
        major = row.get('major', 'Không rõ')
        country = row.get('country', 'Vùng chưa xác định')
        
        text = (
            f"Hồ sơ Holland (RIASEC) thực tế: R={r_score}, I={i_score}, A={a_score}, S={s_score}, E={e_score}, C={c_score}. "
            f"Sinh viên ngành: {major} tại {country}. "
            f"Các chỉ số phụ: Thích máy móc (R1)={row.get('R1')}, Thích nghiên cứu (I1)={row.get('I1')}, "
            f"Thích nghệ thuật (A1)={row.get('A1')}, Thích giúp đỡ (S1)={row.get('S1')}. "
            "Dữ liệu này giúp Adaptive RAG hiểu rõ các mẫu hình tính cách ngoài đời thực."
        )
        return text
    except Exception as e:
        return f"Dữ liệu thô RIASEC: {str(row)}"

def run_ingestion(limit=1000):
    print(f"Reading {DATA_PATH}...")
    # data.csv is tab-separated with a header
    df = pd.read_csv(DATA_PATH, sep='\t', header=0, nrows=limit)
    
    store = get_storage()
    batch_size = 50 # Smaller batch for stability
    
    for i in tqdm(range(0, len(df), batch_size), desc="Ingesting RIASEC data"):
        batch = df.iloc[i:i+batch_size]
        texts = [textualize_row(row.values) for _, row in batch.iterrows()]
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"holland_data:{i+j}")) for j in range(len(texts))]
        payloads = [
            {"source": "holland_data_145k", "source_tag": "riasec_knowledge", "text": t}
            for t in texts
        ]
        
        vectors = embed_texts(texts)
        store.upsert(ids, vectors, payloads)

if __name__ == "__main__":
    run_ingestion()
    print("Done!")
