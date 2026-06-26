import pandas as pd
import uuid
import sys
import os
from tqdm import tqdm

# Add parent dir to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_db import QdrantStorage
from data_loader import embed_texts

def calculate_holland_code(row):
    """Tính điểm RIASEC từ 48 câu hỏi R1-C8."""
    # Mapping câu hỏi tới nhóm (theo codebook.txt)
    groups = {
        'R': [f'R{i}' for i in range(1, 9)],
        'I': [f'I{i}' for i in range(1, 9)],
        'A': [f'A{i}' for i in range(1, 9)],
        'S': [f'S{i}' for i in range(1, 9)],
        'E': [f'E{i}' for i in range(1, 9)],
        'C': [f'C{i}' for i in range(1, 9)]
    }
    
    scores = {}
    for g, cols in groups.items():
        # Lấy giá trị, ép kiểu int, bỏ qua nếu lỗi
        vals = []
        for c in cols:
            try:
                v = int(row.get(c, 0))
                if 1 <= v <= 5: vals.append(v)
            except: pass
        scores[g] = sum(vals)
    
    # Sắp xếp để lấy mã Holland (ví dụ SAI)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    code = "".join([s[0] for s in sorted_scores[:3]])
    return code, scores

def transform_and_ingest(csv_path, limit=150000):
    print(f"Reading RIASEC data from {csv_path}...")
    # Read with specific columns to save memory if necessary, but 28MB is fine
    df = pd.read_csv(csv_path, sep='\t', low_memory=False)
    if len(df.columns) < 10:
        df = pd.read_csv(csv_path)

    print(f"Total rows found: {len(df)}")
    df = df.head(limit)
    
    # ── Vectorized Holland Code Calculation ──────────────────────────────────────
    groups = {
        'R': [f'R{i}' for i in range(1, 9)],
        'I': [f'I{i}' for i in range(1, 9)],
        'A': [f'A{i}' for i in range(1, 9)],
        'S': [f'S{i}' for i in range(1, 9)],
        'E': [f'E{i}' for i in range(1, 9)],
        'C': [f'C{i}' for i in range(1, 9)]
    }
    
    print("Calculating Holland scores (vectorized)...")
    for g, cols in groups.items():
        # Clean data: convert to numeric, fill NaN with 0, clip to 1-5
        cols_present = [c for c in cols if c in df.columns]
        df[f'score_{g}'] = pd.to_numeric(df[cols_present].stack(), errors='coerce').unstack().fillna(0).sum(axis=1)

    def get_top_3(row):
        scores = {g: row[f'score_{g}'] for g in groups}
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return "".join([s[0] for s in sorted_scores[:3]])

    df['holland_code'] = df.apply(get_top_3, axis=1)
    
    # ── Prepare Metadata ─────────────────────────────────────────────────────────
    df['gender_text'] = df['gender'].map({1: 'Nam', 2: 'Nữ', 3: 'Khác'}).fillna('Không xác định')
    df['major_clean'] = df['major'].fillna('Chưa rõ').astype(str).str.strip()
    
    store = QdrantStorage()
    batch_size = 500  # Increased batch size
    
    print(f"Starting ingestion in batches of {batch_size}...")
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        texts = []
        ids = []
        payloads = []
        
        for idx, row in batch_df.iterrows():
            code = row['holland_code']
            major = row['major_clean']
            text_block = (
                f"Hồ sơ người dùng RIASEC thực tế (Anonymous):\n"
                f"- Mã Holland: {code} (R:{int(row['score_R'])}, I:{int(row['score_I'])}, A:{int(row['score_A'])}, S:{int(row['score_S'])}, E:{int(row['score_E'])}, C:{int(row['score_C'])})\n"
                f"- Nhân khẩu học: {row.get('age', 'N/A')} tuổi, giới tính {row['gender_text']}, quốc gia {row.get('country', 'N/A')}.\n"
                f"- Ngành học/Nghề nghiệp: {major}.\n"
                f"Case study này chứng minh sự tương quan giữa tính cách {code} và lựa chọn ngành {major}."
            )
            
            texts.append(text_block)
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"riasec_holland:{idx}"))
            ids.append(doc_id)
            payloads.append({
                "source": "holland_open_dataset",
                "source_tag": "riasec_knowledge",
                "holland_code": code,
                "text": text_block,
                "major": major
            })
            
        vecs = embed_texts(texts)
        store.upsert(ids, vecs, payloads)
        print(f"Progress: {min(i + batch_size, len(df))}/{len(df)} samples indexed.")

    print(f"✅ Finished indexing {len(df)} RIASEC samples.")

if __name__ == "__main__":
    csv_file = "/Users/macos/Downloads/1. KLTN/Holland/data.csv"
    if os.path.exists(csv_file):
        transform_and_ingest(csv_file, limit=150000)
    else:
        print(f"Error: {csv_file} not found.")
