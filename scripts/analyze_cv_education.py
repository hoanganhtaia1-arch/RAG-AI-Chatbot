"""
analyze_cv_education.py  
Trích xuất trình độ học vấn từ text chunks internal_cv trong Qdrant
bằng keyword matching (University, Bachelor, Master, PhD, High School...).
"""
import os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vector_db import QdrantStorage
from qdrant_client.http import models

# ── Keyword mapping cho trình độ ──────────────────────────────────────────────
EDU_KEYWORDS = {
    'Đại học (Bachelor)': [
        'bachelor', 'undergraduate', 'b.s.', 'b.a.', 'b.eng', 'bba', 'b.sc',
        'đại học', 'cử nhân', 'university', 'college', 'trường đại học',
    ],
    'Thạc sĩ (Master)': [
        'master', 'm.s.', 'm.a.', 'mba', 'm.eng', 'thạc sĩ', 'graduate',
        'postgraduate',
    ],
    'Tiến sĩ (PhD)': [
        'phd', 'ph.d', 'doctor', 'doctorate', 'tiến sĩ',
    ],
    'THPT / Chuẩn bị ĐH': [
        'high school', 'secondary school', 'thpt', 'trung học phổ thông',
        'gap year', 'ib diploma', 'a-level', 'sat', 'act', 'ielts', 'toefl',
    ],
    'Chứng chỉ / Khác': [
        'certificate', 'certification', 'diploma', 'chứng chỉ', 'vocational',
    ],
}

def classify_education(text: str) -> str:
    text_lower = text.lower()
    scores = defaultdict(int)
    for level, kws in EDU_KEYWORDS.items():
        for kw in kws:
            if kw in text_lower:
                scores[level] += 1
    if not any(scores.values()):
        return 'Chưa xác định'
    return max(scores, key=scores.get)

def run():
    storage = QdrantStorage()
    client  = storage.client
    col     = storage.collection

    # Lấy toàn bộ chunks internal_cv
    cv_texts: dict[str, list[str]] = defaultdict(list)
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=col,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="source_tag", match=models.MatchValue(value="internal_cv"))
            ]),
            limit=200, offset=offset, with_payload=True, with_vectors=False
        )
        for r in records:
            cv_texts[r.payload.get("source","?")].append(r.payload.get("text",""))
        if offset is None:
            break

    edu_counter = Counter()
    detail_rows = []

    for cv_file, chunks in cv_texts.items():
        level = classify_education(" ".join(chunks))
        edu_counter[level] += 1
        detail_rows.append((os.path.basename(cv_file), level))

    total = sum(edu_counter.values())

    # ── In terminal ───────────────────────────────────────────────────────────
    print("=== TRÌNH ĐỘ HỌC VẤN – BỘ DỮ LIỆU CV ===")
    for lvl, cnt in edu_counter.most_common():
        print(f"  {lvl}: {cnt} CV  ({cnt/total*100:.1f}%)")

    # Trả về kết quả để gắn vào report
    return edu_counter, detail_rows, total

if __name__ == "__main__":
    run()
