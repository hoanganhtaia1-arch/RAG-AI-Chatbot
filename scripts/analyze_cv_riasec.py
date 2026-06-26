"""
analyze_cv_riasec.py
Đọc toàn bộ chunks internal_cv từ Qdrant, áp dụng keyword matching để:
1. Xác định nhóm RIASEC chủ đạo của từng CV
2. Xác định lĩnh vực chuyên môn (major/field) của từng CV
3. Xuất báo cáo thống kê Markdown
"""
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vector_db import QdrantStorage
from qdrant_client.http import models

# ── Keyword mapping (tiếng Anh + tiếng Việt) ──────────────────────────────────
RIASEC_KEYWORDS = {
    'R': [
        # Realistic – kỹ thuật, thực tế
        'engineer', 'engineering', 'kỹ thuật', 'kỹ sư', 'mechanical', 'electrical',
        'civil', 'construction', 'manufacturing', 'technician', 'robotics',
        'hardware', 'automotive', 'cơ khí', 'điện tử', 'xây dựng', 'machine',
    ],
    'I': [
        # Investigative – nghiên cứu, khoa học
        'research', 'nghiên cứu', 'science', 'biology', 'chemistry', 'physics',
        'data', 'analysis', 'analyst', 'phân tích', 'laboratory', 'experiment',
        'ai', 'machine learning', 'deep learning', 'statistics', 'thống kê',
        'computer science', 'khoa học máy tính', 'software', 'programmer',
    ],
    'A': [
        # Artistic – sáng tạo, nghệ thuật
        'design', 'designer', 'graphic', 'art', 'nghệ thuật', 'music', 'âm nhạc',
        'creative', 'sáng tạo', 'writing', 'viết', 'photography', 'video',
        'content', 'media', 'marketing', 'ux', 'ui', 'animation', 'film',
        'truyền thông', 'báo chí', 'journalism', 'fashion', 'thời trang',
    ],
    'S': [
        # Social – xã hội, hỗ trợ
        'education', 'giáo dục', 'teaching', 'giảng dạy', 'counseling', 'social',
        'volunteer', 'tình nguyện', 'community', 'cộng đồng', 'healthcare',
        'nursing', 'y tế', 'psychology', 'tâm lý', 'human resources', 'hr',
        'nhân sự', 'non-profit', 'phi lợi nhuận', 'communication', 'giao tiếp',
    ],
    'E': [
        # Enterprising – kinh doanh, lãnh đạo
        'business', 'kinh doanh', 'management', 'quản lý', 'entrepreneur',
        'khởi nghiệp', 'startup', 'sales', 'bán hàng', 'finance', 'tài chính',
        'investment', 'đầu tư', 'consulting', 'tư vấn', 'economics', 'kinh tế',
        'leadership', 'lãnh đạo', 'project management', 'ceo', 'director',
    ],
    'C': [
        # Conventional – hành chính, dữ liệu, kế toán
        'accounting', 'kế toán', 'administration', 'hành chính', 'secretary',
        'thư ký', 'banking', 'ngân hàng', 'logistics', 'supply chain',
        'law', 'luật', 'compliance', 'audit', 'kiểm toán', 'clerk',
        'insurance', 'bảo hiểm', 'tax', 'thuế', 'payroll',
    ],
}

# Field keywords → friendly field name
FIELD_KEYWORDS = {
    'Công nghệ thông tin': ['software', 'developer', 'programmer', 'computer science', 'it', 'coding', 'programming', 'ai', 'machine learning', 'data'],
    'Kỹ thuật': ['engineer', 'mechanical', 'electrical', 'civil', 'construction', 'robotics', 'cơ khí', 'kỹ sư'],
    'Kinh doanh & Quản trị': ['business', 'management', 'entrepreneur', 'startup', 'mba', 'kinh doanh', 'quản trị'],
    'Tài chính & Kế toán': ['finance', 'accounting', 'investment', 'banking', 'economics', 'tài chính', 'kế toán'],
    'Marketing & Truyền thông': ['marketing', 'content', 'media', 'communication', 'advertising', 'pr', 'truyền thông', 'brand'],
    'Thiết kế & Nghệ thuật': ['design', 'graphic', 'ux', 'ui', 'art', 'creative', 'animation', 'film', 'photography', 'thiết kế'],
    'Giáo dục': ['education', 'teaching', 'tutor', 'giáo dục', 'giảng dạy'],
    'Y tế': ['healthcare', 'nursing', 'medicine', 'medical', 'y tế', 'y khoa'],
    'Khoa học xã hội': ['psychology', 'social', 'counseling', 'sociology', 'tâm lý', 'xã hội'],
    'Luật': ['law', 'legal', 'lawyer', 'luật'],
    'Khách sạn & Du lịch': ['hospitality', 'hotel', 'tourism', 'du lịch', 'vatel', 'nhà hàng'],
    'Khác': [],
}


def classify_text(text: str) -> tuple[str, str]:
    """Trả về (riasec_code, field_name) dựa trên keyword matching."""
    text_lower = text.lower()

    # Score RIASEC
    riasec_scores = defaultdict(int)
    for code, kws in RIASEC_KEYWORDS.items():
        for kw in kws:
            if kw in text_lower:
                riasec_scores[code] += 1

    dominant = max(riasec_scores, key=riasec_scores.get) if riasec_scores else 'Unknown'

    # Score Field
    field_scores = defaultdict(int)
    for field, kws in FIELD_KEYWORDS.items():
        for kw in kws:
            if kw in text_lower:
                field_scores[field] += 1

    top_field = max(field_scores, key=field_scores.get) if any(field_scores.values()) else 'Khác'

    return dominant, top_field


def run():
    storage = QdrantStorage()
    client = storage.client
    collection_name = storage.collection

    # Lấy toàn bộ chunks internal_cv, gom theo source (file)
    cv_texts: dict[str, list[str]] = defaultdict(list)
    next_page_offset = None

    while True:
        records, next_page_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="source_tag", match=models.MatchValue(value="internal_cv"))]
            ),
            limit=200,
            offset=next_page_offset,
            with_payload=True,
            with_vectors=False,
        )
        for rec in records:
            source = rec.payload.get("source", "Unknown")
            text = rec.payload.get("text", "")
            cv_texts[source].append(text)
        if next_page_offset is None:
            break

    print(f"Số CV duy nhất: {len(cv_texts)}")

    # Phân loại từng CV
    riasec_counter = Counter()
    field_counter = Counter()
    riasec_to_fields: dict[str, Counter] = defaultdict(Counter)

    cv_results = []
    for cv_file, chunks in cv_texts.items():
        combined_text = " ".join(chunks)
        riasec_code, field = classify_text(combined_text)
        riasec_counter[riasec_code] += 1
        field_counter[field] += 1
        riasec_to_fields[riasec_code][field] += 1
        cv_results.append((cv_file, riasec_code, field))

    total = sum(riasec_counter.values())

    # ── Tạo báo cáo Markdown ──────────────────────────────────────────────────
    report_lines = [
        "# THỐNG KÊ RIASEC – BỘ DỮ LIỆU CV (internal_cv)\n",
        f"**Tổng số hồ sơ CV phân tích:** {total}\n",
        "## 1. Phân bổ nhóm RIASEC chủ đạo\n",
        "| Nhóm | Tên đầy đủ | Số CV | Tỷ lệ (%) |",
        "| :--- | :--- | :--- | :--- |",
    ]
    GROUP_NAMES = {'R': 'Realistic (Kỹ thuật – Thực tế)',
                   'I': 'Investigative (Nghiên cứu – Phân tích)',
                   'A': 'Artistic (Nghệ thuật – Sáng tạo)',
                   'S': 'Social (Xã hội – Hỗ trợ)',
                   'E': 'Enterprising (Kinh doanh – Lãnh đạo)',
                   'C': 'Conventional (Hành chính – Quy củ)',
                   'Unknown': 'Chưa phân loại'}
    for code, cnt in riasec_counter.most_common():
        perc = cnt / total * 100
        report_lines.append(f"| **{code}** | {GROUP_NAMES.get(code, '')} | {cnt} | {perc:.1f}% |")

    report_lines.append("\n## 2. Lĩnh vực phổ biến cho từng nhóm RIASEC\n")
    for code, _ in riasec_counter.most_common():
        report_lines.append(f"### Nhóm {code} — {GROUP_NAMES.get(code, '')}")
        report_lines.append("| Lĩnh vực | Số CV |")
        report_lines.append("| :--- | :--- |")
        for field, cnt in riasec_to_fields[code].most_common():
            report_lines.append(f"| {field} | {cnt} |")
        report_lines.append("")

    report_lines.append("## 3. Danh sách chi tiết\n")
    report_lines.append("| Hồ sơ CV | Nhóm RIASEC | Lĩnh vực |")
    report_lines.append("| :--- | :--- | :--- |")
    for cv_file, code, field in sorted(cv_results, key=lambda x: x[1]):
        name = os.path.basename(cv_file)
        report_lines.append(f"| {name} | **{code}** | {field} |")

    report_path = "/Users/macos/.gemini/antigravity/brain/9d5b5c53-08d5-47f3-a2d9-69adfcfe3f3a/cv_riasec_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"✅ Báo cáo đã được lưu tại: {report_path}")

    # In preview ra terminal
    print("\n=== PHÂN BỔ NHÓM RIASEC ===")
    for code, cnt in riasec_counter.most_common():
        print(f"  Nhóm {code}: {cnt} CV  ({cnt/total*100:.1f}%)")


if __name__ == "__main__":
    run()
