import os
import pandas as pd
from collections import Counter

def analyze_riasec():
    csv_path = "/Users/macos/Downloads/1. KLTN/Holland/data.csv"
    print(f"Đang đọc dữ liệu từ {csv_path}...")
    
    # Đọc file (sử dụng tab làm separator do dữ liệu thô này thường là TSV hoặc CSV đặc thù)
    # Tuy nhiên codebook nói là data.csv, ta thử với comma trước, nếu lỗi thì dùng sep='\t'
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, sep='\t')

    print(f"Tổng số bản ghi ban đầu: {len(df)}")

    # 1. Làm sạch dữ liệu
    # Giữ lại các bản ghi có tuổi >= 13 và <= 80 để tránh nhiễu
    df = df[(df['age'] >= 13) & (df['age'] <= 80)]
    # Bỏ qua các bản ghi không điền chuyên ngành (major)
    df = df.dropna(subset=['major'])
    # Chuyển major về chữ thường để gom nhóm
    df['major'] = df['major'].str.lower().str.strip()
    
    print(f"Số bản ghi sau khi làm sạch: {len(df)}")

    # 2. Tính điểm RIASEC
    # Cấu trúc: 8 câu cho mỗi nhóm
    riasec_cols = {
        'R': [f'R{i}' for i in range(1, 9)],
        'I': [f'I{i}' for i in range(1, 9)],
        'A': [f'A{i}' for i in range(1, 9)],
        'S': [f'S{i}' for i in range(1, 9)],
        'E': [f'E{i}' for i in range(1, 9)],
        'C': [f'C{i}' for i in range(1, 9)]
    }

    for code, cols in riasec_cols.items():
        df[f'Score_{code}'] = df[cols].mean(axis=1)

    # Xác định mã Holland chính (nhóm có điểm cao nhất)
    score_cols = [f'Score_{code}' for code in riasec_cols.keys()]
    df['Primary_Code'] = df[score_cols].idxmax(axis=1).str.replace('Score_', '')

    # 3. Thống kê phân bổ nhóm RIASEC
    distribution = df['Primary_Code'].value_counts()
    total_valid = len(df)

    # 4. Thống kê top Majors theo từng nhóm
    top_majors_by_code = {}
    for code in riasec_cols.keys():
        subset = df[df['Primary_Code'] == code]
        top_majors = subset['major'].value_counts().head(10)
        top_majors_by_code[code] = top_majors

    # 5. Xuất báo cáo Markdown
    report_path = "/Users/macos/.gemini/antigravity/brain/9d5b5c53-08d5-47f3-a2d9-69adfcfe3f3a/riasec_statistical_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO THỐNG KÊ DỮ LIỆU RIASEC (145K MẪU)\n\n")
        f.write(f"**Tổng số mẫu hợp lệ được phân tích:** {total_valid:,} học sinh/người dùng\n\n")
        
        f.write("## 1. Phân bổ nhóm Holland chủ đạo\n")
        f.write("| Nhóm | Số lượng | Tỷ lệ (%) |\n")
        f.write("| :--- | :--- | :--- |\n")
        for code, count in distribution.items():
            perc = (count / total_valid) * 100
            f.write(f"| **{code}** | {count:,} | {perc:.2f}% |\n")
        
        f.write("\n## 2. Các lĩnh vực (Majors) phổ biến theo nhóm RIASEC\n")
        f.write("Dưới đây là Top 10 chuyên ngành phổ biến nhất cho từng nhóm tính cách chủ đạo:\n\n")
        
        for code, majors in top_majors_by_code.items():
            f.write(f"### Nhóm {code}\n")
            f.write("| Hạng | Chuyên ngành | Số lượng |\n")
            f.write("| :--- | :--- | :--- |\n")
            for i, (m, c) in enumerate(majors.items(), 1):
                f.write(f"| {i} | {m.capitalize()} | {c:,} |\n")
            f.write("\n")

    print(f"✅ Đã tạo báo cáo thống kê tại: {report_path}")

if __name__ == "__main__":
    analyze_riasec()
