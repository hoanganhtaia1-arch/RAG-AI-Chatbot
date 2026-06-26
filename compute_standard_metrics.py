"""
compute_standard_metrics_v2.py — Tính Accuracy, Precision, Recall, F1, Response Time
Ánh xạ chỉ số RAG sang chuẩn ML:
  Accuracy  = Routing Accuracy (phân loại đúng ý định)
  Precision = BERTScore F1 (độ chính xác ngữ nghĩa của câu trả lời)
  Recall    = ROUGE-L (độ bao phủ nội dung so với đáp án tham chiếu)
  F1 Score  = 2 * Precision * Recall / (Precision + Recall)
"""
import json

with open("new_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

agentic = data["details"]["Adaptive Agentic RAG"]

scenarios = {
    "S1": "Chẩn đoán Holland/RIASEC",
    "S2": "Tư vấn Ngoại khóa Nền tảng",
    "S3": "Tư vấn Ngoại khóa Chuyên sâu",
    "S4": "Hoạt động Cá nhân",
    "S5": "Đối chiếu Hồ sơ",
    "S6": "Đánh giá Toàn diện",
    "S7": "Hỏi đáp Tổng quát",
}

print("=" * 95)
print(f"{'Kịch bản Test':40s} {'Accuracy':>9s} {'Precision':>10s} {'Recall':>8s} {'F1 Score':>9s}")
print("=" * 95)

all_results = []

for sid, name in scenarios.items():
    items = [d for d in agentic if d["scenario"] == sid]
    n = len(items)

    # Accuracy = Routing Accuracy (trung bình)
    accuracy = sum(d["routing_correct"] for d in items) / n

    # Precision = BERTScore F1 (trung bình)
    precision = sum(d["bertscore_f1"] for d in items) / n

    # Recall = ROUGE-L (trung bình)
    recall = sum(d["rouge_l"] for d in items) / n

    # F1 = Harmonic mean of Precision & Recall
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    result = {
        "scenario": sid, "name": name,
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2)
    }
    all_results.append(result)
    print(f"{sid}. {name:37s} {accuracy:9.2f} {precision:10.2f} {recall:8.2f} {f1:9.2f}")

print("=" * 95)

# Overall
oa = sum(r["accuracy"] for r in all_results) / 7
op = sum(r["precision"] for r in all_results) / 7
orr = sum(r["recall"] for r in all_results) / 7
of1 = sum(r["f1"] for r in all_results) / 7
print(f"{'TRUNG BÌNH':40s} {oa:9.2f} {op:10.2f} {orr:8.2f} {of1:9.2f}")

with open("standard_metrics.json", "w", encoding="utf-8") as f:
    json.dump({"per_scenario": all_results, "overall": {
        "accuracy": round(oa, 2), "precision": round(op, 2),
        "recall": round(orr, 2), "f1": round(of1, 2)
    }}, f, ensure_ascii=False, indent=2)
print(f"\n📁 Đã lưu: standard_metrics.json")
