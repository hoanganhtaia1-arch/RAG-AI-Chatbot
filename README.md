# Hệ Thống Adaptive Agentic RAG (Tư vấn Du học & Khướng nghiệp)

Hệ thống RAG đa tác nhân thích ứng được thiết kế để cung cấp câu trả lời chính xác, có cơ sở và minh bạch cho các câu hỏi về hướng nghiệp (theo mã Holland/RIASEC) và du học. Hệ thống kết hợp khả năng suy luận của LLM (Ollama) với hạ tầng sự kiện bất đồng bộ (Inngest) và tìm kiếm lai (Hybrid Search).

## 🚀 Các Tính Năng Chính

- **Multi-Agent Architecture**: Tự động định tuyến câu hỏi đến 7 chuyên gia (agent) khác nhau dựa trên ý định của người dùng.
- **Hybrid Search**: Kết hợp tìm kiếm theo từ khóa (BM25) và tìm kiếm ngữ nghĩa (Vector Cosine similarity) trên Qdrant.
- **Dynamic Web Fallback**: Tự động cào dữ liệu từ Internet (Tavily) nếu tài liệu nội bộ không chứa câu trả lời.
- **Semantic Cache**: Phản hồi tức thì (<5ms) cho các câu hỏi trùng lặp phổ biến bằng SQLite.
- **Explainable AI (XAI)**: Cung cấp luồng suy nghĩ (Thinking) và báo cáo độ tin cậy kèm trích dẫn nguồn [n].

## 🛠️ Hướng Dẫn Cài Đặt & Chạy

### 1. Chuẩn bị Môi trường
- Python 3.12+
- Docker (để chạy Qdrant server - tùy chọn)
- Ollama (với model `qwen3.5:0.8b` hoặc tương đương)

### 2. Khởi tạo Cơ sở dữ liệu & Model
```bash
# Docker Qdrant (Nếu dùng server)
docker run -p 6333:6333 qdrant/qdrant

# Kéo model Ollama
ollama pull qwen3.5:0.8b
```

### 3. Cài đặt Phụ thuộc
```bash
python -m venv .venv312
source .venv312/bin/activate  # macOS
pip install -r requirements.txt  # Hoặc dùng uv: uv sync
```

### 4. Chạy Hệ thống (4 Terminal)

1. **Backend API**: `uvicorn main:app --port 8001`
2. **Inngest Worker**: `npx inngest-cli@latest dev -u http://127.0.0.1:8001/api/inngest`
3. **Frontend UI**: `javac HollandRAGGUI.java && java HollandRAGGUI`
4. **Ingest Dữ liệu**: `python scripts/ingest_riasec_data.py` (để nạp tài liệu tham khảo)

## 📊 Đánh Giá (Thesis Evaluation)

Hệ thống được đánh giá qua tập 21 câu hỏi thử nghiệm với 4 chỉ số cốt lõi:
1. **Routing Accuracy** (≥ 0.85): Độ chính xác định tuyến agent.
2. **ROUGE-L** (≥ 0.40): Độ khớp chuỗi con dài nhất.
3. **BERTScore F1** (≥ 0.75): Độ tương đồng ngữ nghĩa.
4. **Citation Accuracy** (≥ 0.80): Độ chính xác của trích dẫn.

Chạy đánh giá live:
```bash
python thesis_eval.py --live --api http://localhost:8001
```

---
*Tài liệu này được biên dịch và tổng hợp từ `OVERVIEW.md`.*

## 📊 Kết quả thử nghiệm (Experimental Results)

Dưới đây là kết quả từ đợt đánh giá gần nhất (21 câu hỏi thử nghiệm):

| Chỉ số | Kết quả | Ngưỡng | Trạng thái |
|---|---|---|---|
| **Routing Accuracy** | 1.00 | ≥ 0.85 | ✅ PASS |
| **ROUGE-L** | 0.058 | ≥ 0.40 | ❌ FAIL |
| **BERTScore F1** | 0.641 | ≥ 0.75 | ❌ FAIL |
| **Citation Accuracy** | 0.00 | ≥ 0.80 | ❌ FAIL |

**Nhận xét:**
- Hệ thống đạt độ chính xác tuyệt đối trong việc định tuyến (Routing).
- Các chỉ số về chất lượng nội dung (ROUGE-L, BERTScore) và trích dẫn (Citation) cần được cải thiện thông qua việc tinh chỉnh prompt và kho dữ liệu.

*Chi tiết xem tại: `thesis_eval_results_final.json`*
