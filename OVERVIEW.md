# Tài Liệu Vận Hành Hệ Thống: Adaptive Agentic RAG

Tài liệu này mô tả chi tiết toàn bộ quy trình vận hành, kiến trúc pipeline và các bước khởi chạy hệ thống Advisory RAG (Tư vấn Du học & Khuyến nghị Ngoại khóa). Hệ thống được vận hành bởi mô hình LLM chuyên trách kết hợp cùng kiến trúc Sự kiện Bất đồng bộ (Async Event-Driven) điều phối bởi **Inngest**.

---

## 1. Kiến Trúc Pipeline Vận Hành (Event-Driven Architecture)

Hệ thống được thiết kế tối ưu bằng cách phân rã (de-coupled) để kháng sụp đổ và phân luồng độc lập, tránh tình trạng "thắt cổ chai" khi thao tác nặng. 

### 1.1 Luồng Nạp Dữ Liệu (Ingestion Pipeline)
Xử lý các tác vụ tiền xử lý, OCR và trích xuất vector mà không làm treo ứng dụng:
- **Bóc tách (Parsing):** Hệ thống sử dụng **PyMuPDF4LLM** kết hợp **Tesseract OCR** để đọc hiểu toàn diện nội dung tài liệu, bao gồm cả văn bản thông thường, bảng biểu có cấu trúc và hình ảnh nhúng. PyMuPDF4LLM xuất văn bản theo định dạng Markdown giữ nguyên cấu trúc logic (tiêu đề, bảng, danh sách), giúp các bước xử lý phía sau không mất ngữ cảnh định dạng. Với các tài liệu scan hoặc hình ảnh, Tesseract OCR thực hiện nhận dạng ký tự quang học trước khi chuyển sang bước tiếp theo.
- **Batch Processing:** Sử dụng script chạy độc lập (`scripts/ingest_cvs_batch.py`) để nhồi hàng loạt dữ liệu tĩnh (VD: 93 hồ sơ CV) vào cơ sở dữ liệu Vector `Qdrant`. 
- **On-the-fly Ingestion:** Khi có nghiệp vụ upload tài liệu người dùng trực tiếp trên Web UI, giao diện chỉ báo "Đang xử lý..." đồng thời kích hoạt một Event Task (`rag/ingest_pdf`) lên hệ thống trung tâm `Inngest`. Worker nền sẽ độc lập thực hiện quy trình OCR, chia khối chunking và cấu thành tập Vector, bảo vệ luồng tương tác Chat độc lập.

### 1.2 Luồng Truy Vấn & Trả Kết Quả (Execution Query Pipeline)
Đây là quy trình khép kín chu kỳ mili-giây diễn ra mỗi khi End-User nhập câu hỏi:
1. **Semantic Cache (Bước -1):** Nhúng câu hỏi về Vector (Cosine $\geq$ 0.92) để tìm trong lịch sử Cache (SQLite). Trả kết quả và ngắt luồng dưới 5ms nếu tìm thấy sự trùng lặp phổ biến.
2. **Multi-Agent Router (Bước 0):** Zero-shot Classification của LLM phân tích từ khóa và mục tiêu của câu hỏi (Hỏi Visa, Học bổng hay Hoạt động ngoại khóa...) nhằm kích hoạt cấu hình Prompt chuyên trách của Agent tương ứng.
3. **Hybrid Search & Reranking (Bước 1):** Tiến hành Search song song `BM25 (Keyword)` và `Vector (Semantic)` bên trong lưới Qdrant. Sau đó, công thức lai RRF được áp dụng và chấm điểm Cross-Encoder (Rerank model) tiến hành loại bỏ văn bản rác, giữ lại Top-K tinh túy nhất.
4. **Dynamic Web Fallback (Bước 2):** Nền tảng dự phòng. Nếu file PDF không có sẵn câu trả lời, crawler kích hoạt quét khẩn API Internet (Tavily), chọn lọc tên miền uy tín (`edu`, `gov`) và trích thông tin bù để tránh lỗi thời.
5. **LLM-as-a-Judge (Bước 3):** Mô hình thẩm định độc lập đứng đánh giá văn bản trước khi đưa vào phễu tổng, loại trừ các đoạn có đánh giá $\approx 2/5$ nhằm ngừa "ảo giác (hallucinations)".
6. **Streaming Generation (Bước 4 & 5):** Chuyển toàn bộ 10 lịch sử Context kèm bộ lệnh lên LLM Engine. Câu trả lời được tuôn ra giao diện Java Desktop GUI qua Server-Sent Events (SSE). 
7. **Cache Registration (Bước 6):** Cập nhật dữ liệu vào SQLite chờ phục vụ đợt truy xuất tiếp sau.

---

## 2. Ưu Điểm Đột Phá Khác Biệt (Why this Architecture?)
1. **Chống sụp đổ (Fault Tolerant):** Đạt được nhờ việc bẻ khóa Step-by-step qua nền tảng hàng đợi `Inngest`. LLM chạy quá giờ (Timeout) hoặc Cào Web thất bại đều có thể giới hạn tỷ lệ (Rate-Limit) và "Thử lại (Retry)" mà không kéo chìm Server chính.
2. **Chống ảo giác & Lỗi thời:** Mạng dự phòng Web Crawler lấp đầy ngay khuyết điểm kiến thức của tài liệu tĩnh. Lớp khiên LLM-Judge ngăn ngừa tin giả triệt để.
3. **Hiệu suất kinh tế phần cứng:** Tiết kiệm hàng ngàn token thay vì dồn thông tin một cách mù quáng vào Prompt vì đã có Router và Cache chặn ngang.

---

## 3. Hướng Dẫn Kích Hoạt Hệ Thống (Local / Production)

Quy chuẩn khởi chạy cần tuần thủ thứ tự đánh cắm từng Service một. Khuyến nghị thao tác bên trong bộ môi trường ảo đã khóa (`.venv312` hoặc môi trường quản lý gói `uv`).

Các Core-Engine tính toán thấp nhất phải luôn có mặt đầu tiên:
1. **OpenAI API / LLM Engine**: Đảm bảo kết nối Internet và API key hợp lệ trong `.env`.
2. **Qdrant DB**: Port `6333` - Cơ sở dữ liệu Vector Search chạy Docker/hoặc file local cục bộ.

*(Thủ thuật tùy chọn)* Nạp luồng tài liệu kho thư viện rỗng:
```bash
uv run python scripts/ingest_cvs_batch.py
```

### 3.2 Khởi chạy Lõi Backend RESTful API (Terminal 2)
Cầu nối trung chuyển của mọi sự tương tác.
```bash
uv run uvicorn main:app --reload
```
API kết nối mặc định tại `http://127.0.0.1:8000`.

### 3.3 Khởi chạy Phễu Điều Phối Inngest Queue (Terminal 3)
Inngest giữ nhịp đập các tác vụ Ingestion dài hơi (ví dụ: upload PDF trên Web UI hay Async Task):
```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

### 3.4 Bật Giao Diện Khách Hàng - Java Desktop GUI (Terminal 4)
Cửa sổ cuối cùng tương tác với người dùng (viết bằng Java Swing).
```bash
javac HollandRAGGUI.java
java HollandRAGGUI
```
*(Lưu ý: Mở ứng dụng Desktop để sử dụng Front-end cùng các thẻ kiểm soát độ tin cậy XAI).*

---

## 4. Độ Đo Đánh Giá Hiệu Quả Mô Hình

Để đánh giá toàn diện chất lượng hệ thống RAG đa tác nhân, đề tài thiết kế quy trình thực nghiệm theo hai cấp độ nhằm đảm bảo tính đúng đắn cả về mặt kỹ thuật lẫn chất lượng ngôn ngữ sinh ra.

**Cấp độ kỹ thuật** kiểm tra tính đúng đắn của các API endpoint bằng bộ kiểm thử tích hợp (integration test) sử dụng kiểm tra kiểu dữ liệu pydantic. **Cấp độ chất lượng RAG** đánh giá thông qua tập 21 câu hỏi kiểm thử (7 kịch bản × 3 dạng: chuẩn, biến thể và tình huống biên) tương ứng với 7 agent chuyên trách trong hệ thống.

### 4.1 Bốn Chỉ Số Đánh Giá Chính

| # | Chỉ số | Ngưỡng chấp nhận | Cấp độ |
|---|---|---|---|
| 1 | Routing Accuracy | ≥ 0,85 | Điều phối |
| 2 | ROUGE-L | ≥ 0,40 | Sinh ngôn ngữ |
| 3 | BERTScore F1 | ≥ 0,75 | Sinh ngôn ngữ |
| 4 | Citation Accuracy | ≥ 0,80 | Sinh ngôn ngữ |

---

#### 4.1.1 Routing Accuracy

**Routing Accuracy** đo tỉ lệ câu hỏi được hệ thống điều phối đúng đến tác nhân chuyên trách tương ứng. Đây là chỉ số then chốt trong kiến trúc đa tác nhân, phản ánh khả năng phân loại ý định của bộ định tuyến LLM Zero-shot.

$$\text{Routing Accuracy} = \frac{\text{Số câu hỏi được định tuyến đúng agent}}{\text{Tổng số câu hỏi}}$$

- **Ngưỡng chấp nhận:** ≥ 0,85
- **Ý nghĩa:** Nếu router định tuyến sai, các bước phía sau (retrieval, generation) sẽ dùng sai prompt và sai kho dữ liệu, khiến mọi chỉ số khác vô nghĩa.

---

#### 4.1.2 ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)

**ROUGE-L** đo độ khớp giữa câu trả lời được sinh ra và câu trả lời tham chiếu dựa trên **chuỗi con chung dài nhất** (Longest Common Subsequence – LCS) ở cấp độ token.

$$\text{ROUGE-L} = \frac{2 \times P_{LCS} \times R_{LCS}}{P_{LCS} + R_{LCS}}$$

Trong đó:
- $P_{LCS} = \frac{|LCS(hypothesis, reference)|}{|hypothesis|}$ — độ chính xác
- $R_{LCS} = \frac{|LCS(hypothesis, reference)|}{|reference|}$ — độ bao phủ

- **Ngưỡng chấp nhận:** ≥ 0,40
- **Ý nghĩa:** Phản ánh khả năng bảo toàn nội dung thông tin so với chuẩn vàng ở cấp độ từ vựng, không yêu cầu khớp thứ tự tuyệt đối.

---

#### 4.1.3 BERTScore F1

**BERTScore F1** sử dụng mô hình ngôn ngữ `bert-base-multilingual-cased` để tính độ tương đồng ngữ nghĩa giữa câu trả lời sinh ra và câu tham chiếu thông qua so sánh các biểu diễn vector theo cơ chế attention.

$$\text{BERTScore F1} = \frac{2 \times P_{BERT} \times R_{BERT}}{P_{BERT} + R_{BERT}}$$

Trong đó $P_{BERT}$ và $R_{BERT}$ được tính bằng cosine similarity giữa từng token embedding của hypothesis và reference (greedy matching).

- **Ngưỡng chấp nhận:** ≥ 0,75
- **Ý nghĩa:** Khác với ROUGE-L chỉ đối sánh từ vựng, BERTScore nắm bắt được sự tương đồng ở cấp độ nghĩa — phù hợp khi hệ thống diễn đạt cùng nội dung bằng từ ngữ khác nhau.
- **Model sử dụng:** `bert-base-multilingual-cased` (hỗ trợ tiếng Việt)

---

#### 4.1.4 Citation Accuracy

**Citation Accuracy** đo tỉ lệ câu trả lời có chứa ít nhất một trích dẫn nguồn hợp lệ theo định dạng `[n]`, trong đó `n` nằm trong phạm vi các nguồn được truy xuất.

$$\text{Citation Accuracy} = \frac{\text{Số câu trả lời có trích dẫn hợp lệ}}{\text{Tổng số câu trả lời}}$$

- **Ngưỡng chấp nhận:** ≥ 0,80
- **Ý nghĩa:** Trong hệ thống RAG, việc trích dẫn nguồn là yêu cầu bắt buộc nhằm đảm bảo tính minh bạch và khả năng kiểm chứng thông tin, đặc biệt quan trọng khi tư vấn hướng nghiệp cho học sinh.

---



### 4.2 Thiết Kế Tập Kiểm Thử

Tập kiểm thử được xây dựng theo nguyên tắc **bao phủ toàn diện chức năng hệ thống**, đảm bảo mỗi agent chuyên trách đều được kiểm tra ở nhiều dạng đầu vào khác nhau. Cấu trúc ma trận **7 kịch bản × 3 dạng câu hỏi** tạo ra **21 trường hợp kiểm thử**, phủ cả trường hợp lý tưởng lẫn các tình huống thách thức khả năng điều phối và sinh ngôn ngữ của mô hình.

#### Ma Trận Kịch Bản

| Kịch bản | Agent mục tiêu | Nguồn dữ liệu | Mục đích kiểm thử |
|---|---|---|---|
| S1 | `test_riasec` | Dataset RIASEC 145k | Kiểm tra khả năng chẩn đoán Holland từ mô tả tính cách, sở thích |
| S2 | `hdnk_nentang` | Hồ sơ nhập học 10 năm | Kiểm tra đề xuất ngoại khóa nền tảng (kỹ năng mềm, giao tiếp) |
| S3 | `hdnk_chuyenmon` | Hồ sơ nhập học 10 năm | Kiểm tra đề xuất ngoại khóa chuyên sâu (nghiên cứu, thực tập) |
| S4 | `hdnk_canhan` | Toàn bộ kho dữ liệu | Kiểm tra đề xuất hoạt động phản ánh đúng màu sắc cá nhân |
| S5 | `match_hoso` | Hồ sơ nhập học 10 năm | Đối chiếu hồ sơ tương đồng (RIASEC, lĩnh vực) + Mô tả lộ trình |
| S6 | `danhgia_all` | Toàn bộ kho dữ liệu | Kiểm tra đánh giá tổng hợp: chẩn đoán RIASEC + thiết kế lộ trình 3 cấp |
| S7 | `hoidapchung` | Toàn bộ kho dữ liệu | Kiểm tra xử lý câu hỏi tổng quát, ngoài phạm vi chuyên biệt |

#### Ba Dạng Câu Hỏi

Mỗi kịch bản được triển khai theo 3 dạng câu hỏi để kiểm tra độ bền của hệ thống dưới các điều kiện đầu vào khác nhau:

| Dạng | Đặc điểm | Mục tiêu đánh giá |
|---|---|---|
| **Chuẩn (Standard)** | Câu hỏi rõ ràng, thông tin đầy đủ, đúng ngữ cảnh nghiệp vụ | Đo hiệu năng nền (baseline) khi đầu vào lý tưởng |
| **Biến thể (Variant)** | Cùng ý định nhưng diễn đạt bằng từ ngữ khác, góc nhìn khác | Kiểm tra tính bất biến ngữ nghĩa của router và mô hình sinh |
| **Tình huống biên (Edge)** | Câu hỏi mơ hồ, thiếu thông tin, ngoài phạm vi, hoặc gây nhiễu | Kiểm tra khả năng degradation graceful và fallback hợp lý |

#### Ví Dụ Minh Họa (Kịch bản S1 — `test_riasec`)

| Dạng | Câu hỏi ví dụ |
|---|---|
| Chuẩn | *"Em hay thích tháo lắp máy móc và làm thủ công. Em thuộc nhóm Holland nào?"* |
| Biến thể | *"Tính cách tôi: thích giải quyết vấn đề thực tế, không thích nói nhiều, làm tốt hơn nói. RIASEC là gì?"* |
| Tình huống biên | *"Em không biết mình thích gì, nhưng không ghét môn nào cả. Làm sao xác định Holland?"* |

#### Căn Cứ Lựa Chọn Ngưỡng

Các ngưỡng chấp nhận được xác định dựa trên tổng hợp từ các công trình liên quan trong lĩnh vực đánh giá hệ thống hỏi đáp và RAG:
- **Routing Accuracy ≥ 0,85**: Ngưỡng tối thiểu để hệ thống đa agent vận hành đúng nghiệp vụ — tương đương với yêu cầu của các hệ thống phân loại ý định thực tế (intent classification).
- **ROUGE-L ≥ 0,40**: Ngưỡng trung bình chấp nhận được cho bài toán sinh văn bản tiếng Việt không yêu cầu khớp hoàn toàn từ vựng.
- **BERTScore F1 ≥ 0,75**: Mức tương đồng ngữ nghĩa đủ cao để đảm bảo câu trả lời không lệch nghĩa so với chuẩn vàng.
- **Citation Accuracy ≥ 0,80**: Phản ánh yêu cầu nghiêm ngặt về tính minh bạch trong tư vấn hướng nghiệp.

### 4.3 Chạy Đánh Giá

```bash
# Offline — chỉ đánh giá Routing Accuracy (không cần server)
uv run python thesis_eval.py

# Live — đánh giá đầy đủ 4 chỉ số (cần FastAPI server đang chạy)
uv run python thesis_eval.py --live --output thesis_eval_results.json

# Với tập dữ liệu tùy chỉnh
uv run python thesis_eval.py --file my_eval.json

# Đánh giá bổ sung bằng RAGAS (faithfulness, answer_relevancy, context_precision)
uv run python ragas_eval.py --file eval.json
```

> **Lưu ý:** BERTScore yêu cầu `bert-score` và `torch` (`uv add bert-score torch`). Routing Accuracy yêu cầu kết nối LLM được cấu hình trong `.env`.

