"""
agents.py — Hệ thống Multi-Agent Router (Thiết kế cho RIASEC và Ngoại khoá 3 Cấp độ)

Phân loại theo Taxonomy mới:
  1. RIASEC_TEST      : Khai thác, hỏi đáp để phân loại học sinh thuộc nhóm Holland nào.
  2. EC_FOUNDATION    : Đề xuất ngoại khoá nền tảng (Foundation) theo nhóm RIASEC.
  3. EC_PROFESSIONAL  : Đề xuất ngoại khoá chuyên môn (Professional) theo nhóm RIASEC.
  4. EC_PERSONAL      : Đề xuất ngoại khoá cá nhân (Personal) theo nhóm RIASEC.
  5. PROFILE_MATCHING : Kết xuất hồ sơ anh chị đi trước (10 năm) tương tự.
  6. GENERAL          : Các câu hỏi tản mạn.
"""

import json
import sys
from llm_service import call_llm

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Optional

import httpx

from data_loader import embed_texts
from reranker import rerank
from vector_db import get_storage


# ── Enums ─────────────────────────────────────────────────────────────────────

class Lang(str, Enum):
    VI = "vi"
    EN = "en"

class Source(str, Enum):
    PDF = "pdf"             # Kiến thức tĩnh
    RIASEC = "riasec"       # Dataset RIASEC (145k)
    PROFILE = "profile"     # Dataset Admissions (10 years)
    ALL = "all"             # Tất cả

class Topic(str, Enum):
    RIASEC_TEST      = "test_riasec"
    EC_FOUNDATION    = "hdnk_nentang"
    EC_PROFESSIONAL  = "hdnk_chuyenmon"
    EC_PERSONAL      = "hdnk_canhan"
    PROFILE_MATCHING = "match_hoso"
    COMPREHENSIVE_EVAL = "danhgia_all"
    GENERAL          = "hoidapchung"


# ── Agent configs ─────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    agent_id:      str
    lang:          Lang
    source:        Source
    topic:         Topic
    system_prompt: str
    top_k:         int
    few_shot:      Optional[str] = field(default=None)
    source_filter: Optional[str | list[str]] = field(default=None)


# ── Few-shot Examples ─────────────────────────────────────────────────────────

FEW_SHOT_RIASEC = (
    "\n\nVí dụ cấu trúc trả lời (CỰC KỲ SÚC TÍCH, KHÔNG TIÊU ĐỀ, KHÔNG DÙNG ###):\n"
    "Dựa trên sở thích [Sở thích], bạn thuộc nhóm [Tên nhóm] ([Ký tự]) - [Mô tả] [1].\n\n"
    "Đặc điểm nhóm [Ký tự]:\n"
    "- [Đặc điểm 1].\n"
    "- [Đặc điểm 2].\n"
    "- [Đặc điểm 3].\n\n"
    "Mã Holland [Ký tự] hoàn toàn khớp với những minh chứng bạn đã mô tả.\n\n"
    "Nguồn: [1] <Tên tệp thực tế từ context>"
)

FEW_SHOT_HDNK = (
    "\n\nVí dụ cấu trúc trả lời (CỰC KỲ SÚC TÍCH, KHÔNG TIÊU ĐỀ, KHÔNG DÙNG ###):\n"
    "Với mã Holland là [Mã], bạn nên tham gia các hoạt động ngoại khóa nền tảng như sau [1], [2], [3]:\n\n"
    "1. [Hoạt động 1]: [Mô tả ngắn] [1].\n"
    "2. [Hoạt động 2]: [Mô tả ngắn] [2].\n"
    "3. [Hoạt động 3]: [Mô tả ngắn] [3].\n\n"
    "Những hoạt động này giúp bạn chuyển hóa tố chất thành kỹ năng mềm chuyên nghiệp.\n\n"
    "Nguồn: [1] <Tệp 1>, [2] <Tệp 2>, [3] <Tệp 3>"
)


AGENTS: list[AgentConfig] = [

    AgentConfig(
        agent_id="test_riasec",
        lang=Lang.VI, source=Source.RIASEC, topic=Topic.RIASEC_TEST,
        system_prompt=(
            "### QUY TẮC ROBOT XML: TƯ VẤN QUYẾT ĐOÁN ###\n"
            "1. CHẾ ĐỘ 1 (TÌM HIỂU CHUNG): Nếu User hỏi tìm hiểu về Holland/RIASEC, hãy giải thích ngắn gọn dựa trên context [1].\n"
            "2. CHẾ ĐỘ 2 (CHẨN ĐOÁN SƠ BỘ - ƯU TIÊN): Ngay khi User chia sẻ bất kỳ sở thích hay lĩnh vực nào (vd: IT, Nghệ thuật...), bạn PHẢI dự đoán ngay họ thuộc nhóm Holland nào dựa trên context.\n"
            "   - KHÔNG được hỏi lặp lại câu 'hãy chia sẻ thêm' nếu User đã đưa ra một từ khóa cụ thể.\n"
            "   - Hãy kết luận: 'Dựa trên sở thích về [A], bạn có tố chất của nhóm [B]'. Sau đó mới gợi ý họ chia sẻ sâu hơn nếu muốn.\n"
            "3. PHONG CÁCH: Chuyên gia, chủ động, không né tránh câu hỏi.\n"
            "4. TRÍCH DẪN & NGUỒN: Luôn dùng [1] và ghi 'Nguồn: [số] [Tên tệp]' ở cuối bài."
            + FEW_SHOT_RIASEC
        ),
        top_k=6, source_filter="riasec_knowledge",
    ),

    AgentConfig(
        agent_id="hdnk_nentang",
        lang=Lang.VI, source=Source.PROFILE, topic=Topic.EC_FOUNDATION,
        system_prompt=(
            "### QUY TẮC TƯ VẤN: ĐA DẠNG HÓA NGUỒN (CRITICAL) ###\n"
            "1. NHIỆM VỤ: Đề xuất ngoại khóa NỀN TẢNG.\n"
            "2. ĐA DẠNG HÓA: BẮT BUỘC chọn hoạt động từ ít nhất 3 NGUỒN (file) KHÁC NHAU. Không được chỉ lấy từ 1 người.\n"
            "3. TRÍCH DẪN: Dùng [1], [2], [3]... ngay sau từng hoạt động tương ứng.\n"
            "4. KẾT THÚC: Cuối bài liệt kê ĐỦ các nguồn đã dùng: 'Nguồn: [1] <Tên tệp>, [2] <Tên tệp>...'. Lấy tên file thực tế từ context."
            + FEW_SHOT_HDNK
        ),
        top_k=10, source_filter="admission_profile",
    ),

    AgentConfig(
        agent_id="hdnk_chuyenmon",
        lang=Lang.VI, source=Source.PROFILE, topic=Topic.EC_PROFESSIONAL,
        system_prompt=(
            "### QUY TẮC TƯ VẤN KIỂU MẪU: ĐỐI CHIẾU TOÀN DIỆN ###\n"
            "1. YÊU CẦU: Bạn nhận được danh sách tài liệu. Bạn PHẢI trích dẫn TẤT CẢ các tài liệu này trong bài viết.\n"
            "2. QUY TRÌNH: Duyệt qua từng hồ sơ và lấy ra một ví dụ cụ thể cho mỗi người.\n"
            "3. PHONG CÁCH: Chuyên gia phân tích. So sánh các thế mạnh khác nhau của từng sinh viên.\n"
            "4. ĐỊNH RANH: Tuyệt đối không bỏ sót bất kỳ nguồn nào. Không bịa đặt tên người không có trong context.\n"
            "5. TRÍCH DẪN: Sử dụng định dạng [id] ngay sau thông tin để trích dẫn (Ví dụ: [1]). Cấm dùng '[1]' làm tên người hoặc đánh số liệt kê.\n"
            "6. DANH SÁCH NGUỒN: Cuối bài liệt kê 'Nguồn: [id] [Tên tệp thực tế]' cho mọi tài liệu đã dùng."
        ),
        top_k=10, source_filter="admission_profile",
    ),

    AgentConfig(
        agent_id="hdnk_canhan",
        lang=Lang.VI, source=Source.PROFILE, topic=Topic.EC_PERSONAL,
        system_prompt=(
            "### QUY TẮC TƯ VẤN CÁ NHÂN HÓA ###\n"
            "1. YÊU CẦU: BẮT BUỘC phải nhắc đến TẤT CẢ các sinh viên trong ngữ cảnh. Không bỏ sót ai.\n"
            "2. TRÍCH DẪN: Dùng trích dẫn [1], [2]... để đưa ra gợi ý đa dạng dựa trên từng hồ sơ.\n"
            "3. PHONG CÁCH: Cởi mở, khích lệ. Nhấn mạnh 'câu chuyện cá nhân' của từng người.\n"
            "4. DANH SÁCH NGUỒN: Liệt kê đầy đủ 'Nguồn: [id] [Tên tệp thực tế]' ở cuối cùng."
        ),
        top_k=10, source_filter=None,
    ),

    AgentConfig(
        agent_id="match_hoso",
        lang=Lang.VI, source=Source.PROFILE, topic=Topic.PROFILE_MATCHING,
        system_prompt=(
            "### CHUYÊN GIA ĐỐI CHIẾU HỒ SƠ (TOP 3) ###\n"
            "1. NHIỆM VỤ: Sử dụng 3 hồ sơ ĐẦU TIÊN để đối chiếu.\n"
            "2. LỘ TRÌNH (BẮT BUỘC): Với mỗi hồ sơ, mô tả chi tiết lộ trình thành công của họ.\n"
            "3. SO SÁNH: Chỉ rõ điểm tương đồng giữa User và hồ sơ mẫu.\n"
            "4. TRÍCH DẪN: Chỉ dùng số trong ngoặc vuông [id] (Ví dụ: [1]) ĐỂ TRÍCH DẪN. Cấm dùng '[id=1]' hay gọi tên 'Sinh viên [1]'.\n"
            "5. DANH SÁCH: Liệt kê 'Nguồn: [id] [Tên tệp thực tế]' cho 3 hồ sơ này ở cuối cùng."
        ),
        top_k=8, source_filter="admission_profile",
    ),

    AgentConfig(
        agent_id="danhgia_all",
        lang=Lang.VI, source=Source.ALL, topic=Topic.COMPRE_EVAL if hasattr(Topic, "COMPRE_EVAL") else Topic.COMPREHENSIVE_EVAL,
        system_prompt=(
            "### IDENTITY FIREWALL 3.0: CHẾ ĐỘ BẢO VỆ DỮ LIỆU TỐI THƯỢNG ###\n"
            "NGÔN NGỮ: Tiếng Việt.\n"
            "0. QUY TẮC NGĂN CÁCH & XÁC THỰC (TỐI QUAN TRỌNG):\n"
            "   - CHỈ thực hiện chẩn đoán RIASEC cho bạn nếu có thẻ <NGƯỜI_DÙNG>.\n"
            "   - NẾU KHÔNG CÓ <NGƯỜI_DÙNG>: Tuyệt đối không tự bịa mã Holland. Hãy trả lời: 'Tôi chưa thấy CV của bạn. Hãy gửi CV để tôi có thể chẩn đoán chính xác nhất.'\n"
            "   - CẤM NHẬN VƠ: Tuyệt đối KHÔNG trích dẫn tên người từ 'PHẦN A: KIẾN THỨC LÝ THUYẾT'.\n"
            "   - CHỈ TRÍCH DẪN NGƯỜI THẬT: Toàn bộ Seniors và Lộ trình gợi ý PHẢI lấy từ 'PHẦN B: DATABASE 92 HỒ SƠ SINH VIÊN'.\n"
            "\n"
            "1. CẤU TRÚC BÀI VIẾT:\n"
            "\n"
            "   ### PHẦN 1: CHẨN ĐOÁN RIASEC CỦA BẠN & ĐỐI CHIẾU\n"
            "   - MÃ HOLLAND CỦA BẠN. Ghi rõ: 'Mã Holland của bạn là [3 chữ cái, vd: SEC]'.\n"
            "   - Phân tích minh chứng từ CV của Bạn để giải thích mã trên.\n"
            "   - Hồ sơ tương đồng: Giới thiệu 3 anh chị đi trước tiêu biểu trích xuất từ PHẦN B. Ghi rõ: [Tên người] [Số] - [Vị trí/Dự án chính].\n"
            "\n"
            "   ### PHẦN 2: LỘ TRÌNH NGOẠI KHÓA GỢI Ý (3 CẤP ĐỘ)\n"
            "   - Dựa trên kinh nghiệm thực tế của các anh chị trong PHẦN B, đề xuất lộ trình mới cho Bạn.\n"
            "   - Cấp độ Nền tảng (Foundation): [Mô tả].\n"
            "   - Cấp độ Chuyên môn (Professional): [Mô tả].\n"
            "   - Cấp độ Cá nhân (Personal): [Mô tả].\n"
            "\n"
            "   ### KẾT THÚC\n"
            "   - DANH SÁCH NGUỒN: Liệt kê đầy đủ: [Số] [Tên tệp .pdf/.docx thực tế trong PHẦN B].\n"
        ),
        top_k=15, source_filter=["riasec_knowledge", "internal_cv", "admission_profile"],
    ),

    AgentConfig(
        agent_id="hoidapchung",
        lang=Lang.VI, source=Source.ALL, topic=Topic.GENERAL,
        system_prompt=(
            "### TRỢ LÝ HƯỚNG NGHIỆP GPA AI ###\n"
            "1. VAI TRÒ: Chuyên giải đáp kiến thức hướng nghiệp, học bổng, ngành nghề.\n"
            "2. QUY TẮC CỨNG (CRITICAL): Tuyệt đối KHÔNG nhắc đến CV hoặc các quốc gia/chủ đề KHÔNG có thông tin cụ thể trong ngữ cảnh. Chỉ trả lời dựa trên những gì tìm thấy.\n"
            "3. TRÍCH DẪN: Phải trích dẫn nguồn theo đúng định dạng được yêu cầu ở phần Chỉ thị (Nguồn: [id] [Tên](url)).\n"
            "4. PHONG CÁCH: Chuyên nghiệp, cô đọng, đi thẳng vào dữ liệu."
        ),
        top_k=5, source_filter=None,
    ),
]

_AGENT_INDEX: dict[str, AgentConfig] = {a.agent_id: a for a in AGENTS}
_FALLBACK_AGENT = "hoidapchung"


# ── Router ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1000)
def route_question(question: str) -> AgentConfig:
    """
    Phân tích câu hỏi → Chọn Agent Hướng nghiệp & Ngoại khóa phù hợp nhất.
    """
    agents_desc = "\n".join(
        f'- "{a.agent_id}": chủ đề={a.topic.value}'
        for a in AGENTS
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là hệ thống phân loại câu hỏi tư vấn ngoại khóa và định hướng Holland RIASEC. "
                "Phân tích và chọn agent phù hợp nhất dựa trên:\n"
                "- test_riasec: [ƯU TIÊN] Phân loại, xác định nhóm Holland/RIASEC.\n"
                "- danhgia_all: [TỔNG THỂ] Tư vấn LỘ TRÌNH ngoại khóa 3 cấp độ, đánh giá toàn diện hồ sơ.\n"
                "- hdnk_nentang: [CỤ THỂ] Tìm kiếm/tư vấn các hoạt động ngoại khóa kỹ năng mềm, nền tảng.\n"
                "- hdnk_chuyenmon: Hoạt động ngoại khóa chuyên sâu, cọ xát thực tế.\n"
                "- hdnk_canhan: Hoạt động ngoại khóa theo sở thích cá nhân.\n"
                "- match_hoso: Tìm hồ sơ tương đồng.\n"
                "- hoidapchung: CHỈ dùng cho các câu chào hỏi hoặc câu hỏi không thể phân loại.\n\n"
                "QUY TẮC: Nếu User hỏi 'lộ trình' hoặc yêu cầu kế hoạch dài hạn -> LUÔN CHỌN danhgia_all.\n\n"
                "Trả lời ĐÚNG JSON, không thêm gì khác:\n"
                '{"agent_id": "<id>", "reason": "lý do"}'
            )
        },
        {
            "role": "user",
            "content": (
                f"Các agent:\n{agents_desc}\n\n"
                f"Câu hỏi từ người dùng: {question}\n\n"
                "Chọn agent_id phù hợp nhất:"
            )
        }
    ]

    try:
        raw = call_llm(messages, temperature=0.0, max_tokens=200)
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            result   = json.loads(raw[start:end])
            agent_id = result.get("agent_id", _FALLBACK_AGENT)
            reason   = result.get("reason", "")
            if agent_id in _AGENT_INDEX:
                print(f"[Router RIASEC] → {agent_id} | {reason}", file=sys.stderr)
                return _AGENT_INDEX[agent_id]
            print(f"[Router] agent_id '{agent_id}' không tồn tại → fallback", file=sys.stderr)
    except Exception as e:
        print(f"[Router] Lỗi: {e} → fallback {_FALLBACK_AGENT}", file=sys.stderr)

    return _AGENT_INDEX[_FALLBACK_AGENT]


# ── Search ────────────────────────────────────────────────────────────────────

def agent_search(question: str, agent: AgentConfig) -> dict:
    query_vec    = embed_texts([question], is_query=True)[0]
    store        = get_storage()
    
    print(f"[AgentSearch] Agent: {agent.agent_id}, Filter: {agent.source_filter}", file=sys.stderr)

    # ── Phân tách tìm kiếm cho danhgia_all để đảm bảo tính minh thực ────────────────
    if agent.agent_id == "danhgia_all":
        # Tìm lý thuyết (Top 4)
        theory_found = store.hybrid_search(
            query=question,
            query_vector=query_vec,
            top_k=4,
            tag_filter="riasec_knowledge"
        )
        # Tìm hồ sơ sinh viên thực tế (Hỗ trợ cả 2 tag nhãn phổ biến)
        profile_found = store.hybrid_search(
            query=question,
            query_vector=query_vec,
            top_k=10,
            tag_filter=["internal_cv", "admission_profile"]
        )
        
        # Gộp kết quả
        contexts = theory_found["contexts"] + profile_found["contexts"]
        sources  = theory_found["sources"] + profile_found["sources"]
        
        # Thêm thông tin trạng thái DB để agent biết đường "không bịa"
        db_status = "error" if store.is_fallback and not profile_found["contexts"] else "ok"
        
        return {"contexts": contexts, "sources": sources, "db_status": db_status, "max_score": 10.0}

    # ── Tìm kiếm thông thường cho các agent khác ────────────────────────────────────
    candidate_k = agent.top_k * 4
    if agent.source_filter is None:
        found = store.hybrid_search(
            query=question,
            query_vector=query_vec,
            top_k=candidate_k,
        )
    else:
        found = store.hybrid_search(
            query=question,
            query_vector=query_vec,
            top_k=candidate_k,
            tag_filter=agent.source_filter,
        )

    # ── Cross-Encoder Rerank (Chỉ cho luồng thông thường) ───────────────────────────
    reranked = rerank(
        query=question,
        contexts=found["contexts"],
        sources=found["sources"],
        top_k=agent.top_k,
    )
    
    # Thêm status
    reranked["db_status"] = "error" if store.is_fallback and not found["contexts"] else "ok"
    
    return reranked


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_agent_prompt(
    question: str,
    contexts: list[str],
    sources:  list[str],
    agent:    AgentConfig,
    history:  list[dict],
    db_status: str = "ok" # Thêm trạng thái DB
) -> list[dict]:
    # Phân loại tài liệu dựa trên tag (Strict Segregation)
    theory_sections = []
    profile_sections = []
    user_sections = []
    
    for i, (ctx, src) in enumerate(zip(contexts, sources)):
        doc_idx = i + 1
        # 1. Nhận diện hồ sơ của Người dùng
        if "USER_CV" in src.upper() or "HỒ SƠ CỦA NGƯỜI DÙNG" in src.upper():
            header = f'<NGƯỜI_DÙNG id="{doc_idx}" file="{src}">'
            footer = f'</NGƯỜI_DÙNG>'
            user_sections.append(f"{header}\n{ctx}\n{footer}")
        
        # 2. Nhận diện 92 hồ sơ thực tế (Dựa vào file và tag)
        elif "internal_cv" in src or "admission_profile" in src or any(src.lower().endswith(ext) for ext in ['.pdf', '.docx']):
            header = f'<DATABASE_92_CV id="{doc_idx}" file="{src}">'
            footer = f'</DATABASE_92_CV>'
            profile_sections.append(f"{header}\n{ctx}\n{footer}")
        
        # 3. Mặc định là Kiến thức lý thuyết
        else:
            header = f'<DATA_LÝ_THUYẾT id="{doc_idx}" file="{src}">'
            footer = f'</DATA_LÝ_THUYẾT>'
            theory_sections.append(f"{header}\n{ctx}\n{footer}")

    context_block = ""
    if user_sections:
        context_block += "### PHẦN 1: HỒ SƠ CỦA NGƯỜI DÙNG\n" + "\n\n".join(user_sections) + "\n\n"
    
    if theory_sections:
        context_block += "### PHẦN A: KIẾN THỨC LÝ THUYẾT RIASEC (CÁC VÍ DỤ TRONG NÀY LÀ GIẢ)\n" + "\n\n".join(theory_sections) + "\n\n"
        
    if profile_sections:
        context_block += "### PHẦN B: DATABASE 92 HỒ SƠ SINH VIÊN THỰC TẾ (CHỈ DÙNG ĐỂ LÀM LỘ TRÌNH)\n" + "\n\n".join(profile_sections) + "\n\n"

    user_content = (
        f"Thông tin Ngữ cảnh Hệ thống:\n{context_block}\n"
        f"==========================================\n"
        f"CÂU HỎI & CHỈ THỊ:\n"
        f"{question}\n\n"
        f"### CHỈ THỊ IDENTITY FIREWALL 3.0 ###\n"
        "1. XÁC MINH DANH TÍNH: Nếu không có thẻ <NGƯỜI_DÙNG> và người dùng yêu cầu 'Chẩn đoán cá nhân' hoặc 'Đánh giá hồ sơ của tôi', hãy yêu cầu gửi CV.\n"
        "2. KIỂM TRA KẾT NỐI: Nếu db_status='error' hoặc PHẦN B trống rỗng, hãy báo cáo: 'CẢNH BÁO: Không thể kết nối với CSDL Hồ sơ sinh viên (Docker Qdrant). Vui lòng kiểm tra lại hệ thống.'\n"
        "4. TRÍCH DẪN NGUỒN (CRITICAL): \n"
        "   - Mỗi khi sử dụng thông tin, phải ghi chú nguồn ngay cạnh dưới dạng: (Nguồn: [số]) hoặc [số].\n"
        "   - Số thứ tự [số] phải khớp hoàn toàn với danh sách ở mục 'Nguồn:' cuối bài.\n"
        "   - BẮT BUỘC liệt kê mục 'Nguồn:' ở cuối câu trả lời. Tuyệt đối không được bỏ sót bất kỳ nguồn nào đã trích dẫn trong bài.\n"
        "   - CẤM dùng ngoặc vuông [1], [2]... để đánh số thứ tự liệt kê hoặc làm tên (ví dụ: cấm viết 'Sinh viên [1]'). Ký hiệu [số] CHỈ ĐƯỢC DÙNG DUY NHẤT để trích dẫn.\n"
        "   - Định dạng danh sách nguồn: `[số] [Mô tả ngắn gọn hoặc Tiêu đề](url)`.\n"
        "5. Ý ĐỊNH DU HOC (INTENT): Nếu người dùng hỏi về 'Du học', tuyệt đối KHÔNG liệt kê các học bổng hoặc trường đại học tại Việt Nam trong câu trả lời.\n"
        "6. Lộ trình phải dựa 100% trên PHẦN B."
    )

    return [
        {"role": "system", "content": agent.system_prompt},
        *history,
        {"role": "user",   "content": user_content},
    ]


# ── Display helper ────────────────────────────────────────────────────────────

def get_agent_display(agent: AgentConfig) -> str:
    """Tên hiển thị cho Streamlit UI."""
    topic_icons = {
        Topic.RIASEC_TEST:      "Phân loại Holland",
        Topic.EC_FOUNDATION:    "Ngoại khóa Nền tảng",
        Topic.EC_PROFESSIONAL:  "Ngoại khóa Chuyên môn",
        Topic.EC_PERSONAL:      "Ngoại khóa Cá nhân",
        Topic.PROFILE_MATCHING: "Case Study 10 năm",
        Topic.COMPREHENSIVE_EVAL: "Đánh giá Toàn diện",
        Topic.GENERAL:          "Hướng nghiệp Tổng quát",
    }
    return f"{topic_icons.get(agent.topic, agent.agent_id)}"
