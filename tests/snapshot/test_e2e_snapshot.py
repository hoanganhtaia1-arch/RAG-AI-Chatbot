import pytest
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents import route_question

def test_routing_golden_snapshot(snapshot):
    """
    Tầng 2: Golden Snapshot - Ghi lại kết xuất chuẩn của Query Router.
    Coi đây như một End-to-End block. Bất cứ ai thay đổi Router prompt 
    hoặc LLM settings khiến agent switch sai mục tiêu, test này sẽ FAILED.
    """
    # Mẫu data thật
    questions_e2e = [
        "Làm sao để chuẩn bị hồ sơ xin visa du học Mỹ 2026?",
        "Mức học phí đại học Toronto của Canada và chi phí sinh hoạt là bao nhiêu?",
        "Đăng ký thi IELTS cần đạt bn điểm để học thạc sĩ IT?",
        "Bạn có thể viết cho tôi một bài code Python không?"
    ]
    
    results = []
    
    for q in questions_e2e:
        try:
            # Lưu ý Ollama phải đang chạy ở locahost:11434 với model qwen3.5:0.8b
            agent = route_question(q, ollama_url="http://localhost:11434/api/chat", ollama_model="qwen3.5:0.8b")
            results.append({
                "question": q,
                "selected_agent": agent.agent_id,
                "topic": agent.topic.value
            })
        except Exception as e:
            # Fallback capture để snapshot luôn tình trạng lỗi kết nối nếu có
            results.append({
                "question": q,
                "error": str(e)
            })

    snapshot_str = json.dumps(results, indent=2, ensure_ascii=False)
    
    # Snapshot_update flag trong pytest sẽ lưu đè JSON đầu ra thành baseline chuẩn
    snapshot.assert_match(snapshot_str, "routing_decisions.json")
