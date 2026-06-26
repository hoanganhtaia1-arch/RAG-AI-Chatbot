import sys
import json
import pickle
import os
import asyncio
from typing import List, Dict

# Đảm bảo import được các module local
sys.path.append(os.getcwd())

from agents import route_question, agent_search, build_agent_prompt, _AGENT_INDEX
from llm_service import call_llm
from web_search import search_tavily
from data_loader import load_and_chunk_pdf

def load_system():
    pickle_path = "production_model.pkl"
    if not os.path.exists(pickle_path):
        return None
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def run_query(input_data: str):
    # 1. Load system state
    system_data = load_system()
    if not system_data:
        return {"error": "Hệ thống chưa được đóng gói (.pkl). Hãy chạy export_pkl.py trước."}

    # 2. Parse input (JSON or raw string)
    file_path = None
    question = ""
    target_agent_id = None
    try:
        data = json.loads(input_data)
        question = data.get("question", "")
        file_path = data.get("file_path")
        target_agent_id = data.get("agent_id")
    except:
        question = input_data

    try:
        # 3. File Context Extraction (CV Analysis)
        user_file_context = ""
        if file_path and os.path.exists(file_path):
            print(f"[Bridge] Đang bóc tách dữ liệu từ file: {file_path}...")
            chunks = load_and_chunk_pdf(file_path)
            # Lấy khoảng 2000 ký tự đầu tiên của CV để làm context (vì LLM có giới hạn)
            user_file_context = "\n".join([c["text"] for c in chunks[:5]])
            if not question:
                question = "Hãy phân tích hồ sơ/CV này và đưa ra tư vấn theo định hướng RIASEC và ngoại khóa."

        # 4. Routing
        if target_agent_id and target_agent_id in _AGENT_INDEX:
            agent = _AGENT_INDEX[target_agent_id]
            print(f"[Bridge] Cưỡng bức sử dụng Agent: {target_agent_id}")
        elif user_file_context:
            # Nếu có CV, ta ưu tiên Agent 6 (danhgia_all) - Comprehensive Evaluation
            agent = _AGENT_INDEX.get("danhgia_all") or route_question(question)
        else:
            agent = route_question(question)
        
        # 5. Retrieval (Local Search)
        search_results = agent_search(question, agent)
        contexts = search_results["contexts"]
        sources = search_results["sources"]
        db_status = search_results.get("db_status", "ok")

        # 5.1. Thêm CV vào đầu danh sách Nguồn nếu có
        if user_file_context:
            sources.insert(0, f"[USER_CV] {os.path.basename(file_path)}")
            contexts.insert(0, user_file_context)

        # 6. Tavily Fallback
        triggered_web = False
        max_score = search_results.get("max_score", -100.0)
        
        # Ngưỡng kích hoạt Tavily (Nâng lên 10.0 để đảm bảo chỉ dùng Database nếu cực kỳ khớp)
        WEB_TRIGGER_THRESHOLD = 10.0
        
        if (not contexts and not user_file_context) or (max_score < WEB_TRIGGER_THRESHOLD and not user_file_context):
            print(f"[Bridge] Dữ liệu nội bộ chưa đủ mạnh (score={round(max_score,2)} < {WEB_TRIGGER_THRESHOLD}). Kích hoạt Tavily Fallback...", file=sys.stderr)
            try:
                web_results = search_tavily(question, max_results=5)
                if web_results:
                    contexts = [r["content"] for r in web_results]
                    sources = [f"[Web] {r['url']}" for r in web_results]
                    triggered_web = True
            except: pass

        if not contexts and not user_file_context:
            return {
                "answer": "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu và không thể truy cập internet lúc này.",
                "sources": [],
                "agent": agent.agent_id
            }

        # 7. Prompt Building (Append CV context if exists)
        if user_file_context:
            question = f"### THÔNG TIN CV/HỒ SƠ CỦA NGƯỜI DÙNG: ###\n{user_file_context}\n\n### CÂU HỎI: ###\n{question}"

        messages = build_agent_prompt(question, contexts, sources, agent, history=[], db_status=db_status)

        # 8. Generation
        answer = call_llm(messages, temperature=0.2, max_tokens=1500)

        return {
            "answer": answer,
            "sources": sources,
            "agent": "CV_Analysis" if user_file_context else agent.agent_id,
            "status": "success"
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Thiếu dữ liệu đầu vào."}))
        sys.exit(1)

    input_arg = sys.argv[1]
    result = run_query(input_arg)
    print(json.dumps(result, ensure_ascii=False))
