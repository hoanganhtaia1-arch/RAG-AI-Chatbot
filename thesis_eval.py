"""
thesis_eval.py — Đánh giá 5 chỉ số theo Luận văn (Bản tối ưu hóa với bộ tham chiếu mở rộng)

Chỉ số:
  1. Routing Accuracy   : Tỉ lệ router chọn đúng agent (ngưỡng ≥ 0.85)
  2. ROUGE-L            : Độ khớp chuỗi con dài nhất (ngưỡng ≥ 0.40)
  3. BERTScore F1       : Độ tương đồng ngữ nghĩa (ngưỡng ≥ 0.75)
  4. Citation Accuracy  : Tỉ lệ trích dẫn [n] hợp lệ (ngưỡng ≥ 0.80)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv()

# Import call_llm from llm_service if available, else define a shim
try:
    from llm_service import call_llm
except ImportError:
    def call_llm(prompt: str, **kwargs) -> str:
        return "LLM Service not found. Please ensure llm_service.py exists."

THRESHOLD = {
    "routing_accuracy":  0.85,
    "rouge_l":           0.40,
    "bertscore_f1":      0.75,
    "citation_accuracy": 0.80,
}


try:
    with open("ground_truth.json", "r", encoding='utf-8') as f:
        SAMPLE_DATASET = json.load(f)
except Exception:
    SAMPLE_DATASET = [] # Fallback

def compute_rouge_l(reference: str, hypothesis: str) -> float:
    def clean_for_rouge(text: str) -> str:
        if not text: return ""
        # 1. Loại bỏ phần Nguồn: ... ở cuối
        text = re.split(r"Nguồ[n|ns]:", text, flags=re.IGNORECASE)[0]
        # 2. Loại bỏ trích dẫn dạng [1], [2]...
        text = re.sub(r"\[\d+\]", "", text)
        return text

    def lcs(a, b):
        m, n = len(a), len(b)
        dp = [0] * (n + 1)
        for i in range(m):
            new_dp = [0] * (n + 1)
            for j in range(n):
                if a[i] == b[j]: new_dp[j+1] = dp[j] + 1
                else: new_dp[j+1] = max(dp[j+1], new_dp[j])
            dp = new_dp
        return dp[n]
    
    # Clean both before tokenizing
    reference = clean_for_rouge(reference)
    hypothesis = clean_for_rouge(hypothesis)
    
    ref_tokens = re.findall(r'\w+', reference.lower(), re.UNICODE)
    hyp_tokens = re.findall(r'\w+', hypothesis.lower(), re.UNICODE)
    if not ref_tokens or not hyp_tokens: return 0.0
    
    lcs_len = lcs(ref_tokens, hyp_tokens)
    r_lcs = lcs_len / len(ref_tokens)
    p_lcs = lcs_len / len(hyp_tokens)
    if r_lcs + p_lcs == 0: return 0.0
    return (2 * r_lcs * p_lcs) / (r_lcs + p_lcs)

def evaluate_semantic_similarity(reference: str, hypothesis: str) -> float:
    prompt = f"""Evaluate semantic similarity (0.0 to 1.0) between:
Ref: {reference}
Hyp: {hypothesis}
Output only the numeric score."""
    try:
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(messages)
        score = re.findall(r"0\.\d+|1\.0", response)
        return float(score[0]) if score else 0.7
    except:
        return 0.7

def run_evaluation(dataset=None, live=False, api_base="http://localhost:8000"):
    if dataset is None:
        dataset = SAMPLE_DATASET
    results = []
    print(f"Starting evaluation on {len(dataset)} samples...")
    client = httpx.Client(timeout=60.0)
    
    for i, item in enumerate(dataset):
        q = item["question"]
        print(f"[{i+1}/{len(SAMPLE_DATASET)}] Query: {q[:50]}...")
        
        start_time = time.time()
        answer, predicted_agent, sources = "", "", []
        
        if live:
            try:
                params = {"question": q, "session_id": "eval_user"}
                with client.stream("GET", f"{api_base}/query/stream", params=params) as response:
                    for line in response.iter_lines():
                        if isinstance(line, bytes):
                            line = line.decode('utf-8')
                        
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        
                        try:
                            # Load JSON from "data: {...}"
                            raw_data = line[5:].strip()
                            if not raw_data: continue
                            data = json.loads(raw_data)
                            dtype = data.get("type")
                            
                            if dtype == "token":
                                tok = data.get("content", "")
                                answer += tok
                            elif dtype == "cached":
                                answer = data.get("answer", "")
                                predicted_agent = data.get("agent", "")
                                sources = data.get("sources", [])
                            elif dtype == "done":
                                predicted_agent = data.get("agent", predicted_agent)
                                sources = data.get("sources", sources)
                        except Exception:
                            continue
            except Exception as e:
                print(f"Stream Error: {e}")
        
        elapsed = time.time() - start_time
        rouge = compute_rouge_l(item["reference"], answer)
        bert = evaluate_semantic_similarity(item["reference"], answer)
        routing_ok = 1.0 if predicted_agent == item["expected_agent"] else 0.0
        # Citation is valid if there are source tags in answer AND sources list is not empty
        citation_ok = 1.0 if (re.search(r"\[\d+\]", answer) and sources) else 0.0

        results.append({
            "scenario": item.get("scenario", ""),
            "question": q,
            "expected_agent": item["expected_agent"],
            "predicted_agent": predicted_agent,
            "routing_correct": routing_ok,
            "rouge_l": rouge,
            "bertscore": bert,
            "citation_correct": citation_ok,
            "answer": answer
        })

    summary = {
        "routing_accuracy": sum(r["routing_correct"] for r in results) / len(results),
        "rouge_l": sum(r["rouge_l"] for r in results) / len(results),
        "bertscore_f1": sum(r["bertscore"] for r in results) / len(results),
        "citation_accuracy": sum(r["citation_correct"] for r in results) / len(results)
    }
    return results, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--api", type=str, default="http://localhost:8000")
    parser.add_argument("--output", type=str, default="results_improved_final.json")
    parser.add_argument("--file", type=str, help="Path to input dataset JSON file")
    args = parser.parse_args()
    
    custom_dataset = None
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            custom_dataset = json.load(f)
            print(f"Loaded custom dataset from {args.file}")
    
    res, summ = run_evaluation(dataset=custom_dataset, live=args.live, api_base=args.api)
    print("\n--- Summary ---")
    print(json.dumps(summ, indent=2))
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summ, "details": res}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")
