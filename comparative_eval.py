"""
comparative_eval.py — So sánh 4 Pipeline trên 21 Kịch bản Thử nghiệm

Pipeline:
  1. Standalone LLM    : Chỉ gọi LLM, không truy xuất
  2. Naive RAG         : Vector search đơn giản + generic prompt
  3. Hybrid RAG        : BM25 + Vector + RRF + Reranker + generic prompt
  4. Agentic RAG (đề xuất): Full pipeline (Router + Hybrid + Rerank + Agent Prompt)

Chỉ số:
  - ROUGE-L, BERTScore F1, Citation Accuracy, Routing Accuracy

Chạy:
  uv run python comparative_eval.py
  uv run python comparative_eval.py --output results.json
"""

import json, os, re, sys, time, argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from llm_service import call_llm
from data_loader import embed_texts
from vector_db import get_storage
from reranker import rerank
from agents import route_question, agent_search, build_agent_prompt, get_agent_display

# ── Load ground truth ─────────────────────────────────────────────────────────
GT_PATH = Path("ground_truth.json")
with open(GT_PATH, "r", encoding="utf-8") as f:
    DATASET = json.load(f)

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L based on Longest Common Subsequence."""
    def clean(t):
        if not t: return ""
        t = re.split(r"Nguồ[n|ns]:", t, flags=re.IGNORECASE)[0]
        t = re.sub(r"\[\d+\]", "", t)
        return t
    def lcs(a, b):
        m, n = len(a), len(b)
        dp = [0] * (n + 1)
        for i in range(m):
            ndp = [0] * (n + 1)
            for j in range(n):
                if a[i] == b[j]: ndp[j+1] = dp[j] + 1
                else: ndp[j+1] = max(dp[j+1], ndp[j])
            dp = ndp
        return dp[n]
    ref = re.findall(r'\w+', clean(reference).lower(), re.UNICODE)
    hyp = re.findall(r'\w+', clean(hypothesis).lower(), re.UNICODE)
    if not ref or not hyp: return 0.0
    l = lcs(ref, hyp)
    r, p = l / len(ref), l / len(hyp)
    return (2 * r * p) / (r + p) if (r + p) > 0 else 0.0


def compute_bertscore(reference: str, hypothesis: str) -> float:
    """BERTScore F1 using bert-score library."""
    try:
        from bert_score import score as bscore
        P, R, F1 = bscore([hypothesis], [reference], lang="vi", verbose=False)
        return float(F1[0])
    except Exception:
        # Fallback: dùng embedding cosine similarity
        try:
            ref_vec = embed_texts([reference], is_query=False)[0]
            hyp_vec = embed_texts([hypothesis], is_query=False)[0]
            import math
            dot = sum(a * b for a, b in zip(ref_vec, hyp_vec))
            nr = math.sqrt(sum(a*a for a in ref_vec))
            nh = math.sqrt(sum(b*b for b in hyp_vec))
            return dot / (nr * nh) if nr * nh > 0 else 0.0
        except:
            return 0.5


def compute_citation(answer: str) -> float:
    """1.0 nếu câu trả lời có ít nhất 1 trích dẫn [n] hợp lệ."""
    return 1.0 if re.search(r"\[\d+\]", answer) else 0.0


# ── 4 Pipeline ────────────────────────────────────────────────────────────────

GENERIC_SYSTEM_PROMPT = (
    "Bạn là chuyên gia tư vấn hướng nghiệp Holland RIASEC và hoạt động ngoại khóa. "
    "Trả lời chính xác, có trích dẫn nguồn dạng [n]. "
    "Luôn liệt kê 'Nguồn: [n] tên_tệp' ở cuối câu trả lời."
)


def pipeline_standalone_llm(question: str) -> dict:
    """Pipeline 1: Chỉ LLM, không truy xuất."""
    t0 = time.time()
    messages = [
        {"role": "system", "content": GENERIC_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    answer = call_llm(messages, temperature=0.2, max_tokens=1500)
    return {
        "answer": answer,
        "sources": [],
        "agent": "none",
        "time": time.time() - t0,
    }


def pipeline_naive_rag(question: str) -> dict:
    """Pipeline 2: Vector search đơn giản + generic prompt."""
    t0 = time.time()
    q_vec = embed_texts([question], is_query=True)[0]
    store = get_storage()
    found = store.search(q_vec, top_k=5)
    contexts, sources = found["contexts"], found["sources"]

    if not contexts:
        return {"answer": "Không tìm thấy thông tin.", "sources": [], "agent": "naive", "time": time.time() - t0}

    ctx_block = "\n\n".join(
        f"[{i+1}] (Nguồn: {s})\n{c}" for i, (c, s) in enumerate(zip(contexts, sources))
    )
    messages = [
        {"role": "system", "content": GENERIC_SYSTEM_PROMPT},
        {"role": "user", "content": f"Ngữ cảnh:\n{ctx_block}\n\nCâu hỏi: {question}"},
    ]
    answer = call_llm(messages, temperature=0.2, max_tokens=1500)
    return {"answer": answer, "sources": sources, "agent": "naive", "time": time.time() - t0}


def pipeline_hybrid_rag(question: str) -> dict:
    """Pipeline 3: Hybrid Search + Reranker + generic prompt (không routing)."""
    t0 = time.time()
    q_vec = embed_texts([question], is_query=True)[0]
    store = get_storage()
    found = store.hybrid_search(query=question, query_vector=q_vec, top_k=20)
    reranked = rerank(query=question, contexts=found["contexts"], sources=found["sources"], top_k=5)
    contexts, sources = reranked["contexts"], reranked["sources"]

    if not contexts:
        return {"answer": "Không tìm thấy thông tin.", "sources": [], "agent": "hybrid", "time": time.time() - t0}

    ctx_block = "\n\n".join(
        f"[{i+1}] (Nguồn: {s})\n{c}" for i, (c, s) in enumerate(zip(contexts, sources))
    )
    messages = [
        {"role": "system", "content": GENERIC_SYSTEM_PROMPT},
        {"role": "user", "content": f"Ngữ cảnh:\n{ctx_block}\n\nCâu hỏi: {question}"},
    ]
    answer = call_llm(messages, temperature=0.2, max_tokens=1500)
    return {"answer": answer, "sources": sources, "agent": "hybrid", "time": time.time() - t0}


def pipeline_agentic_rag(question: str) -> dict:
    """Pipeline 4 (Đề xuất): Full Adaptive Agentic RAG."""
    t0 = time.time()
    agent = route_question(question)
    search_results = agent_search(question, agent)
    contexts = search_results["contexts"]
    sources = search_results["sources"]
    db_status = search_results.get("db_status", "ok")

    if not contexts:
        return {
            "answer": "Không tìm thấy thông tin.",
            "sources": [], "agent": agent.agent_id,
            "time": time.time() - t0,
        }

    messages = build_agent_prompt(question, contexts, sources, agent, history=[], db_status=db_status)
    answer = call_llm(messages, temperature=0.2, max_tokens=1500)
    return {
        "answer": answer,
        "sources": sources,
        "agent": agent.agent_id,
        "time": time.time() - t0,
    }


PIPELINES = {
    "Standalone LLM":      pipeline_standalone_llm,
    "Naive RAG":           pipeline_naive_rag,
    "Hybrid RAG":          pipeline_hybrid_rag,
    "Adaptive Agentic RAG": pipeline_agentic_rag,
}


# ── Main evaluation ──────────────────────────────────────────────────────────

def run_comparative_eval(use_bertscore: bool = True):
    all_results = {}

    for pipe_name, pipe_fn in PIPELINES.items():
        print(f"\n{'='*70}")
        print(f"  PIPELINE: {pipe_name}")
        print(f"{'='*70}")
        results = []

        for i, item in enumerate(DATASET):
            q = item["question"]
            ref = item["reference"]
            expected = item["expected_agent"]
            print(f"  [{i+1}/{len(DATASET)}] {item['scenario']}/{item['scenario_type']}: {q[:50]}...")

            try:
                out = pipe_fn(q)
            except Exception as e:
                print(f"    ⚠ ERROR: {e}")
                out = {"answer": f"Error: {e}", "sources": [], "agent": "error", "time": 0}

            # Compute metrics
            rouge = compute_rouge_l(ref, out["answer"])
            bert = compute_bertscore(ref, out["answer"]) if use_bertscore else 0.0
            citation = compute_citation(out["answer"])
            routing = 1.0 if out["agent"] == expected else 0.0

            results.append({
                "scenario": item["scenario"],
                "type": item["scenario_type"],
                "question": q,
                "expected_agent": expected,
                "predicted_agent": out["agent"],
                "routing_correct": routing,
                "rouge_l": round(rouge, 4),
                "bertscore_f1": round(bert, 4),
                "citation": citation,
                "response_time": round(out["time"], 2),
                "answer_preview": out["answer"][:200],
            })
            print(f"    → ROUGE={rouge:.3f} BERT={bert:.3f} Cite={citation:.0f} Time={out['time']:.1f}s Agent={out['agent']}")

        all_results[pipe_name] = results

    return all_results


def compute_summary(all_results: dict) -> dict:
    """Tính trung bình các chỉ số cho từng pipeline."""
    summary = {}
    for pipe_name, results in all_results.items():
        n = len(results)
        summary[pipe_name] = {
            "rouge_l":           round(sum(r["rouge_l"] for r in results) / n, 4),
            "bertscore_f1":      round(sum(r["bertscore_f1"] for r in results) / n, 4),
            "citation_accuracy": round(sum(r["citation"] for r in results) / n, 4),
            "routing_accuracy":  round(sum(r["routing_correct"] for r in results) / n, 4),
            "n_samples":         n,
        }
    return summary


def print_comparison_table(summary: dict):
    """In bảng so sánh đẹp ra terminal."""
    thresholds = {"rouge_l": 0.40, "bertscore_f1": 0.75, "citation_accuracy": 0.80, "routing_accuracy": 0.85}

    print(f"\n{'='*90}")
    print(f"{'BẢNG SO SÁNH HIỆU QUẢ 4 PIPELINE':^90}")
    print(f"{'='*90}")
    header = f"{'Chỉ số':<22}" + "".join(f"{name:>17}" for name in summary.keys())
    print(header)
    print("-" * 90)

    metrics = [
        ("ROUGE-L",           "rouge_l"),
        ("BERTScore F1",      "bertscore_f1"),
        ("Citation Accuracy", "citation_accuracy"),
        ("Routing Accuracy",  "routing_accuracy"),
    ]
    for label, key in metrics:
        row = f"{label:<22}"
        for name, s in summary.items():
            val = s[key]
            threshold = thresholds.get(key)
            if threshold:
                mark = "✅" if val >= threshold else "❌"
            else:
                mark = ""
            row += f"{val:>13.3f} {mark:>2}"
        print(row)

    print(f"{'='*90}")
    # Best pipeline
    best = max(summary.items(), key=lambda x: x[1]["rouge_l"] + x[1]["bertscore_f1"] + x[1]["citation_accuracy"])
    print(f"\n🏆 Pipeline tốt nhất (tổng hợp): {best[0]}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sánh 4 pipeline RAG trên 21 kịch bản")
    parser.add_argument("--output", type=str, default="comparative_results.json")
    parser.add_argument("--no-bert", action="store_true", help="Bỏ qua BERTScore (nhanh hơn)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   COMPARATIVE EVALUATION — 4 Pipelines × 21 Scenarios      ║")
    print("║   Adaptive Agentic RAG vs Baselines                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    all_results = run_comparative_eval(use_bertscore=not args.no_bert)
    summary = compute_summary(all_results)
    print_comparison_table(summary)

    # Save
    output = {"summary": summary, "details": all_results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Kết quả đã lưu: {args.output}")
