"""
query_rewriter.py — Query Rewriting: HyDE + Multi-query

Trước khi search, LLM sinh ra nhiều biến thể truy vấn để tăng recall:

1. Multi-query  : sinh 3 câu hỏi tương đương từ góc độ khác nhau
2. HyDE         : sinh 1 đoạn văn giả định (hypothetical document) chứa câu trả lời
3. Merge        : embed tất cả biến thể, search song song, dedup + merge kết quả
"""

from __future__ import annotations
from llm_service import call_llm


async def generate_query_variants(
    question: str,
    n_variants: int = 3,
) -> list[str]:
    """
    Sinh n_variants biến thể câu hỏi + 1 HyDE document (Song song).
    """
    import asyncio
    
    # ── Multi-query ───────────────────────────────────────────────────────
    multi_query_prompt = [
        {
            "role": "system",
            "content": f"Hãy viết {n_variants} câu hỏi tương đương với câu hỏi gốc để tăng khả năng tìm kiếm. Không đánh số."
        },
        {"role": "user", "content": f"Câu hỏi gốc: {question}"}
    ]

    # ── HyDE ─────────────────────────────────────────────────────────────
    hyde_prompt = [
        {
            "role": "system",
            "content": "Viết một đoạn văn ngắn (3-5 câu) giả định chứa câu trả lời cho câu hỏi sau."
        },
        {"role": "user", "content": f"Câu hỏi: {question}"}
    ]

    variants = [question]

    async def _call_multi():
        try:
            raw = await asyncio.to_thread(call_llm, multi_query_prompt, 0.7, 300)
            return [q.strip() for q in raw.split("\n") if q.strip()][:n_variants]
        except: return []

    async def _call_hyde():
        try:
            return await asyncio.to_thread(call_llm, hyde_prompt, 0.5, 200)
        except: return None

    # Chạy song song cả 2 luồng LLM
    print(f"[QueryRewriter] Generatng variants in parallel...")
    mq_res, hyde_res = await asyncio.gather(_call_multi(), _call_hyde())
    
    variants.extend(mq_res)
    if hyde_res:
        variants.append(hyde_res)
    
    print(f"[QueryRewriter] Tổng {len(variants)} biến thể (mq={len(mq_res)}, hyde={'1' if hyde_res else '0'})")
    return variants


def merge_search_results(
    results_list: list[dict],
    top_k: int = 20,
) -> dict:
    """
    Merge nhiều kết quả search, dedup theo text, giữ thứ tự xuất hiện đầu tiên.

    Args:
        results_list : list[{"contexts": [...], "sources": [...]}]
        top_k        : số kết quả tối đa sau merge

    Returns:
        {"contexts": list[str], "sources": list[str]}
    """
    seen_texts: set[str] = set()
    merged_contexts: list[str] = []
    merged_sources:  list[str] = []

    for result in results_list:
        for ctx, src in zip(result.get("contexts", []), result.get("sources", [])):
            # Dedup theo 100 ký tự đầu của text
            key = ctx[:100].strip()
            if key and key not in seen_texts:
                seen_texts.add(key)
                merged_contexts.append(ctx)
                merged_sources.append(src)
                if len(merged_contexts) >= top_k:
                    break
        if len(merged_contexts) >= top_k:
            break

    print(f"[QueryRewriter] Merge: {sum(len(r.get('contexts',[])) for r in results_list)} → {len(merged_contexts)} unique contexts")
    return {"contexts": merged_contexts, "sources": merged_sources}
