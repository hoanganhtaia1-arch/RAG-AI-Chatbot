"""
reranker.py — Cross-Encoder Reranker

Sau khi retrieve top-20 từ Hybrid Search,
rerank bằng ms-marco-MiniLM-L-6-v2 → lấy top-5.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Nhỏ gọn (~85MB), nhanh, tốt với tiếng Anh và đa ngôn ngữ
  - Score cao = đoạn văn liên quan với câu hỏi
"""

from __future__ import annotations
import os

_reranker = None
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker():
    """Lazy load Cross-Encoder model."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"[Reranker] Loading model: {_RERANKER_MODEL}")
        _reranker = CrossEncoder(_RERANKER_MODEL)
        print("[Reranker] Model loaded ✅")
    return _reranker


def rerank(
    query: str,
    contexts: list[str],
    sources: list[str],
    top_k: int = 5,
) -> dict:
    """
    Rerank danh sách contexts theo độ liên quan với query.

    Args:
        query    : câu hỏi gốc
        contexts : list đoạn văn cần rerank
        sources  : list nguồn tương ứng với contexts
        top_k    : số kết quả trả về sau rerank

    Returns:
        {"contexts": list[str], "sources": list[str]}
    """
    # Optional bypass for environments without Torch/Transformers
    if os.getenv("USE_DUMMY_RERANK", "0").strip() in ("1", "true", "True"):
        print(f"[Reranker] USE_DUMMY_RERANK=1 → return top-{top_k} without rerank")
        return {"contexts": contexts[:top_k], "sources": sources[:top_k]}

    if not contexts:
        return {"contexts": [], "sources": []}

    try:
        model = _get_reranker()

        # Tạo pairs (query, passage) cho Cross-Encoder
        pairs = [(query, ctx) for ctx in contexts]
        scores = model.predict(pairs)

        # Sắp xếp theo score giảm dần
        full_ranked = sorted(
            zip(scores, contexts, sources),
            key=lambda x: x[0],
            reverse=True,
        )

        # Lọc theo threshold (Giảm xuống -10.0 để đảm bảo đủ số lượng Top-K cho tính đa dạng)
        RERANK_THRESHOLD = -10.0
        filtered = [r for r in full_ranked if r[0] >= RERANK_THRESHOLD]

        # ĐA DẠNG HÓA NGUỒN: Ưu tiên chọn mỗi file ít nhất 1 chunk trước
        diverse_ranked = []
        seen_sources = set()
        
        # Bước 1: Lấy chunk top 1 của mỗi file
        for r in filtered:
            if r[2] not in seen_sources:
                diverse_ranked.append(r)
                seen_sources.add(r[2])
            if len(diverse_ranked) >= top_k:
                break
        
        # Bước 2: Nếu chưa đủ top_k, lấy thêm các chunk khác theo thứ tự score
        if len(diverse_ranked) < top_k:
            for r in filtered:
                if r not in diverse_ranked:
                    diverse_ranked.append(r)
                if len(diverse_ranked) >= top_k:
                    break

        reranked_contexts = [r[1] for r in diverse_ranked]
        reranked_sources  = [r[2] for r in diverse_ranked]
        max_score = float(diverse_ranked[0][0]) if diverse_ranked else -100.0

        print(f"[Reranker] {len(contexts)} → {len(reranked_contexts)} (Diversified) | Max Score: {round(max_score, 3)}")
        for r in diverse_ranked:
            print(f"  - Score: {round(float(r[0]), 3)} | File: {r[2]}")

        return {
            "contexts": reranked_contexts, 
            "sources": reranked_sources,
            "max_score": max_score
        }

    except Exception as e:
        print(f"[Reranker] Lỗi: {e} → trả về top-{top_k} không rerank")
        return {"contexts": contexts[:top_k], "sources": sources[:top_k]}
