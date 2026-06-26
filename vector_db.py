"""
vector_db.py — Qdrant wrapper với Hybrid Search (BM25 + Vector + RRF)

dim=1024 (multilingual-e5-large)
collection="docs"
"""

import os
import pickle
import time
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchText, MatchValue
)


_storage_instance = None

def get_storage(url="http://localhost:6333", collection="docs", dim=1024):
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = QdrantStorage(url, collection, dim)
    return _storage_instance


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=1024):
        # Thử kết nối server, nếu lỗi thì fallback sang local storage
        self.collection = collection
        self.is_fallback = False
        try:
            self.client = QdrantClient(url, timeout=5)
            # Thử một gọi lệnh nhẹ để check connection
            self.client.get_collections()
            print(f"[VectorDB] Connected to Qdrant server at {url}")
        except Exception:
            print(f"[VectorDB] Qdrant server not found at {url}. Falling back to local storage: ./qdrant_storage")
            self.client = QdrantClient(path="./qdrant_storage")
            self.is_fallback = True
        
        self._ensure_collection(dim)
        # BM25 index (lazy build, invalidated on upsert)
        self.bm25_cache_file = "bm25_index.pkl"
        self._bm25_index: BM25Okapi | None = None
        self._bm25_texts:    list[str]  = []
        self._bm25_ids:      list       = []
        self._bm25_payloads: list[dict] = []

    def _ensure_collection(self, dim: int):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        # Invalidate BM25 index sau khi upsert
        self._bm25_index = None
        if os.path.exists(self.bm25_cache_file):
            os.remove(self.bm25_cache_file)

    def clear_dynamic_web(self):
        """Xóa toàn bộ các chunk được ingest từ Tavily fallback (source_tag='dynamic_web')."""
        try:
            res = self.client.delete(
                collection_name=self.collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_tag",
                            match=MatchValue(value="dynamic_web")
                        )
                    ]
                )
            )
            print(f"[VectorDB] Đã dọn dẹp rác dynamic_web (Tavily cache)")
            self._bm25_index = None  # Invalidate BM25
            if os.path.exists(self.bm25_cache_file):
                os.remove(self.bm25_cache_file)
            return res
        except Exception as e:
            print(f"[VectorDB] Lỗi khi dọn dẹp dynamic_web: {e}")
            return None

    def search(self, query_vector: list[float], top_k: int = 5) -> dict:
        """Tìm kiếm tất cả nguồn — không filter."""
        resp = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        results = resp.points or []
        return {
            "contexts": [r.payload.get("text", "") for r in results],
            "sources": [r.payload.get("source", "") for r in results],
        }

    def search_with_filter(
            self,
            query_vector: list[float],
            top_k: int = 5,
            source_contains: str | None = None,
    ) -> dict:
        """
        Tìm kiếm với filter theo trường 'source' trong payload.

        source_contains:
            "uploads" → chỉ lấy chunk từ file PDF đã upload
            "http"    → chỉ lấy chunk từ URL đã crawl
            None      → tìm tất cả (gọi search() thường)
        """
        if source_contains is None:
            return self.search(query_vector, top_k)

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchText(text=source_contains)
                )
            ]
        )

        resp = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        results = resp.points or []

        # Fallback: nếu filter không ra kết quả → tìm tất cả
        if not results:
            print(f"[VectorDB] Filter '{source_contains}' không có kết quả → fallback tìm tất cả")
            return self.search(query_vector, top_k)

        return {
            "contexts": [r.payload.get("text", "") for r in results],
            "sources": [r.payload.get("source", "") for r in results],
        }

    def search_with_tag(
            self,
            query_vector: list[float],
            *,
            tag: str,
            top_k: int = 5,
            tag_key: str = "source_tag",
    ) -> dict:
        """
        Filter by an exact payload tag (safer than substring matching).
        """
        query_filter = Filter(
            must=[
                FieldCondition(
                    key=tag_key,
                    match=MatchValue(value=tag),
                )
            ]
        )
        resp = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        results = resp.points or []
        return {
            "contexts": [r.payload.get("text", "") for r in results],
            "sources": [r.payload.get("source", "") for r in results],
        }

    # ── BM25 helpers ──────────────────────────────────────────────────────────

    def _build_bm25_index(self):
        """
        Build BM25 index từ toàn bộ texts trong Qdrant.
        Load từ file .pkl nếu có để tối ưu tốc độ.
        """
        t0 = time.time()
        if os.path.exists(self.bm25_cache_file):
            try:
                with open(self.bm25_cache_file, "rb") as f:
                    data = pickle.load(f)
                    self._bm25_texts = data["texts"]
                    self._bm25_ids = data["ids"]
                    self._bm25_payloads = data["payloads"]
                    self._bm25_index = data["index"]
                print(f"[BM25] Đã load index từ {self.bm25_cache_file} ({time.time() - t0:.2f}s) - {len(self._bm25_texts)} docs")
                return
            except Exception as e:
                print(f"[BM25] Không thể load cache file: {e}. Đang rebuild...")

        all_points = []
        offset = None
        while True:
            result, offset = self.client.scroll(
                collection_name=self.collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(result)
            if offset is None:
                break

        self._bm25_texts    = [p.payload.get("text", "") for p in all_points]
        self._bm25_ids      = [p.id for p in all_points]
        self._bm25_payloads = [p.payload for p in all_points]

        tokenized = [t.lower().split() for t in self._bm25_texts]
        self._bm25_index = BM25Okapi(tokenized) if tokenized else None

        # Lưu lại cache
        try:
            with open(self.bm25_cache_file, "wb") as f:
                pickle.dump({
                    "texts": self._bm25_texts,
                    "ids": self._bm25_ids,
                    "payloads": self._bm25_payloads,
                    "index": self._bm25_index
                }, f)
        except Exception as e:
            print(f"[BM25] Lỗi khi lưu cache file: {e}")

        print(f"[BM25] Index built & saved: {len(self._bm25_texts)} docs ({time.time() - t0:.2f}s)")

    # ── Hybrid Search (BM25 + Vector + RRF) ──────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 5,
        rrf_k: int = 60,
        candidate_k: int | None = None,
        source_filter: str | None = None,
        tag_filter: str | None = None,
    ) -> dict:
        """
        Hybrid Search: kết hợp BM25 (keyword) + Vector (semantic) qua RRF.

        RRF score(d) = Σ 1 / (rrf_k + rank(d))

        Args:
            query         : câu hỏi gốc (dùng cho BM25)
            query_vector  : embedding của câu hỏi (dùng cho vector search)
            top_k         : số kết quả trả về cuối cùng
            rrf_k         : hằng số RRF (mặc định 60)
            candidate_k   : số ứng viên mỗi nhánh (mặc định top_k * 4)
            source_filter : filter theo substring trong trường "source"
            tag_filter    : filter theo exact match trường "source_tag"
        """
        if candidate_k is None:
            candidate_k = top_k * 4

        # ── Nhánh 1: Vector search ────────────────────────────────────────
        query_filter = None
        if tag_filter:
            if isinstance(tag_filter, list):
                from qdrant_client.models import MatchAny
                query_filter = Filter(must=[
                    FieldCondition(key="source_tag", match=MatchAny(any=tag_filter))
                ])
            else:
                query_filter = Filter(must=[
                    FieldCondition(key="source_tag", match=MatchValue(value=tag_filter))
                ])
        elif source_filter:
            query_filter = Filter(must=[
                FieldCondition(key="source", match=MatchText(text=source_filter))
            ])

        print(f"[HybridSearch] Filtering by: tag={tag_filter}, source={source_filter}")
        vec_resp = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=candidate_k,
            with_payload=True,
        )
        vec_results = vec_resp.points or []

        # ── Nhánh 2: BM25 search ──────────────────────────────────────────
        if self._bm25_index is None:
            self._build_bm25_index()

        bm25_top = []
        if self._bm25_index is not None and self._bm25_texts:
            tokenized_query = query.lower().split()
            scores = list(self._bm25_index.get_scores(tokenized_query))
            
            # Apply filters to BM25 scores
            for i in range(len(scores)):
                payload = self._bm25_payloads[i]
                if tag_filter:
                    if isinstance(tag_filter, list):
                        if payload.get("source_tag") not in tag_filter:
                            scores[i] = -1e9
                    elif payload.get("source_tag") != tag_filter:
                        scores[i] = -1e9
                elif source_filter and source_filter not in payload.get("source", ""):
                    scores[i] = -1e9
            
            bm25_top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidate_k]

        # ── RRF Fusion ────────────────────────────────────────────────────
        rrf_scores: dict = {}

        for rank, r in enumerate(vec_results):
            rrf_scores[r.id] = rrf_scores.get(r.id, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, idx in enumerate(bm25_top):
            if scores[idx] < 0: continue # Skip filtered out items
            doc_id = self._bm25_ids[idx]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        # ── Build payload map ─────────────────────────────────────────────
        id_to_payload: dict = {r.id: r.payload for r in vec_results}
        for idx in bm25_top:
            doc_id = self._bm25_ids[idx]
            if doc_id not in id_to_payload:
                id_to_payload[doc_id] = self._bm25_payloads[idx]

        contexts = [id_to_payload.get(did, {}).get("text", "")   for did in sorted_ids]
        sources  = [id_to_payload.get(did, {}).get("source", "") for did in sorted_ids]

        print(f"[HybridSearch] vec={len(vec_results)} bm25={len(bm25_top)} → merged top-{len(sorted_ids)}")
        return {"contexts": contexts, "sources": sources}

    def delete_collection(self):
        self.client.delete_collection(self.collection)
