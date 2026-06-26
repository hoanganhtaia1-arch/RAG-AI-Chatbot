"""
semantic_cache.py — SQLite-backed Semantic Cache

Cache câu hỏi + câu trả lời dựa trên cosine similarity của embeddings (trấn giữ SQLite & Numpy).
Dữ liệu sẽ không bị mất khi restart (persistent cache) nhưng vẫn siêu nhanh.
"""

from __future__ import annotations
import math
import time
import sqlite3
import json
import uuid
import numpy as np


class SemanticCache:
    """
    SQLite-backed semantic cache cho RAG queries.

    TTL: mặc định 3600s (1 giờ), 0 = không hết hạn.
    Max size: mặc định 500 dòng (dùng LRU eviction).
    """

    def __init__(self, db_path="semantic_cache.db", threshold: float = 0.92, ttl: int = 3600, max_size: int = 500):
        self.db_path   = db_path
        self.threshold = threshold
        self.ttl       = ttl
        self.max_size  = max_size
        self._stats    = {"hits": 0, "misses": 0, "evictions": 0}
        self._init_db()

    def _init_db(self):
        """Tạo bảng nếu chưa có."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id TEXT PRIMARY KEY,
                    question_vec BLOB,
                    answer TEXT,
                    sources TEXT,
                    metadata TEXT,
                    timestamp REAL,
                    last_hit REAL,
                    hits INTEGER
                )
            """)
            conn.commit()

    def get(self, question_vec: list[float]) -> dict | None:
        now = time.time()
        
        # 1. Dọn dẹp cache hết hạn (TTL)
        if self.ttl > 0:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries WHERE ? - timestamp > ?", (now, self.ttl))
                conn.commit()

        # 2. Convert vector câu hỏi sang NumPy để tính toán nhanh
        q_vec_np = np.array(question_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec_np)
        if q_norm == 0:
            return None

        best_score = -1.0
        best_row = None

        # 3. Quét toàn bộ cache (tối đa 500 rows, xử lý bằng numpy < 5ms)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM cache_entries").fetchall()

            for r in rows:
                c_vec_np = np.frombuffer(r["question_vec"], dtype=np.float32)
                c_norm = np.linalg.norm(c_vec_np)
                if c_norm == 0:
                    continue
                
                score = float(np.dot(q_vec_np, c_vec_np) / (q_norm * c_norm))
                if score > best_score:
                    best_score = score
                    best_row = r

        # 4. Check ngưỡng threshold (cache hit)
        if best_row is not None and best_score >= self.threshold:
            self._stats["hits"] += 1
            print(f"[SemanticCache] HIT (similarity={best_score:.4f}, hits={best_row['hits'] + 1})")
            
            # Cập nhật số lần hit và lần hit cuối
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE cache_entries SET hits = hits + 1, last_hit = ? WHERE id = ?",
                    (now, best_row["id"])
                )
                conn.commit()

            return {
                "answer":   best_row["answer"],
                "sources":  json.loads(best_row["sources"]),
                "metadata": json.loads(best_row["metadata"]),
            }

        self._stats["misses"] += 1
        return None

    def set(self, question_vec: list[float], answer: str, sources: list[str], metadata: dict | None = None):
        now = time.time()

        with sqlite3.connect(self.db_path) as conn:
            # 1. Lọc bớt quá size limit (LRU Cache Eviction)
            count = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
            if count >= self.max_size:
                # Xóa ưu tiên hit_count thấp nhất rồi mới đến cũ nhất
                conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE id IN (
                        SELECT id FROM cache_entries 
                        ORDER BY hits ASC, last_hit ASC 
                        LIMIT 1
                    )
                """)
                self._stats["evictions"] += 1

            # 2. Insert row mới
            q_vec_bytes = np.array(question_vec, dtype=np.float32).tobytes()
            conn.execute("""
                INSERT INTO cache_entries (id, question_vec, answer, sources, metadata, timestamp, last_hit, hits)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                q_vec_bytes,
                answer,
                json.dumps(sources, ensure_ascii=False),
                json.dumps(metadata or {}, ensure_ascii=False),
                now,  # timestamp
                now,  # last_hit
                0     # hits
            ))
            conn.commit()
            print(f"[SemanticCache] SET (đã lưu sqlite)")

    def invalidate_all(self):
        """Xóa toàn bộ cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache_entries")
            conn.commit()
        print("[SemanticCache] Kho cache SQLite đã bị xóa sạch")

    def stats(self) -> dict:
        """Trả về thống kê."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]

        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        return {
            **self._stats,
            "size":     count,
            "hit_rate": round(hit_rate, 3),
        }


# ── Singleton instance ────────────────────────────────────────────────────────
_cache_instance: SemanticCache | None = None


def get_cache(threshold: float = 0.92, ttl: int = 3600) -> SemanticCache:
    """Lấy singleton SemanticCache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache(threshold=threshold, ttl=ttl)
        print(f"[SemanticCache] Khởi tạo SQLite cache (threshold={threshold}, ttl={ttl}s)")
    return _cache_instance
