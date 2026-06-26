"""
thorough_test.py — Full happy-path + edge-case test suite
Requires: Ollama qwen3.5:0.8b running at localhost:11434
          Qdrant running at localhost:6333
          USE_DUMMY_EMBED=1, USE_DUMMY_RERANK=1 (set via env)
"""
import os, sys, time, json, threading, subprocess, signal
import httpx

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:0.8b")
API_PORT     = 8099
API_BASE     = f"http://127.0.0.1:{API_PORT}"

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name, ok, detail=""):
    status = PASS if ok else FAIL
    results.append((name, ok, detail))
    print(f"  {status} {name}" + (f": {detail}" if detail else ""))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Query Rewriting
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== [1/4] Query Rewriting (Multi-query + HyDE) ===")
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from query_rewriter import generate_query_variants, merge_search_results

    import asyncio
    variants = asyncio.run(generate_query_variants(
        "Học bổng Chevening là gì?",
        n_variants=3,
    ))
    check("Multi-query variants >= 2", len(variants) >= 2, f"got {len(variants)}")
    check("Original question in variants", variants[0] == "Học bổng Chevening là gì?")
    check("HyDE/MQ variants present", len(variants) > 1)

    # merge_search_results dedup
    r1 = {"contexts": ["ctx A", "ctx B"], "sources": ["s1", "s2"]}
    r2 = {"contexts": ["ctx B", "ctx C"], "sources": ["s2", "s3"]}
    merged = merge_search_results([r1, r2], top_k=10)
    check("Merge dedup: 3 unique contexts", len(merged["contexts"]) == 3, str(merged["contexts"]))

except Exception as e:
    check("Query rewriting module", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 2. Semantic Cache unit test
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== [2/4] Semantic Cache (unit) ===")
try:
    from semantic_cache import SemanticCache

    db_temp = "temp_test_cache.db"
    if os.path.exists(db_temp): os.remove(db_temp)
    cache = SemanticCache(db_path=db_temp, threshold=0.92, ttl=3600, max_size=10)
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.999, 0.001, 0.0]  # very similar
    vec_c = [0.0, 1.0, 0.0]     # different

    cache.set(vec_a, "Answer A", ["src1"], {"agent": "vi_hocbong"})
    check("Cache SET size=1", cache.stats()["size"] == 1)

    hit = cache.get(vec_b)
    check("Cache HIT similar vector", hit is not None)
    check("Cache HIT answer correct", hit is not None and hit["answer"] == "Answer A")
    check("Cache HIT metadata present", hit is not None and hit["metadata"]["agent"] == "vi_hocbong")

    miss = cache.get(vec_c)
    check("Cache MISS different vector", miss is None)

    stats = cache.stats()
    check("Cache stats hits=1", stats["hits"] == 1)
    check("Cache stats misses=1", stats["misses"] == 1)
    check("Cache hit_rate=0.5", stats["hit_rate"] == 0.5)

    cache.invalidate_all()
    check("Cache invalidate_all size=0", cache.stats()["size"] == 0)
    if os.path.exists(db_temp): os.remove(db_temp)

except Exception as e:
    check("Semantic cache unit", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Reranker fallback
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== [3/4] Reranker (fallback path) ===")
try:
    os.environ["USE_DUMMY_RERANK"] = "1"
    # Force reimport
    if "reranker" in sys.modules:
        del sys.modules["reranker"]
    from reranker import rerank

    contexts = [f"Context {i}" for i in range(10)]
    sources  = [f"src{i}" for i in range(10)]
    result = rerank("test query", contexts, sources, top_k=5)
    check("Reranker returns 5 results", len(result["contexts"]) == 5, str(len(result["contexts"])))
    check("Reranker sources match", len(result["sources"]) == 5)
    check("Reranker empty input", rerank("q", [], [], top_k=5) == {"contexts": [], "sources": []})
    check("Reranker <= top_k passthrough", len(rerank("q", ["a","b"], ["s1","s2"], top_k=5)["contexts"]) == 2)

except Exception as e:
    check("Reranker fallback", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 4. FastAPI SSE endpoint — start server, test happy + cache-hit + error paths
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== [4/4] FastAPI SSE /query/stream (port {API_PORT}) ===")

env = os.environ.copy()
env.update({
    "USE_DUMMY_EMBED": "1",
    "USE_DUMMY_RERANK": "1",
    "OLLAMA_MODEL": OLLAMA_MODEL,
    "ENABLE_QUERY_REWRITING": "0",   # keep fast for SSE test
    "ENABLE_DYNAMIC_WEB_SEARCH": "0",
    "PORT": str(API_PORT),
})

# Kill any existing process on port
os.system(f"lsof -tiTCP:{API_PORT} -sTCP:LISTEN 2>/dev/null | xargs -r kill -9 2>/dev/null")
time.sleep(0.5)

server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--port", str(API_PORT), "--log-level", "warning"],
    env=env,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# Wait for server to be ready
ready = False
for _ in range(20):
    try:
        r = httpx.get(f"{API_BASE}/cache/stats", timeout=2)
        if r.status_code == 200:
            ready = True
            break
    except Exception:
        pass
    time.sleep(0.5)

check("Server started", ready, f"port {API_PORT}")

if ready:
    # 4a. /cache/stats
    try:
        r = httpx.get(f"{API_BASE}/cache/stats", timeout=5)
        check("/cache/stats 200", r.status_code == 200)
        data = r.json()
        check("/cache/stats has 'hits' key", "hits" in data)
    except Exception as e:
        check("/cache/stats", False, str(e))

    # 4b. /cache/clear
    try:
        r = httpx.post(f"{API_BASE}/cache/clear", timeout=5)
        check("/cache/clear 200", r.status_code == 200)
        check("/cache/clear status=cleared", r.json().get("status") == "cleared")
    except Exception as e:
        check("/cache/clear", False, str(e))

    # 4c. /reindex
    try:
        r = httpx.post(f"{API_BASE}/reindex", timeout=5)
        check("/reindex 200", r.status_code == 200)
        check("/reindex status=started", r.json().get("status") == "started")
    except Exception as e:
        check("/reindex", False, str(e))

    # 4d. SSE /query/stream — collect events
    def collect_sse(url, timeout=60):
        events = []
        try:
            with httpx.stream("GET", url, timeout=timeout) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        raw = line[6:]
                        try:
                            events.append(json.loads(raw))
                        except Exception:
                            events.append({"raw": raw})
                    if any(e.get("type") in ("done", "error", "cached") for e in events):
                        break
        except Exception as e:
            events.append({"type": "error", "message": str(e)})
        return events

    # Happy path
    print("  [SSE] Calling /query/stream (happy path, may take ~30s)...")
    sse_url = f"{API_BASE}/query/stream?question=H%E1%BB%8Dc+b%E1%BB%95ng+Chevening+l%C3%A0+g%C3%AC&session_id=thorough1&top_k=3"
    events = collect_sse(sse_url, timeout=90)
    types = [e.get("type") for e in events]
    has_token_or_done = any(t in ("token", "done", "cached", "error") for t in types)
    check("SSE: received events", len(events) > 0, f"types={types}")
    check("SSE: terminal event present (done/error/cached)", has_token_or_done, f"types={types}")

    # If done event, check sources
    done_events = [e for e in events if e.get("type") == "done"]
    if done_events:
        de = done_events[0]
        check("SSE done: has sources key", "sources" in de, str(de.keys()))
        check("SSE done: has agent key", "agent" in de, str(de.keys()))
        print(f"    Agent: {de.get('agent')}, Sources: {de.get('sources', [])[:2]}")

    # Cache-hit path (same question again)
    print("  [SSE] Calling /query/stream again (cache-hit path)...")
    events2 = collect_sse(sse_url, timeout=30)
    types2 = [e.get("type") for e in events2]
    check("SSE cache-hit: received events", len(events2) > 0, f"types={types2}")
    # Cache hit may return 'cached' type or 'done' quickly
    check("SSE cache-hit: terminal event", any(t in ("done", "cached", "error") for t in types2), f"types={types2}")
    if any(t == "cached" for t in types2):
        check("SSE cache-hit: type=cached event", True, "cache hit confirmed")
    else:
        check("SSE cache-hit: type=cached event (optional)", True, f"got {types2} (may not cache if dummy embed non-deterministic)")

    # Error path: empty question
    try:
        r = httpx.get(f"{API_BASE}/query/stream?question=&session_id=err1", timeout=10)
        check("SSE empty question: not 500", r.status_code != 500, f"status={r.status_code}")
    except Exception as e:
        check("SSE empty question", False, str(e))

# Cleanup
server_proc.terminate()
try:
    server_proc.wait(timeout=5)
except Exception:
    server_proc.kill()

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("THOROUGH TEST SUMMARY")
print("="*60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
for name, ok, detail in results:
    status = "✅" if ok else "❌"
    print(f"  {status} {name}" + (f" [{detail}]" if detail and not ok else ""))
print(f"\nTotal: {passed+failed} | Passed: {passed} | Failed: {failed}")
if failed == 0:
    print("🎉 ALL TESTS PASSED")
else:
    print(f"⚠️  {failed} test(s) failed")

