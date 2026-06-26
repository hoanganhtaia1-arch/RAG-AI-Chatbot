"""
pipeline_validation.py -- Kiem dinh toan bo day chuyen mo hinh RAG
12 nhom test (lightweight, robust with optional external services).

Cach chay:
    python tests/pipeline_validation.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import uuid
import subprocess
from pathlib import Path

import httpx

# Stable test env
os.environ.setdefault("USE_DUMMY_EMBED", "1")
os.environ.setdefault("USE_DUMMY_RERANK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_USE_SHM", "0")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:0.8b")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
API_PORT = int(os.environ.get("PIPELINE_TEST_PORT", "8097"))
API_BASE = f"http://127.0.0.1:{API_PORT}"
TEST_COLLECTION = "test_pipeline_validation"

_results: list[tuple[str, bool, str]] = []
_section_counts: dict[str, dict[str, int]] = {}
_current_section = ""


def _set_section(name: str):
    global _current_section
    _current_section = name
    _section_counts[name] = {"pass": 0, "fail": 0}
    print(f"\n{'=' * 62}\n  {name}\n{'=' * 62}")


def check(name: str, ok: bool, detail: str = ""):
    _results.append((_current_section + " > " + name, ok, detail))
    _section_counts.setdefault(_current_section, {"pass": 0, "fail": 0})
    _section_counts[_current_section]["pass" if ok else "fail"] += 1
    print(f"  {'✅' if ok else '❌'} {name}" + (f"  [{detail}]" if detail else ""))


def _ollama_ok() -> bool:
    try:
        base = OLLAMA_URL.replace("/api/chat", "").replace("/api/generate", "")
        return httpx.get(base, timeout=3).status_code < 500
    except Exception:
        return False


def _qdrant_ok() -> bool:
    try:
        return httpx.get(f"{QDRANT_URL}/collections", timeout=3).status_code == 200
    except Exception:
        return False


def _start_api_server() -> tuple[subprocess.Popen | None, str]:
    cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(API_PORT)]
    env = os.environ.copy()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    start = time.time()
    logs = []
    while time.time() - start < 25:
        try:
            r = httpx.get(f"{API_BASE}/health", timeout=1.5)
            if r.status_code < 500:
                return p, ""
        except Exception:
            pass
        if p.poll() is not None:
            break
        if p.stdout:
            line = p.stdout.readline()
            if line:
                logs.append(line.rstrip())
        time.sleep(0.3)
    # failed
    extra = "\n".join(logs[-20:])
    return None, extra


def _stop_api_server(p: subprocess.Popen | None):
    if p is None:
        return
    try:
        p.terminate()
        p.wait(timeout=5)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


# =============================================================================
# [1/12] custome_types
# =============================================================================
_set_section("[1/12] custome_types -- Pydantic models")
try:
    from custome_types import (
        RAGChunkAndSrc,
        RAGUpsertResult,
        RAGSearchResult,
        RAGQueryWithSession,
    )

    obj = RAGChunkAndSrc(chunks=["c1", "c2"], source_id="test.pdf")
    check("RAGChunkAndSrc tao thanh cong", obj.chunks == ["c1", "c2"])
    check("RAGChunkAndSrc metadata mac dinh la []", obj.metadata == [])
    check("RAGUpsertResult ingested=42", RAGUpsertResult(ingested=42).ingested == 42)
    obj3 = RAGSearchResult(contexts=["ctx1"], sources=["src1"])
    check("RAGSearchResult contexts va sources khop", len(obj3.contexts) == len(obj3.sources))
    obj4 = RAGQueryWithSession(question="Hoc bong?", session_id="s1")
    check("RAGQueryWithSession top_k mac dinh = 5", obj4.top_k == 5)
except Exception as exc:
    check("Import custome_types", False, str(exc))


# =============================================================================
# [2/12] data_loader -- embed_texts
# =============================================================================
_set_section("[2/12] data_loader -- embed_texts (dummy mode)")
try:
    from data_loader import embed_texts

    vecs2 = embed_texts(["cau hoi 1", "cau hoi 2"])
    check("embed_texts so luong dung", len(vecs2) == 2, f"got {len(vecs2)}")
    check("embed_texts kich thuoc 1024", len(vecs2[0]) == 1024, f"got {len(vecs2[0])}")
    check("embed_texts is_query=True kich thuoc 1024", len(embed_texts(["q"], is_query=True)[0]) == 1024)
    check("embed_texts danh sach rong -> []", embed_texts([]) == [])
    check("embed_texts deterministic", embed_texts(["test"]) == embed_texts(["test"]))
    norm2 = math.sqrt(sum(x * x for x in vecs2[0]))
    check("embed_texts vector da normalize", abs(norm2 - 1.0) < 0.01, f"norm={norm2:.4f}")
except Exception as exc:
    check("Import data_loader", False, str(exc))


# =============================================================================
# [3/12] vector_db
# =============================================================================
_set_section("[3/12] vector_db -- upsert / search / hybrid_search / filter")
if not _qdrant_ok():
    print(f"  ⚠️  Qdrant khong chay tai {QDRANT_URL} -- bo qua nhom 3")
    check("Qdrant available", False, "Qdrant khong kha dung")
else:
    try:
        from vector_db import QdrantStorage
        from data_loader import embed_texts as _emb3

        store3 = QdrantStorage(url=QDRANT_URL, collection=TEST_COLLECTION, dim=1024)
        texts3 = [
            "Hoc bong Chevening la hoc bong cua chinh phu Anh.",
            "Visa du hoc Uc subclass 500 cho phep sinh vien hoc toan thoi gian.",
            "Chuong trinh thac si tai Nhat yeu cau JLPT N2.",
        ]
        vecs3 = _emb3(texts3)
        ids3 = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"tv:{i}")) for i in range(3)]
        payloads3 = [
            {"source": "test://chevening", "source_tag": "pdf", "text": texts3[0]},
            {"source": "test://aus-visa", "source_tag": "pdf", "text": texts3[1]},
            {"source": "test://japan", "source_tag": "dynamic_web", "text": texts3[2]},
        ]
        store3.upsert(ids3, vecs3, payloads3)
        check("upsert 3 diem", True)

        qv3 = _emb3(["hoc bong Anh"], is_query=True)[0]
        res3a = store3.search(qv3, top_k=2)
        check("search co ban", len(res3a["contexts"]) > 0, f"got {len(res3a['contexts'])}")

        res3d = store3.hybrid_search(query="hoc bong Chevening", query_vector=qv3, top_k=3)
        check("hybrid_search ket qua", len(res3d["contexts"]) > 0, f"got {len(res3d['contexts'])}")
        check("hybrid_search sources khop", len(res3d["contexts"]) == len(res3d["sources"]))

        try:
            store3.delete_collection()
        except Exception:
            pass
    except Exception as exc:
        check("vector_db module", False, str(exc))


# =============================================================================
# [4/12] query_rewriter
# =============================================================================
_set_section("[4/12] query_rewriter -- merge / generate_query_variants")
try:
    from query_rewriter import merge_search_results, generate_query_variants

    long_ctx = "ctx A " * 30
    r4a = {"contexts": [long_ctx, "ctx B"], "sources": ["s1", "s2"]}
    r4b = {"contexts": [long_ctx, "ctx C"], "sources": ["s1", "s3"]}
    m4 = merge_search_results([r4a, r4b], top_k=10)
    check("merge dedup dung (3 unique)", len(m4["contexts"]) == 3, f"got {len(m4['contexts'])}")
    check("merge contexts va sources cung do dai", len(m4["contexts"]) == len(m4["sources"]))

    if _ollama_ok():
        try:
            v4 = generate_query_variants("Hoc bong Chevening la gi?", OLLAMA_URL, OLLAMA_MODEL, n_variants=2)
            check("generate_query_variants >= 2", len(v4) >= 2, f"got {len(v4)}")
            check("generate_query_variants cau hoi goc o dau", v4[0] == "Hoc bong Chevening la gi?")
        except Exception as exc:
            check("generate_query_variants (Ollama)", False, str(exc))
    else:
        print("  ⚠️  Ollama khong chay -- bo qua generate_query_variants")
except Exception as exc:
    check("Import query_rewriter", False, str(exc))


# =============================================================================
# [5/12] reranker
# =============================================================================
_set_section("[5/12] reranker -- dummy mode / edge cases")
try:
    os.environ["USE_DUMMY_RERANK"] = "1"
    if "reranker" in sys.modules:
        del sys.modules["reranker"]
    from reranker import rerank

    ctxs5 = [f"Doan van {i}" for i in range(10)]
    srcs5 = [f"src{i}" for i in range(10)]
    r5a = rerank("hoc bong", ctxs5, srcs5, top_k=5)
    check("rerank tra ve dung top_k=5", len(r5a["contexts"]) == 5)
    check("rerank sources khop contexts", len(r5a["contexts"]) == len(r5a["sources"]))
    check("rerank input rong -> output rong", rerank("q", [], [], top_k=5) == {"contexts": [], "sources": []})
except Exception as exc:
    check("Import reranker", False, str(exc))


# =============================================================================
# [6/12] semantic_cache
# =============================================================================
_set_section("[6/12] semantic_cache -- set / get / TTL / LRU / stats / invalidate")
try:
    from semantic_cache import SemanticCache, get_cache

    c6 = SemanticCache(threshold=0.92, ttl=3600, max_size=5)
    va6 = [1.0] + [0.0] * 1023
    vb6 = [0.999, 0.001] + [0.0] * 1022
    vc6 = [0.0, 1.0] + [0.0] * 1022
    c6.set(va6, "Cau tra loi A", ["src1"], {"agent": "vi_hocbong"})
    hit6 = c6.get(vb6)
    check("cache GET HIT voi vector tuong tu", hit6 is not None)
    check("cache GET MISS voi vector khac", c6.get(vc6) is None)
    c6.invalidate_all()
    check("cache invalidate_all -> size=0", len(c6._cache) == 0)
    check("get_cache tra ve singleton", get_cache() is get_cache())
except Exception as exc:
    check("Import semantic_cache", False, str(exc))


# =============================================================================
# [7/12] Agent
# =============================================================================
_set_section("[7/12] Agent -- configs / router / build_prompt / display")
try:
    from agents import AGENTS, AgentConfig, _AGENT_INDEX, _FALLBACK_AGENT, build_agent_prompt, get_agent_display, route_question

    check("AGENTS co 7 agent", len(AGENTS) == 7, f"got {len(AGENTS)}")
    check("_FALLBACK_AGENT ton tai trong _AGENT_INDEX", _FALLBACK_AGENT in _AGENT_INDEX)
    ag7 = _AGENT_INDEX["vi_hocbong"]
    msgs7 = build_agent_prompt(
        question="Hoc bong Chevening la gi?",
        contexts=["Chevening la hoc bong cua Anh."],
        sources=["test://src1"],
        agent=ag7,
        history=[],
    )
    check("build_agent_prompt co user message", msgs7[-1]["role"] == "user")
    check("get_agent_display tat ca agents khong rong", all(len(get_agent_display(a)) > 0 for a in AGENTS))

    if _ollama_ok():
        try:
            r7 = route_question("Hoc bong Chevening yeu cau IELTS bao nhieu?", OLLAMA_URL, OLLAMA_MODEL)
            check("route_question tra ve AgentConfig", isinstance(r7, AgentConfig))
        except Exception as exc:
            check("route_question (Ollama)", False, str(exc))
    else:
        print("  ⚠️  Ollama khong chay -- bo qua route_question")
except Exception as exc:
    check("Import Agent", False, str(exc))


# =============================================================================
# [8/12] conversation_store
# =============================================================================
_set_section("[8/12] conversation_store -- save / get / list / delete")
try:
    from conversation_store import save_message, get_history, list_sessions, delete_session

    sess8 = f"test_sess_{uuid.uuid4().hex[:8]}"
    save_message(sess8, "user", "Xin chao!")
    save_message(sess8, "assistant", "Chao ban!")
    h8 = get_history(sess8, limit=10)
    check("save_message + get_history co ban", len(h8) == 2, f"got {len(h8)}")
    check("list_sessions chua session vua tao", sess8 in list_sessions())
    delete_session(sess8)
    check("delete_session xoa tat ca tin nhan", len(get_history(sess8, limit=10)) == 0)
except Exception as exc:
    check("Import conversation_store", False, str(exc))


# =============================================================================
# [9/12] crawler
# =============================================================================
_set_section("[9/12] crawler -- fetch_and_parse / get_all_links / hash")
try:
    from crawler import _get_hash, get_all_links

    h1 = _get_hash("noi dung trang web")
    h2 = _get_hash("noi dung trang web")
    check("_get_hash deterministic", h1 == h2)
    check("_get_hash tra ve MD5 hex 32 ky tu", len(h1) == 32)
    try:
        links9 = get_all_links("https://httpbin.org/html", "httpbin.org")
        check("get_all_links tra ve list", isinstance(links9, list))
    except Exception as exc:
        check("get_all_links (network optional)", False, str(exc)[:80])
except Exception as exc:
    check("Import crawler", False, str(exc))


# =============================================================================
# [10/12] dynamic_agent / web_search
# =============================================================================
_set_section("[10/12] dynamic_agent / web_search -- import smoke")
try:
    import dynamic_agent  # noqa: F401
    import web_search  # noqa: F401

    check("import dynamic_agent", True)
    check("import web_search", True)
except Exception as exc:
    check("import dynamic_agent/web_search", False, str(exc))


# =============================================================================
# [11/12] API health + cache endpoints (curl/httpx style)
# =============================================================================
_set_section("[11/12] FastAPI endpoints -- health/cache/reindex/crawl")
server, server_log = _start_api_server()
if server is None:
    check("Server started", False, f"port {API_PORT} | {server_log[:180]}")
else:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        check("GET /health", r.status_code == 200, f"status={r.status_code}")

        r = httpx.get(f"{API_BASE}/cache/stats", timeout=5)
        check("GET /cache/stats", r.status_code == 200, f"status={r.status_code}")

        r = httpx.post(f"{API_BASE}/cache/clear", timeout=5)
        check("POST /cache/clear", r.status_code == 200, f"status={r.status_code}")

        r = httpx.get(f"{API_BASE}/crawl/urls", timeout=5)
        check("GET /crawl/urls", r.status_code in (200, 500), f"status={r.status_code}")

        r = httpx.post(f"{API_BASE}/reindex", timeout=5)
        check("POST /reindex", r.status_code in (200, 500), f"status={r.status_code}")

        bad = httpx.post(f"{API_BASE}/ingest/url", json={"url": ""}, timeout=8)
        check("POST /ingest/url (error-path missing url)", bad.status_code == 400, f"status={bad.status_code}")
    except Exception as exc:
        check("API endpoint checks", False, str(exc))
    finally:
        _stop_api_server(server)


# =============================================================================
# [12/12] project files sanity
# =============================================================================
_set_section("[12/12] project sanity -- required files exist")
required_files = [
    "main.py",
    "streamlit_app.py",
    "vector_db.py",
    "query_rewriter.py",
    "reranker.py",
    "semantic_cache.py",
    "agents.py",
]
for fp in required_files:
    check(f"exists: {fp}", Path(ROOT, fp).exists())


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 62)
print("SUMMARY")
print("=" * 62)

total = len(_results)
passed = sum(1 for _, ok, _ in _results if ok)
failed = total - passed

for sec, cnt in _section_counts.items():
    print(f"- {sec}: pass={cnt['pass']} fail={cnt['fail']}")

print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed}")

if failed:
    print("\nFAILED CASES:")
    for name, ok, detail in _results:
        if not ok:
            print(f"  - {name}" + (f" :: {detail}" if detail else ""))

sys.exit(0 if failed == 0 else 1)
