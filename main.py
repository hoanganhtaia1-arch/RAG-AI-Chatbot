"""
main.py — FastAPI Backend cho hệ thống Adaptive Agentic RAG

Endpoints:
  GET  /health          : Kiểm tra trạng thái hệ thống & kết nối Qdrant
  POST /query           : Truy vấn RAG đầy đủ pipeline (Semantic Cache → Router → Hybrid Search → Rerank → LLM)
  POST /query/stream    : SSE streaming version của /query
  POST /ingest/pdf      : Upload & ingest PDF vào Qdrant
  POST /ingest/url      : Crawl URL & ingest vào Qdrant
  GET  /cache/stats     : Thống kê Semantic Cache
  POST /cache/clear     : Xóa toàn bộ Semantic Cache
  GET  /conversations   : Liệt kê các session hội thoại
  GET  /conversations/{session_id} : Lấy lịch sử hội thoại

Khởi chạy:
  uv run uvicorn main:app --reload
"""

import os
import sys
import json
import uuid
import asyncio
import tempfile
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── Local modules ──────────────────────────────────────────────────────────────
from agents import route_question, agent_search, build_agent_prompt, get_agent_display
from llm_service import call_llm, stream_llm
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import get_storage
from semantic_cache import get_cache
from conversation_store import save_message, get_history, list_sessions, delete_session
from web_search import search_tavily
from crawler import fetch_and_parse
from query_rewriter import generate_query_variants, merge_search_results

import inngest
from inngest.fast_api import serve

# ── App init ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Adaptive Agentic RAG API",
    description="Backend API cho hệ thống tư vấn hướng nghiệp RIASEC & Ngoại khóa",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Feature flags ──────────────────────────────────────────────────────────────
ENABLE_CACHE = os.getenv("ENABLE_SEMANTIC_CACHE", "1").strip() in ("1", "true")
ENABLE_WEB   = os.getenv("ENABLE_DYNAMIC_WEB_SEARCH", "1").strip() in ("1", "true")
ENABLE_QR    = os.getenv("ENABLE_QUERY_REWRITING", "1").strip() in ("1", "true")

inngest_client = inngest.Inngest(app_id="adaptive-rag", is_production=False)
serve(app, inngest_client, [])
# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: int = 5

class IngestURLRequest(BaseModel):
    url: str
    source_tag: str = "dynamic_web"


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái hệ thống."""
    store = get_storage()
    qdrant_ok = not store.is_fallback
    try:
        collections = store.client.get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    return {
        "status": "ok",
        "qdrant_ok": qdrant_ok,
        "flags": {
            "semantic_cache": ENABLE_CACHE,
            "dynamic_web_search": ENABLE_WEB,
            "query_rewriting": ENABLE_QR,
        }
    }


# ── Query (non-streaming) ─────────────────────────────────────────────────────

@app.post("/query")
@limiter.limit("30/minute")
async def query_rag(req: QueryRequest, request: Request):
    """
    Pipeline RAG đầy đủ:
    1. Semantic Cache check
    2. Multi-Agent Router
    3. (Optional) Query Rewriting
    4. Hybrid Search + Rerank
    5. (Optional) Dynamic Web Fallback
    6. LLM Generation
    7. Cache + Conversation store
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")

    session_id = req.session_id or str(uuid.uuid4())

    # ── Step 0: Semantic Cache ──────────────────────────────────────────
    if ENABLE_CACHE:
        cache = get_cache(
            threshold=float(os.getenv("CACHE_THRESHOLD", "0.92")),
        )
        q_vec = embed_texts([question], is_query=True)[0]
        cached = cache.get(q_vec)
        if cached:
            save_message(session_id, "user", question)
            save_message(session_id, "assistant", cached["answer"])
            return {
                "answer": cached["answer"],
                "sources": cached["sources"],
                "agent": cached.get("metadata", {}).get("agent", "cache"),
                "from_cache": True,
                "session_id": session_id,
            }

    # ── Step 1: Router ──────────────────────────────────────────────────
    agent = route_question(question)

    # ── Step 2: Query Rewriting (optional) ──────────────────────────────
    if ENABLE_QR:
        variants = await generate_query_variants(question, n_variants=3)
    else:
        variants = [question]

    # ── Step 3: Hybrid Search + Rerank ──────────────────────────────────
    all_results = []
    for v in variants:
        result = agent_search(v, agent)
        all_results.append(result)

    if len(all_results) > 1:
        merged = merge_search_results(all_results, top_k=agent.top_k * 2)
        contexts = merged["contexts"][:agent.top_k]
        sources  = merged["sources"][:agent.top_k]
    else:
        contexts = all_results[0]["contexts"]
        sources  = all_results[0]["sources"]

    db_status = all_results[0].get("db_status", "ok")

    # ── Step 4: Dynamic Web Fallback ────────────────────────────────────
    triggered_web = False
    max_score = all_results[0].get("max_score", 0.0)
    WEB_TRIGGER_THRESHOLD = 10.0

    if ENABLE_WEB and (not contexts or max_score < WEB_TRIGGER_THRESHOLD):
        print(f"[Main] Kích hoạt Tavily Fallback (score={max_score:.2f})")
        try:
            web_results = search_tavily(question, max_results=5)
            if web_results:
                contexts = [r["content"] for r in web_results]
                sources  = [f"[Web] {r['url']}" for r in web_results]
                triggered_web = True
        except Exception as e:
            print(f"[Main] Tavily failed: {e}")

    if not contexts:
        return {
            "answer": "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu.",
            "sources": [],
            "agent": agent.agent_id,
            "session_id": session_id,
        }

    # ── Step 5: History ─────────────────────────────────────────────────
    history = get_history(session_id, limit=10)

    # ── Step 6: LLM Generation ──────────────────────────────────────────
    messages = build_agent_prompt(question, contexts, sources, agent, history, db_status)
    answer = call_llm(messages, temperature=0.2, max_tokens=1500)

    # ── Step 7: Save ────────────────────────────────────────────────────
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)

    # Cache the result
    if ENABLE_CACHE:
        cache.set(q_vec, answer, sources, metadata={"agent": agent.agent_id})

    return {
        "answer": answer,
        "sources": sources,
        "agent": get_agent_display(agent),
        "agent_id": agent.agent_id,
        "from_cache": False,
        "triggered_web": triggered_web,
        "session_id": session_id,
    }


# ── Query (SSE streaming) ─────────────────────────────────────────────────────

@app.api_route("/query/stream", methods=["GET", "POST"])
@limiter.limit("20/minute")
async def query_rag_stream(request: Request, req: Optional[QueryRequest] = None):
    """SSE streaming version của /query. Hỗ trợ cả GET và POST."""
    if request.method == "GET":
        question = request.query_params.get("question", "").strip()
        session_id = request.query_params.get("session_id", str(uuid.uuid4()))
        top_k = int(request.query_params.get("top_k", 5))
    else:
        # POST
        if not req:
            body = await request.json()
            req = QueryRequest(**body)
        question = req.question.strip()
        session_id = req.session_id or str(uuid.uuid4())
        top_k = req.top_k

    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")

    # Router
    agent = route_question(question)

    # Search
    search_results = agent_search(question, agent)
    contexts = search_results["contexts"]
    sources  = search_results["sources"]
    db_status = search_results.get("db_status", "ok")

    # Web fallback
    max_score = search_results.get("max_score", 0.0)
    if ENABLE_WEB and (not contexts or max_score < 10.0):
        try:
            web_results = search_tavily(question, max_results=5)
            if web_results:
                contexts = [r["content"] for r in web_results]
                sources  = [f"[Web] {r['url']}" for r in web_results]
        except:
            pass

        async def _empty():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Không tìm thấy thông tin'})}\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    # Build prompt
    history = get_history(session_id, limit=10)
    messages = build_agent_prompt(question, contexts, sources, agent, history, db_status)

    # Stream
    async def _stream():
        full_answer = ""
        # Send metadata first
        meta = {
            "type": "metadata",
            "agent": get_agent_display(agent),
            "sources": sources,
            "session_id": session_id
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        async for chunk in stream_llm(messages, temperature=0.2, max_tokens=1500):
            full_answer += chunk
            yield f"data: {json.dumps({'type': 'token', 'token': chunk}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'agent': get_agent_display(agent), 'sources': sources})}\n\n"

        # Save to history
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", full_answer)

        # Cache
        if ENABLE_CACHE:
            q_vec = embed_texts([question], is_query=True)[0]
            cache = get_cache()
            cache.set(q_vec, full_answer, sources, metadata={"agent": agent.agent_id})

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Ingest PDF ─────────────────────────────────────────────────────────────────

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload & ingest file PDF/DOCX vào Qdrant."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file")

    # Save to temp
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="uploads") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = load_and_chunk_pdf(tmp_path)
        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts)

        store = get_storage()
        ids = [str(uuid.uuid4()) for _ in texts]
        payloads = [
            {"text": t, "source": file.filename, "source_tag": "uploaded_pdf", "page": c.get("page", 1)}
            for t, c in zip(texts, chunks)
        ]
        store.upsert(ids, vectors, payloads)

        return {"status": "ok", "filename": file.filename, "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


# ── Ingest URL ─────────────────────────────────────────────────────────────────

@app.post("/ingest/url")
async def ingest_url(req: IngestURLRequest):
    """Crawl URL & ingest nội dung vào Qdrant."""
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url'")

    text = fetch_and_parse(url)
    if not text or len(text.strip()) < 50:
        raise HTTPException(status_code=422, detail=f"Không thể trích xuất nội dung từ URL: {url}")

    # Chunk
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core import Document
    from llama_index.core.schema import MediaResource

    splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
    doc = Document(
        id_=url,
        text_resource=MediaResource(text=text),
        metadata={"source": url, "type": "web"},
    )
    nodes = splitter.get_nodes_from_documents([doc])
    texts = [n.get_content() for n in nodes]

    if not texts:
        raise HTTPException(status_code=422, detail="Không tạo được chunk từ nội dung URL.")

    vectors = embed_texts(texts)
    store = get_storage()
    ids = [str(uuid.uuid4()) for _ in texts]
    payloads = [
        {"text": t, "source": url, "source_tag": req.source_tag}
        for t in texts
    ]
    store.upsert(ids, vectors, payloads)

    return {"status": "ok", "url": url, "chunks": len(texts)}



# ── Cache management ───────────────────────────────────────────────────────────

@app.get("/cache/stats")
async def cache_stats():
    """Thống kê Semantic Cache."""
    if not ENABLE_CACHE:
        return {"enabled": False}
    cache = get_cache()
    stats = cache.stats()
    return {"enabled": True, **stats}


@app.post("/cache/clear")
async def cache_clear():
    """Xóa toàn bộ Semantic Cache."""
    if ENABLE_CACHE:
        cache = get_cache()
        cache.invalidate_all()
    return {"status": "cleared"}


# ── Conversation management ────────────────────────────────────────────────────

@app.get("/conversations")
async def get_conversations():
    """Liệt kê các session hội thoại."""
    sessions = list_sessions()
    return {"sessions": sessions}


@app.get("/conversations/{session_id}")
async def get_conversation(session_id: str):
    """Lấy lịch sử hội thoại theo session."""
    history = get_history(session_id, limit=50)
    return {"session_id": session_id, "messages": history}


@app.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str):
    """Xóa một session hội thoại."""
    delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}
