import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pydantic import ValidationError
from custome_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult, RAGQueryWithSession

def test_rag_chunk_and_src_valid():
    model = RAGChunkAndSrc(chunks=["a", "b"], source_id="123", metadata=[{"page": 1, "type": "pdf"}])
    assert model.chunks == ["a", "b"]
    assert model.source_id == "123"

def test_rag_chunk_and_src_invalid_chunks():
    with pytest.raises(ValidationError):
        # chunks expects a list of strings
        RAGChunkAndSrc(chunks="not a list")

def test_rag_upsert_result():
    model = RAGUpsertResult(ingested=10)
    assert model.ingested == 10

def test_rag_search_result():
    model = RAGSearchResult(contexts=["c1", "c2"], sources=["s1", "s2"])
    assert len(model.contexts) == 2
    assert model.sources == ["s1", "s2"]

def test_rag_query_result():
    model = RAGQueryResult(answer="Success", sources=["src1"], num_contexts=1)
    assert model.answer == "Success"
    assert model.num_contexts == 1

def test_rag_query_with_session():
    model = RAGQueryWithSession(question="Học phí visa Mỹ?", session_id="abc-123")
    assert model.question == "Học phí visa Mỹ?"
    assert model.session_id == "abc-123"
    assert model.top_k == 5  # Test default value
