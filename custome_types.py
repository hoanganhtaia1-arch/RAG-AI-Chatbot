import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None
    metadata: list[dict] = []   # [{"page": int, "type": str}, ...]

class RAGUpsertResult(pydantic.BaseModel):
    ingested:int

class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

class RAGQueryResult(pydantic.BaseModel):
    answer:str
    sources: list[str]
    num_contexts:int

class RAGQueryWithSession(pydantic.BaseModel):
    question: str
    session_id: str
    top_k: int = 5