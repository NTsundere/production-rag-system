from typing import List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceChunk(BaseModel):
    content: str
    source: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    latency_ms: float
    model: str


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_created: int
    collection: str