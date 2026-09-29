import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, QueryResponse, IngestResponse
from app.graph.graph import rag_graph
from app.config import settings
from app.logger import setup_logger
from app.ingestion.loaders import load_document
from app.ingestion.chunkers import chunk_text
from app.ingestion.indexers import index_chunks

logger = setup_logger(__name__)

app = FastAPI(
    title="Production RAG System",
    description="RAG with hybrid search, reranking, LangGraph agents, and RAGAS evaluation",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return {"status": "healthy", "collection": settings.qdrant_collection}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start = time.perf_counter()
    try:
        state = await rag_graph.ainvoke({"question": request.question})
        latency = (time.perf_counter() - start) * 1000
        logger.info(f"Query: {request.question[:50]}... | Latency: {latency:.0f}ms")
        return QueryResponse(
            answer=state["answer"],
            sources=state.get("sources", []),
            latency_ms=round(latency, 2),
            model=settings.gemini_model,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    text = load_document(path)
    chunks = chunk_text(text)
    count = index_chunks(chunks, source=path.name)
    return IngestResponse(
        documents_ingested=1,
        chunks_created=count,
        collection=settings.qdrant_collection,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)