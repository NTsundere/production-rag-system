from typing import List, Dict, Any
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from app.config import settings
from app.ingestion.indexers import get_embedder
from app.logger import setup_logger

logger = setup_logger(__name__)


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    sources: Dict[str, str] = {}

    for rank, doc in enumerate(dense_results):
        cid = doc["content"]
        scores[cid] = scores.get(cid, 0) + settings.dense_weight / (k + rank + 1)
        sources[cid] = doc.get("source", "unknown")

    for rank, doc in enumerate(sparse_results):
        cid = doc["content"]
        scores[cid] = scores.get(cid, 0) + settings.bm25_weight / (k + rank + 1)
        sources.setdefault(cid, doc.get("source", "unknown"))

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for content, score in sorted_docs[: settings.top_k_retrieval]:
        results.append({
            "content": content,
            "source": sources.get(content, "unknown"),
            "score": score,
        })
    return results


def hybrid_search(query: str) -> List[Dict[str, Any]]:
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        check_compatibility=False,
    )
    embedder = get_embedder()

    query_vector = embedder.embed_query(query)
    dense_response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=settings.top_k_retrieval,
    )
    dense_results = [
        {"content": h.payload["content"], "source": h.payload["source"], "score": h.score}
        for h in dense_response.points
    ]

    all_points = client.scroll(
        collection_name=settings.qdrant_collection, limit=10000
    )[0]
    corpus = [p.payload["content"] for p in all_points]
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[: settings.top_k_retrieval]
    sparse_results = [
        {
            "content": corpus[i],
            "source": all_points[i].payload["source"],
            "score": float(scores[i]),
        }
        for i in top_indices
    ]

    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    logger.info(f"Hybrid search: {len(fused)} results for query: {query[:50]}...")
    return fused