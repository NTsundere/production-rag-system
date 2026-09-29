from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from app.config import settings
from app.logger import setup_logger

logger = setup_logger(__name__)

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not documents:
        return []
    reranker = get_reranker()
    pairs = [(query, doc["content"]) for doc in documents]
    scores = reranker.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
    logger.info(f"Reranked {len(reranked)} docs, top score: {reranked[0]['rerank_score']:.4f}")
    return reranked[: settings.top_k_rerank]