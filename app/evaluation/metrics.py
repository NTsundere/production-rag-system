from typing import List, Dict, Any
from app.logger import setup_logger

logger = setup_logger(__name__)


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    top_k = set(retrieved[:k])
    return len(top_k & set(relevant)) / len(relevant) if relevant else 0.0


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    top_k = retrieved[:k]
    return sum(1 for d in top_k if d in relevant) / k if k > 0 else 0.0


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(eval_results: List[Dict[str, Any]], k: int = 5) -> Dict[str, float]:
    recalls, precisions, mrrs = [], [], []
    for item in eval_results:
        retrieved = item["retrieved"]
        relevant = item["relevant"]
        recalls.append(recall_at_k(retrieved, relevant, k))
        precisions.append(precision_at_k(retrieved, relevant, k))
        mrrs.append(mrr(retrieved, relevant))

    n = len(eval_results)
    return {
        f"recall@{k}": sum(recalls) / n,
        f"precision@{k}": sum(precisions) / n,
        "mrr": sum(mrrs) / n,
    }