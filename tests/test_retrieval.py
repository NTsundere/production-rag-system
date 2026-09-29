from app.retrieval.hybrid_search import reciprocal_rank_fusion
from app.evaluation.metrics import recall_at_k, precision_at_k, mrr

def test_rrf():
    dense = [{"content": "a", "score": 0.9}, {"content": "b", "score": 0.8}]
    sparse = [{"content": "b", "score": 5.0}, {"content": "c", "score": 4.0}]
    result = reciprocal_rank_fusion(dense, sparse)
    assert len(result) > 0

def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "d"], 3) == 0.5

def test_mrr():
    assert mrr(["x", "a", "b"], ["a"]) == 0.5
    assert mrr(["a", "b"], ["a"]) == 1.0