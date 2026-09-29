from app.evaluation.metrics import compute_retrieval_metrics

def test_compute_metrics():
    results = [
        {"retrieved": ["a", "b", "c"], "relevant": ["a"]},
        {"retrieved": ["x", "y"], "relevant": ["z"]},
    ]
    metrics = compute_retrieval_metrics(results, k=3)
    assert "recall@3" in metrics
    assert "mrr" in metrics
    assert metrics["recall@3"] == 0.5