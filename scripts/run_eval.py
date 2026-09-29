import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from typing import List, Dict, Any

from app.graph.graph import rag_graph
from app.retrieval.hybrid_search import hybrid_search
from app.evaluation.golden_set import GOLDEN_SET
from app.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
)
from app.logger import setup_logger

logger = setup_logger(__name__)


async def run_evaluation():
    retrieval_results: List[Dict[str, Any]] = []
    ragas_rows: List[Dict[str, Any]] = []

    for item in GOLDEN_SET:
        logger.info(f"Running query: {item['question']}")

        # 1) Метрики retrieval (без LLM)
        retrieved_docs = hybrid_search(item["question"])
        retrieved_sources = [d["source"] for d in retrieved_docs]
        relevant = item.get("relevant_sources", [])

        retrieval_results.append({
            "retrieved": retrieved_sources,
            "relevant": relevant,
        })

        # 2) Полный RAG-запрос (для проверки ответа)
        state = await rag_graph.ainvoke({"question": item["question"]})
        answer = state.get("answer", "")
        contexts = [d["content"] for d in state.get("reranked_docs", [])]

        ragas_rows.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })

        logger.info(f"Q: {item['question']}")
        logger.info(f"A: {answer[:200]}...")

    # 3) Считаем retrieval-метрики
    k = 5
    recalls = [recall_at_k(r["retrieved"], r["relevant"], k) for r in retrieval_results]
    precisions = [precision_at_k(r["retrieved"], r["relevant"], k) for r in retrieval_results]
    mrrs = [mrr(r["retrieved"], r["relevant"]) for r in retrieval_results]

    print("\n" + "=" * 60)
    print("RETRIEVAL METRICS (no LLM judge needed)")
    print("=" * 60)
    print(f"Recall@{k}   : {sum(recalls) / len(recalls):.4f}")
    print(f"Precision@{k}: {sum(precisions) / len(precisions):.4f}")
    print(f"MRR         : {sum(mrrs) / len(mrrs):.4f}")

    # 4) Краткий отчёт по ответам (без LLM-судьи)
    print("\n" + "=" * 60)
    print("ANSWERS (for manual review)")
    print("=" * 60)
    for row in ragas_rows:
        print(f"\nQ: {row['question']}")
        print(f"Ground truth: {row['ground_truth']}")
        print(f"Model answer: {row['answer'][:300]}...")
        print(f"Sources used: {len(row['contexts'])} chunks")

    return {
        "retrieval": {
            f"recall@{k}": sum(recalls) / len(recalls),
            f"precision@{k}": sum(precisions) / len(precisions),
            "mrr": sum(mrrs) / len(mrrs),
        },
        "answers": ragas_rows,
    }


if __name__ == "__main__":
    asyncio.run(run_evaluation())