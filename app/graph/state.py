from typing import TypedDict, List, Dict, Any


class RAGState(TypedDict, total=False):
    question: str
    rewritten_query: str
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]
    needs_rewrite: bool