from app.llm import get_llm
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.rerankers import rerank
from app.logger import setup_logger
from app.graph.state import RAGState

logger = setup_logger(__name__)


def rewrite_query(state: RAGState) -> dict:
    llm = get_llm()
    prompt = (
        "Rewrite this question for better document retrieval. "
        "Return only the rewritten question.\n"
        f"Question: {state['question']}"
    )
    result = llm.invoke(prompt)
    return {"rewritten_query": result.content.strip()}


def retrieve(state: RAGState) -> dict:
    query = state.get("rewritten_query") or state["question"]
    docs = hybrid_search(query)
    return {"retrieved_docs": docs}



def rerank_documents(state: RAGState) -> dict:
    query = state.get("rewritten_query") or state["question"]
    reranked = rerank(query, state["retrieved_docs"])
    return {"reranked_docs": reranked}


def generate_answer(state: RAGState) -> dict:
    llm = get_llm()
    context = "\n\n---\n\n".join(
        f"[Source: {d['source']}]\n{d['content']}" for d in state["reranked_docs"]
    )
    prompt = (
        "Answer the question based ONLY on the context below. "
        "If the answer is not in the context, say \"I don't know\".\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['question']}\n\n"
        "Answer:"
    )
    result = llm.invoke(prompt)
    sources = [
        {"content": d["content"][:200], "source": d["source"], "score": d.get("rerank_score", 0)}
        for d in state["reranked_docs"]
    ]
    return {"answer": result.content.strip(), "sources": sources}


def decide_rewrite(state: RAGState) -> str:
    """Если документов нет — переписываем запрос. Если есть — сразу генерируем ответ."""
    if not state.get("retrieved_docs"):
        logger.info("No documents retrieved, rewriting query...")
        return "rewrite"
    return "generate"