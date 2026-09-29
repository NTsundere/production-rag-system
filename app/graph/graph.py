from langgraph.graph import StateGraph, END
from app.graph.state import RAGState
from app.graph.nodes import (
    rewrite_query,
    retrieve,
    rerank_documents,
    generate_answer,
    decide_rewrite,
)


def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank_documents)
    workflow.add_node("generate", generate_answer)

    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        decide_rewrite,
        {
            "rewrite": "rewrite",
            "generate": "rerank",
        },
    )
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


rag_graph = build_graph()