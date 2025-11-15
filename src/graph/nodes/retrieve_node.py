from typing import Dict, Any
from src.rag.retriever import retrieve_top_k

def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("question", "").strip()
    if not question:
        state["rag_docs"] = []
        return state

    docs = retrieve_top_k(question, k=4)

    # docs is a list of dicts: {"text": ..., "metadata": {...}, "score": ...}
    state["rag_docs"] = docs
    return state