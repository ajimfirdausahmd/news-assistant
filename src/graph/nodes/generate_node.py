from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from src.models.llm_client import get_llm

llm = get_llm() 

def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("question", "")
    docs: List[Dict[str, Any]] = state.get("rag_docs") or []

    # If retrieval failed or returned nothing
    if not docs:
        state["answer"] = "Not found in internal dataset."
        return state

    # Build context from retrieved docs
    context_parts = []
    for d in docs:
        meta = d.get("metadata", {})
        title = meta.get("title") or ""
        source = meta.get("author") or meta.get("source") or "Unknown source"
        text = d.get("text", "")

        snippet = text[:1000]
        context_parts.append(
            f"Title: {title}\nSource: {source}\nContent:\n{snippet}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Malaysian news assistant. "
                "Answer ONLY using the context. "
                "If the answer is not in the context, reply exactly: "
                "\"Not found in internal dataset.\"",
            ),
            (
                "human",
                "Question:\n{question}\n\nContext:\n{context}",
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})

    state["answer"] = getattr(response, "content", str(response))
    return state