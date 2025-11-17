from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from src.models.llm_client import get_llm

llm = get_llm()


def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("question", "")
    docs: List[Dict[str, Any]] = state.get("rag_docs") or []


    if not docs:
        state["answer"] = "Not found in internal dataset."
        return state

    context_parts = []
    source_lines = []
    for d in docs:
        meta = d.get("metadata", {}) or {}
        title = meta.get("title") or ""
        source = meta.get("author") or meta.get("source") or "Unknown source"
        text = d.get("text", "")

        snippet = text[:1000]
        context_parts.append(
            f"Title: {title}\nSource: {source}\nContent:\n{snippet}"
        )
        source_lines.append(f"- {title} — {source}")

    context = "\n\n---\n\n".join(context_parts)
    sources_block = "\n".join(source_lines) if source_lines else "No sources."

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Malaysian news assistant using ONLY the INTERNAL NEWS DATASET. "
                "Use only the provided context to answer. "
                "If the answer is not in the context, reply exactly: "
                "\"Not found in internal dataset.\"",
            ),
            (
                "human",
                "Question:\n{question}\n\nContext (internal news articles):\n{context}",
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    body = getattr(response, "content", str(response)).strip()

    # If model says not found, don't append sources
    if body == "Not found in internal dataset.":
        final_answer = body
    else:
        final_answer = (
            f"{body}\n\n"
            "Sources (internal):\n"
            f"{sources_block}"
        )

    state["answer"] = final_answer
    return state