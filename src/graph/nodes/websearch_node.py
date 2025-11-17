from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from src.tools.tavily_client import tavily_search
from src.models.llm_client import get_llm

llm = get_llm()


def websearch_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("question", "").strip()
    if not question:
        state["answer"] = "No question provided."
        return state

    results: List[dict] = tavily_search(question)

    if not results:
        state["answer"] = "I could not find relevant web results."
        return state

    snippets = []
    source_lines = []
    for r in results[:5]:
        title = r.get("title", "") or "Untitled"
        url = r.get("url", "") or ""
        content = (r.get("content") or "")[:900]
        source = r.get("source") or ""

        snippets.append(
            f"Title: {title}\nSource: {source}\nURL: {url}\nContent:\n{content}"
        )
        source_lines.append(f"- {title} ({source}) — {url}")

    context = "\n\n---\n\n".join(snippets)
    sources_block = "\n".join(source_lines)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant summarising up-to-date WEB SEARCH results. "
                "Use ONLY the web snippets provided. "
                "Answer clearly and concisely.",
            ),
            (
                "human",
                "Question:\n{question}\n\nWeb results:\n{context}",
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    body = getattr(response, "content", str(response))

    # Explicitly mark this as web search + list sources
    final_answer = (
        "(Web search result)\n\n"
        f"{body.strip()}\n\n"
        "Sources (web):\n"
        f"{sources_block}"
    )

    state["answer"] = final_answer
    return state