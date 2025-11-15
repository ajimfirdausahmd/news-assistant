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
    for r in results[:5]:
        title = r.get("title", "")
        url = r.get("url", "")
        content = (r.get("content") or "")[:900]
        snippets.append(f"Title: {title}\nURL: {url}\nContent:\n{content}")

    context = "\n\n---\n\n".join(snippets)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant summarising up-to-date web news. "
                "Use ONLY the web snippets provided. "
                "Mention key points clearly and briefly.",
            ),
            (
                "human",
                "Question:\n{question}\n\nWeb results:\n{context}",
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    state["answer"] = getattr(response, "content", str(response))
    return state