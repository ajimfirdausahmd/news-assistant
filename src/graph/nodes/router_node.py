from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.models.llm_client import get_llm

# LLM used only for routing
router_llm = get_llm(model_name="gpt-4o-mini", temperature=0.0)

ROUTE_OPTIONS = ["rag", "stats", "web_search"]


def build_router_prompt(question: str) -> str:
    return f"""
You are a routing assistant for an AI-powered news QA system.

You MUST choose exactly ONE of the following routes for the user's question:

1. "rag"
   - Use this when the question can be answered from the INTERNAL NEWS DATASET
     using Retrieval-Augmented Generation (vector store over news.csv).
   - Examples:
     - "What are some initiatives launched by MCMC?"
     - "Adakah SSM terbabit dengan kes-kes mahkamah?"
     - "What issues have been reported about SSM?"

2. "stats"
   - Use this when the user is asking about STATISTICS over the dataset.
   - This includes counts, totals, percentages, distributions, or filters by date,
     sentiment, or author.
   - Examples:
     - "How many positive news articles are there?"
     - "How many of the news are before June 2025?"
     - "Berapa banyak berita negatif mengenai ekonomi?"
     - "How many articles are written by The Star?"

3. "web_search"
   - Use this when the question clearly needs EXTERNAL, UP-TO-DATE information
     that may not exist in the internal dataset.
   - Examples:
     - "What is the status of the Malaysian economy in 2025 and associated headwinds?"
     - "Who won the Cricket World Cup 2023?"
     - "Latest news about US interest rates?"

If the question could be answered by both "rag" and "web_search":
- Prefer "rag" if it seems likely to be covered by the Malaysian news dataset.
- Prefer "web_search" if it is clearly global, very recent, or generic.

USER QUESTION:
\"\"\"{question}\"\"\"

Reply with EXACTLY ONE WORD (lowercase), one of:
rag
stats
web_search

Do NOT add explanations, punctuation, or any other text.
""".strip()


def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: decides which route to take.
    Sets state["route"] to one of: "rag", "stats", "web_search".
    """
    question = (state.get("question") or "").strip()

    # fallback default
    if not question:
        state["route"] = "rag"
        return state

    # build prompt text
    prompt_text = build_router_prompt(question)

    # LangChain prompt -> LLM chain
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict routing assistant. Follow the instructions exactly.",
            ),
            ("human", prompt_text),
        ]
    )

    chain = prompt | router_llm
    response = chain.invoke({})

    raw_output = response.content if hasattr(response, "content") else str(response)
    route = raw_output.strip().lower()

    if route not in ROUTE_OPTIONS:
        route = "rag"

    state["route"] = route
    # Optional: clear these when routing
    state["rag_docs"] = None
    state["answer"] = None

    return state
