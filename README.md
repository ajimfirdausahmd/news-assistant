### News Assistant - RAG + Stats + Web Search

This project implements an AI-powered news assistant that can:
- Answer questions from an **internal news knowledge base** (RAG over `news.csv`).
- Provide **statistics** over the news dataset (e.g., sentiment counts, date filters).
- Use **web search** for up-to-date, external news (via Tavily).
- Route each user question automatically to the best path: `RAG`, `STATS`, or `WEB_SEARCH`.

The app uses:
- **LangChain + LangGraph** for agentic routing and orchestration
- **FAISS + sentence-transformers** for vector search over internal news
- **OpenAI** for LLM generation
- **Tavily** for web search
- **Streamlit** for the chat UI

---

## 1. Features

- 🧠 **RAG over internal news**  
  - Embeds `src/data/news.csv` into FAISS.
  - Retrieves top-k relevant articles and generates grounded answers.

- 📊 **Dataset statistics** (`stats` route)  
  - Sentiment distribution: positive / negative / other.  
  - Example: “How many positive and negative news are there?”  
  - Date filter: “How many news are before June 2025?”

- 🌐 **Web search** (`web_search` route)  
  - For questions that need external & up-to-date info.  
  - Example: “What is the status of the Malaysian economy in 2025?”

- 🧭 **Routing with LangGraph**  
  - `router_node` decides between:
    - `rag` → internal vector store
    - `stats` → pandas over `news.csv`
    - `web_search` → Tavily
  - `generate_node` creates the final answer.

- 💬 **Streamlit chat UI**  
  - Simple chat interface.
  - Shows assistant messages and which route was used.
