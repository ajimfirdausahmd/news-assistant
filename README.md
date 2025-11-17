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

---

## 2. Setup Instructions

### 2.1 Prerequisites

- Python 3.12+
- pip
- An OpenAI API key
- A Tavily API key

### 2.2 Create and activate virtual environment

    python -m venv venv
    venv\Scripts\activate   

### 2.3 Install dependencies

    pip install -r requirements.txt

### 2.4 Environment variables

Create a .env file in the project root:

    OPENAI_API_KEY=your_openai_key_here
    TAVILY_API_KEY=your_tavily_key_here

### 2.5 Build the vector store (RAG index)


    python -m src.rag.build_index


This will create:
- data/vectorstore/index.faiss
- data/vectorstore/news_metadata.json

---

## 3. Running the Application

Option 1 – Streamlit UI (recommended)

    streamlit run app/streamlit_app.py

Then open the displayed URL (e.g. http://localhost:8501).

Option 2 – CLI sanity test

    python test_graph.py


This will:
- Call the LangGraph with several sample questions
- Show which route was used (`rag`,`stats`,`web_search`)
- Print the answers to the console

---

## 4. Project Structure

    news-assistant/
    ├── app/
    │   └── streamlit_app.py               # Streamlit frontend
    │
    ├── src/
    │   ├── data/
    │   │   └── news.csv                   # Raw dataset
    │   │
    │   ├── models/
    │   │   ├── llm_client.py              # OpenAI/Gemini model wrapper
    │   │   └── embeddings_client.py       # Embedding model wrapper
    │   │
    │   ├── rag/
    │   │   ├── loader.py
    │   │   ├── build_index.py
    │   │   ├── retriever.py
    │   │   ├── embedder.py
    │   │   └── indexer.py
    │   │
    │   ├── tools/
    │   │   └── tavily_client.py          # Web search API wrapper
    │   │
    │   ├── graph/
    │   │   ├── state.py                  # Graph state definition
    │   │   ├── graph_builder.py          # LangGraph wiring
    │   │   └── nodes/
    │   │       ├── router_node.py
    │   │       ├── retrieve_node.py
    │   │       ├── generate_node.py
    │   │       ├── stats_node.py
    │   │       └── websearch_node.py
    │
    ├── data/
    │   └── vectorstore/
    │       ├── index.faiss
    │       └── news_metadata.json
    │
    ├── tests/
    │   └── test_graph.py                 # Script to validate your graph pipeline
    │
    ├── requirements.txt
    ├── .env.example                      # Sample env file (no real keys)
    └── README.md


## 5. Key Components

-  `src/models/llm_client.py`
Wraps the OpenAI chat model used by router, generator, and web search summarisation.

- `src/rag/build_index.py`
Loads news.csv, builds documents, and creates a FAISS index + metadata.

- `src/rag/retriever.py`
Provides retrieve_top_k(query, k) to search the FAISS index.

- `src/graph/nodes/router_node.py`
Uses LLM to route each question to rag, stats, or web_search.

- `src/graph/nodes/stats_node.py`
Uses pandas to compute sentiment counts and date-based stats.

- `src/graph/nodes/websearch_node.py`
Calls Tavily, then passes results to the LLM for summarisation.

- `app/streamlit_app.py`
User-facing chat interface.

---

## 6. Limitations / Future Work

Stats node currently supports:

- Sentiment counts
- A simple date filter (“before June 2025”)

Could be extended with:

- More flexible date queries
- Filtering by author, source, or sentiment via natural language
- Per-article citation display in the UI

## 7. Evaluation & improvement section

Evaluate:
- Manual checklist with the 3–4 test questions.
- LLM-as-judge for factual accuracy given context.
- Retrieval metrics: proportion of queries where at least one gold article is in top-k.

Improve:
- Better chunking + titles in embeddings.
- Add reranking (e.g. cross-encoder or similarity re-rank).
- Add retrieval grader (LLM filter) for irrelevant docs.
- Cache web search + answers for popular queries.