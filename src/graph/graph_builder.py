from langgraph.graph import StateGraph, END

from src.graph.state import GraphState
from src.graph.nodes.router_node import router_node
from src.graph.nodes.retrieve_node import retrieve_node
from src.graph.nodes.generate_node import generate_node
from src.graph.nodes.stats_node import stats_node
from src.graph.nodes.websearch_node import websearch_node

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("stats", stats_node)
    graph.add_node("web_search", websearch_node)

    # start → router
    graph.set_entry_point("router")

    # router decides which path
    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {
            "rag": "retrieve",
            "stats": "stats",
            "web_search": "web_search",
        },
    )

    # RAG path
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # Stats path ends here
    graph.add_edge("stats", END)

    # Web search path ends here (websearch_node must set state["answer"])
    graph.add_edge("web_search", END)

    return graph.compile()