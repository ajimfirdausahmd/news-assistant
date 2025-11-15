import streamlit as st

from src.graph.graph_builder import build_graph

graph = build_graph()

st.set_page_config(page_title="News Assistant", page_icon="📰")
st.title("📰 News Assistant (RAG + Stats + Web Search)")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    role = msg["role"]
    if role == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")

user_input = st.chat_input("Ask about news, stats, or economy...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    result = graph.invoke({"question": user_input})
    answer = result.get("answer", "Sorry, I could not generate an answer.")

    st.session_state["messages"].append({"role": "assistant", "content": answer})

    st.rerun()
    