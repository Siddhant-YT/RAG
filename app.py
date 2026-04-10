"""
app.py
------
Main entry point for the RAG Studio Streamlit application.
Sets up page config, global styles, sidebar, and renders the selected tab.

Run with:
    streamlit run app.py

Tabs:
    1. Knowledge Base    - Upload docs, choose chunking, build FAISS + BM25
    2. Chunking Comparison - Compare all 4 strategies side by side
    3. RAG Patterns      - Simple, HyDE, Multi-Query, Branched, Citations
    4. Hybrid Search     - Dense vs BM25 vs Hybrid + Reranking
    5. Multimodal RAG    - Image + text RAG using Gemini Vision
    6. Chatbot           - Conversational RAG with memory
    7. Evaluation        - RAGAS + DeepEval metrics
    8. Learning          - Reserved for future content
"""

import streamlit as st
from utils.helpers import init_session_state

# ---- Page configuration ----
st.set_page_config(
    page_title="RAG Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Initialize session state ----
# Must be called before any tab renders so all state keys exist
init_session_state()

# ---- Sidebar ----
with st.sidebar:
    st.title("RAG Studio")
    st.caption("A hands-on learning environment for Retrieval-Augmented Generation.")

    st.divider()

    # Navigation
    tab_options = [
        "Knowledge Base",
        "Chunking Comparison",
        "RAG Patterns",
        "Hybrid Search & Reranking",
        "Multimodal RAG",
        "Chatbot",
        "Evaluation",
        "Learning",
    ]

    selected_tab = st.radio(
        "Navigate",
        tab_options,
        label_visibility="collapsed",
    )

    st.divider()

    # Global status indicator
    kb_status = "Ready" if st.session_state.get("kb_ready") else "Not built"
    st.write(f"Knowledge Base: **{kb_status}**")

    if st.session_state.get("kb_ready"):
        chunks = st.session_state.get("chunks", [])
        docs = st.session_state.get("documents", [])
        st.write(f"Documents: {len(docs)} | Chunks: {len(chunks)}")

    eval_count = len(st.session_state.get("eval_samples", []))
    chat_count = len(st.session_state.get("chat_history", []))
    st.write(f"Chat turns: {chat_count} | Eval queue: {eval_count}")

    st.divider()

    st.caption(
        "Models used:\n"
        "- LLM: Groq llama-3.1-8b-instant\n"
        "- Embeddings: all-MiniLM-L6-v2 (local)\n"
        "- Multimodal/Eval: gemini-3.1-flash-lite-preview\n"
        "- Reranker: ms-marco-MiniLM-L-6-v2 (local)"
    )

# ---- Tab rendering ----
# Each tab is a separate module in the app/ folder.
# Only the selected tab is imported and rendered, keeping startup fast.

if selected_tab == "Knowledge Base":
    from app.tab_knowledge_base import render
    render()

elif selected_tab == "Chunking Comparison":
    from app.tab_chunking_comparison import render
    render()

elif selected_tab == "RAG Patterns":
    from app.tab_rag_patterns import render
    render()

elif selected_tab == "Hybrid Search & Reranking":
    from app.tab_retrieval import render
    render()

elif selected_tab == "Multimodal RAG":
    from app.tab_multimodal import render
    render()

elif selected_tab == "Chatbot":
    from app.tab_chatbot import render
    render()

elif selected_tab == "Evaluation":
    from app.tab_evaluation import render
    render()

elif selected_tab == "Learning":
    from app.tab_learning import render
    render()
