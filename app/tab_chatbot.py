"""
app/tab_chatbot.py
-------------------
Tab 6: RAG Chatbot (Conversational RAG with Memory)
A full multi-turn chatbot that:
- Maintains conversation history across turns
- Retrieves relevant context for each new message
- Shows which chunks were used (citations)
- Supports session reset
- Captures QA pairs for the Evaluation tab
"""

import streamlit as st

from core.models import load_groq_llm
from core.rag_engine import conversational_rag
from utils.helpers import display_retrieved_docs


def render():
    """Render the Chatbot tab."""
    st.header("RAG Chatbot")
    st.write(
        "A conversational RAG chatbot that maintains memory across turns. "
        "Ask follow-up questions and the system will use previous exchanges for context. "
        "Answers are grounded in the knowledge base, with citations shown for each response."
    )

    if not st.session_state.get("kb_ready"):
        st.warning("Please build the knowledge base first in the 'Knowledge Base' tab.")
        return

    vectorstore = st.session_state["vectorstore"]
    llm = load_groq_llm()

    # Sidebar-style controls
    col_main, col_controls = st.columns([3, 1])

    with col_controls:
        st.write("**Controls**")
        k = st.slider("Retrieve k chunks", 1, 6, 3, key="chatbot_k")
        max_history = st.slider("Max history turns", 2, 10, 6, key="chatbot_history")

        if st.button("Reset Conversation"):
            st.session_state["chat_history"] = []
            st.rerun()

        st.divider()
        st.write(f"Turns so far: **{len(st.session_state['chat_history'])}**")
        st.write(f"Eval samples queued: **{len(st.session_state.get('eval_samples', []))}**")

    with col_main:
        # Render existing chat history
        for human_msg, ai_msg in st.session_state["chat_history"]:
            with st.chat_message("user"):
                st.write(human_msg)
            with st.chat_message("assistant"):
                st.write(ai_msg)

        # New message input
        user_input = st.chat_input("Ask a question about your documents...")

        if user_input:
            # Show user message immediately
            with st.chat_message("user"):
                st.write(user_input)

            # Run conversational RAG
            with st.chat_message("assistant"):
                with st.spinner("Retrieving and generating..."):
                    result = conversational_rag(
                        query=user_input,
                        vectorstore=vectorstore,
                        llm=llm,
                        chat_history=st.session_state["chat_history"],
                        k=k,
                        max_history=max_history,
                    )

                # Display the answer
                st.write(result["answer"])

                # Show sources in a small expander
                with st.expander(f"Sources ({len(result['docs'])} chunks used)", expanded=False):
                    for i, doc in enumerate(result["docs"]):
                        cid = doc.metadata.get("chunk_id", i)
                        source = doc.metadata.get("source", "unknown")
                        st.caption(f"Chunk {cid} | {source}")
                        st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                        st.divider()

            # Save to conversation history
            st.session_state["chat_history"].append((user_input, result["answer"]))

            # Auto-add to evaluation queue for later RAGAS/DeepEval analysis
            st.session_state["eval_samples"].append({
                "question": user_input,
                "answer": result["answer"],
                "contexts": [d.page_content for d in result["docs"]],
                "ground_truth": "",  # Can be filled in the Evaluation tab
            })

        # Show message count
        if not st.session_state["chat_history"]:
            st.info(
                "Start by asking a question. Try: 'What is FAISS?' or 'How does RAG reduce hallucination?'"
            )
