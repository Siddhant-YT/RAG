"""
app/tab_rag_patterns.py
------------------------
Tab 3: RAG Patterns
Lets the user pick a RAG pattern and run it against the knowledge base.
Patterns covered:
- Simple RAG
- RAG with Citations
- HyDE (Hypothetical Document Embeddings)
- Multi-Query RAG
- Branched RAG (Query Routing)
Each pattern shows its answer, the retrieved documents, and any intermediate output.
"""

import streamlit as st

from core.models import load_groq_llm, load_embedding_model
from core.rag_engine import (
    simple_rag,
    rag_with_citations,
    hyde_rag,
    multi_query_rag,
    branched_rag,
)
from utils.helpers import display_retrieved_docs


# Pattern metadata — description shown in the UI
PATTERN_INFO = {
    "Simple RAG": {
        "description": (
            "The baseline RAG pattern. Embeds the query, retrieves top-k chunks via cosine "
            "similarity in FAISS, builds context, and sends it to the LLM."
        ),
        "when_to_use": "General-purpose. Start here.",
        "limitation": "Single phrasing of query may miss relevant documents.",
    },
    "RAG with Citations": {
        "description": (
            "Same as Simple RAG but each chunk is labelled with its chunk ID. "
            "The LLM is prompted to cite which chunks support its answer."
        ),
        "when_to_use": "When you need explainability — knowing exactly which source was used.",
        "limitation": "LLM may not always cite correctly.",
    },
    "HyDE": {
        "description": (
            "Hypothetical Document Embeddings. Instead of embedding the raw query, the LLM first "
            "writes a hypothetical answer. That hypothetical answer is embedded and used for retrieval. "
            "Hypothetical answers live closer to real documents in embedding space."
        ),
        "when_to_use": "When query-document semantic gap is large (very different phrasing).",
        "limitation": "Adds one extra LLM call. If the hypothetical doc is poor, retrieval suffers.",
    },
    "Multi-Query RAG": {
        "description": (
            "Generates N alternative phrasings of the query, retrieves documents for each, "
            "deduplicates, then generates a single answer from the merged result set. "
            "Increases recall by covering multiple angles of the same question."
        ),
        "when_to_use": "When a single phrasing might miss relevant documents.",
        "limitation": "Slower — makes N+1 LLM calls and N retrieval passes.",
    },
    "Branched RAG": {
        "description": (
            "Uses an LLM router to classify the query as 'factual', 'analytical', or 'general', "
            "then dispatches to a specialized retrieval pipeline for that type. "
            "Factual -> dense k=2. Analytical -> HyDE. General -> dense k=4."
        ),
        "when_to_use": "Production systems with mixed query types needing different strategies.",
        "limitation": "Router may misclassify edge-case queries.",
    },
}


def render():
    """Render the RAG Patterns tab."""
    st.header("RAG Patterns")
    st.write(
        "Explore different RAG architectural patterns. Each pattern addresses a different limitation "
        "of simple retrieval-augmented generation. Select a pattern, enter a query, and compare results."
    )

    if not st.session_state.get("kb_ready"):
        st.warning("Please build the knowledge base first in the 'Knowledge Base' tab.")
        return

    vectorstore = st.session_state["vectorstore"]
    llm = load_groq_llm()

    # Pattern selector
    selected_pattern = st.selectbox(
        "Select RAG Pattern",
        list(PATTERN_INFO.keys()),
    )

    # Show pattern info
    info = PATTERN_INFO[selected_pattern]
    with st.expander("About this pattern", expanded=True):
        st.write(f"**How it works:** {info['description']}")
        col1, col2 = st.columns(2)
        col1.write(f"**When to use:** {info['when_to_use']}")
        col2.write(f"**Limitation:** {info['limitation']}")

    st.divider()

    # Query input
    query = st.text_area(
        "Enter your query",
        placeholder="e.g. How does RAG reduce hallucination?",
        height=80,
        key="patterns_query",
    )

    # Pattern-specific settings
    k = st.slider("Number of chunks to retrieve (k)", 1, 6, 3, key="patterns_k")

    n_variations = 3
    if selected_pattern == "Multi-Query RAG":
        n_variations = st.slider("Number of query variations", 2, 5, 3)

    run_btn = st.button("Run Pattern", type="primary", disabled=not query.strip())

    if run_btn and query.strip():
        with st.spinner(f"Running {selected_pattern}..."):

            if selected_pattern == "Simple RAG":
                result = simple_rag(query.strip(), vectorstore, llm, k=k)

            elif selected_pattern == "RAG with Citations":
                result = rag_with_citations(query.strip(), vectorstore, llm, k=k)

            elif selected_pattern == "HyDE":
                result = hyde_rag(query.strip(), vectorstore, llm, k=k)

            elif selected_pattern == "Multi-Query RAG":
                result = multi_query_rag(query.strip(), vectorstore, llm,
                                         n_variations=n_variations, k=k)

            elif selected_pattern == "Branched RAG":
                result = branched_rag(query.strip(), vectorstore, llm, k=k)

        st.divider()

        # ---- Show intermediate outputs (pattern-specific) ----

        if selected_pattern == "HyDE" and "hypothetical_doc" in result:
            st.subheader("Hypothetical Document (Generated by LLM)")
            st.write(
                "The LLM wrote this passage as a hypothetical answer. "
                "This was embedded and used for retrieval instead of the raw query."
            )
            st.info(result["hypothetical_doc"])

        if selected_pattern == "Multi-Query RAG" and "query_variations" in result:
            st.subheader("Query Variations Generated")
            st.write("These phrasings were used to retrieve from different angles:")
            for i, q in enumerate(result["query_variations"]):
                label = "Original" if i == 0 else f"Variation {i}"
                st.write(f"- **{label}:** {q}")

        if selected_pattern == "Branched RAG":
            col1, col2 = st.columns(2)
            col1.metric("Query Type Detected", result.get("query_type", "—").capitalize())
            col2.metric("Routing Path", result.get("routing_path", "—"))

        # ---- Main answer ----
        st.subheader("Generated Answer")
        st.write(result.get("answer", "No answer generated."))

        # Optionally add to eval samples for later evaluation
        if result.get("docs"):
            contexts = [d.page_content for d in result.get("docs", [])]
        else:
            contexts = []

        if st.button("Add to Evaluation Queue"):
            st.session_state["eval_samples"].append({
                "question": query.strip(),
                "answer": result.get("answer", ""),
                "contexts": contexts,
                "ground_truth": "",  # User can fill this in the Evaluation tab
            })
            st.success("Added to evaluation queue.")

        # ---- Retrieved documents ----
        st.subheader("Retrieved Context Chunks")
        docs = result.get("docs", [])
        st.write(f"{len(docs)} chunk(s) used for context generation.")
        display_retrieved_docs(docs, collapsed=True)
