"""
app/tab_chunking_comparison.py
-------------------------------
Tab 2: Chunking Strategy Comparison
Runs all 4 chunking strategies on the same query and displays
answers side by side so the user can compare quality differences.
Also shows the chunk statistics and preview for each strategy.
"""

import streamlit as st

from core.models import load_groq_llm, load_embedding_model
from core.rag_engine import compare_chunking_strategies


def render():
    """Render the Chunking Comparison tab."""
    st.header("Chunking Strategy Comparison")
    st.write(
        "Enter a query below. The system will run all four chunking strategies "
        "independently and show the answer each one produces. "
        "This helps you understand how the choice of chunking affects retrieval and generation quality."
    )

    # Require knowledge base
    if not st.session_state.get("kb_ready"):
        st.warning("Please build the knowledge base first in the 'Knowledge Base' tab.")
        return

    documents = st.session_state["documents"]
    embedding_model = load_embedding_model()
    llm = load_groq_llm()

    # Strategy descriptions for reference
    strategy_info = {
        "Recursive": {
            "description": "Splits on paragraph, sentence, then word boundaries in order. Fixed chunk size with overlap.",
            "best_for": "General-purpose. Most commonly used in production.",
        },
        "Semantic": {
            "description": "Uses embeddings to detect where topic shifts occur and splits there.",
            "best_for": "Documents with clear topic transitions. Keeps topically related sentences together.",
        },
        "Hierarchical": {
            "description": "Small child chunks (200 chars) for precise retrieval; large parent chunks (1000 chars) returned as context.",
            "best_for": "Long-form documents where you want precision in retrieval but richness in context.",
        },
        "Sentence Window": {
            "description": "Each sentence is embedded individually. Surrounding sentences (window) are returned for context.",
            "best_for": "High-precision retrieval tasks. Good when individual sentences are self-contained.",
        },
    }

    st.divider()

    # Query input
    query = st.text_area(
        "Enter your query",
        placeholder="e.g. What is RAG and how does it reduce hallucination?",
        height=80,
    )

    k = st.slider("Number of chunks to retrieve per strategy", 1, 5, 3)

    run = st.button("Compare All Strategies", type="primary", disabled=not query.strip())

    if run and query.strip():
        with st.spinner("Running all 4 strategies... this may take 30-60 seconds."):
            results = compare_chunking_strategies(
                query=query.strip(),
                documents=documents,
                embedding_model=embedding_model,
                llm=llm,
                k=k,
            )

        st.divider()
        st.subheader("Results")

        # Display in a 2x2 grid
        strategies = list(results.keys())

        for row_start in range(0, len(strategies), 2):
            cols = st.columns(2)
            for col_idx, strat_name in enumerate(strategies[row_start:row_start + 2]):
                result = results[strat_name]
                info = strategy_info.get(strat_name, {})

                with cols[col_idx]:
                    st.markdown(f"### {strat_name}")

                    # Strategy info
                    with st.expander("Strategy details", expanded=False):
                        st.write(f"**How it works:** {info.get('description', '')}")
                        st.write(f"**Best for:** {info.get('best_for', '')}")
                        st.write(f"**Strategy note:** {result.get('strategy_note', '')}")

                    # Metrics row
                    m1, m2 = st.columns(2)
                    m1.metric("Total Chunks Built", result.get("num_chunks", "N/A"))
                    m2.metric("Chunks Retrieved", result.get("retrieved", "N/A"))

                    # Answer
                    st.write("**Answer:**")
                    st.info(result.get("answer", "No answer generated."))

            st.divider()

        # Summary comparison table
        st.subheader("Quick Comparison Table")

        table_data = {
            "Strategy": list(results.keys()),
            "Total Chunks": [r.get("num_chunks", "-") for r in results.values()],
            "Chunks Retrieved": [r.get("retrieved", "-") for r in results.values()],
            "Answer Length (chars)": [len(r.get("answer", "")) for r in results.values()],
        }

        import pandas as pd
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        st.caption(
            "Note: Answer quality differences become more visible with domain-specific documents "
            "and queries that require precise context. Upload your own documents for a richer comparison."
        )
