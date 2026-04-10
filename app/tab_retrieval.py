"""
app/tab_retrieval.py
---------------------
Tab 4: Hybrid Search and Reranking
Demonstrates and compares three retrieval approaches:
1. Dense only (FAISS semantic search)
2. Sparse only (BM25 keyword search)
3. Hybrid (Dense + BM25 fused with RRF)
Also demonstrates cross-encoder reranking on top of any retrieval result.
Allows the user to inspect exactly which documents were retrieved by each method.
"""

import streamlit as st
import pandas as pd

from core.models import load_groq_llm, load_embedding_model, load_reranker
from core.rag_engine import (
    dense_retrieve,
    bm25_retrieve,
    hybrid_retrieve,
    rerank_documents,
    simple_rag,
    build_vectorstore,
)
from utils.helpers import display_retrieved_docs, truncate


def render():
    """Render the Hybrid Search and Reranking tab."""
    st.header("Hybrid Search and Reranking")
    st.write(
        "Compare dense (FAISS), sparse (BM25), and hybrid retrieval side by side. "
        "Then see how cross-encoder reranking reorders the results."
    )

    if not st.session_state.get("kb_ready"):
        st.warning("Please build the knowledge base first in the 'Knowledge Base' tab.")
        return

    vectorstore = st.session_state["vectorstore"]
    bm25_index = st.session_state["bm25_index"]
    bm25_chunks = st.session_state["bm25_chunks"]
    llm = load_groq_llm()

    st.divider()

    # Explain the retrieval modes
    with st.expander("How these retrieval modes work", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Dense (FAISS)**")
            st.write(
                "Encodes query and documents as vectors. "
                "Finds documents with the highest cosine similarity. "
                "Understands semantic meaning even with different wording."
            )
        with col2:
            st.markdown("**Sparse (BM25)**")
            st.write(
                "Classic keyword-frequency algorithm (like search engines). "
                "Scores documents by how many query terms appear and how often. "
                "Excels at exact keyword matches."
            )
        with col3:
            st.markdown("**Hybrid (RRF)**")
            st.write(
                "Runs both dense and sparse retrievals, then fuses the ranked lists "
                "using Reciprocal Rank Fusion. "
                "Gets semantic understanding AND exact keyword precision."
            )

    st.divider()

    query = st.text_area(
        "Enter your query",
        placeholder="e.g. FAISS similarity search vector database",
        height=70,
        key="retrieval_query",
    )

    k = st.slider("Number of results per retrieval method (k)", 1, 8, 4)
    show_reranking = st.checkbox("Apply cross-encoder reranking to hybrid results", value=True)

    run_btn = st.button("Run Retrieval Comparison", type="primary", disabled=not query.strip())

    if run_btn and query.strip():
        q = query.strip()

        with st.spinner("Running retrieval methods..."):
            # Dense retrieval
            dense_docs = dense_retrieve(q, vectorstore, k=k)

            # BM25 retrieval
            bm25_results = bm25_retrieve(q, bm25_index, bm25_chunks, k=k)
            bm25_docs = [doc for doc, _ in bm25_results]
            bm25_scores = [score for _, score in bm25_results]

            # Hybrid retrieval
            hybrid_docs = hybrid_retrieve(q, vectorstore, bm25_index, bm25_chunks, k=k)

            # Reranking (on hybrid results)
            reranked = []
            if show_reranking:
                reranker = load_reranker()
                reranked = rerank_documents(q, hybrid_docs, reranker, top_k=min(3, len(hybrid_docs)))

        st.divider()

        # ---- Side-by-side results ----
        st.subheader("Retrieval Results Comparison")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Dense (FAISS)**")
            st.caption(f"{len(dense_docs)} results")
            for i, doc in enumerate(dense_docs):
                cid = doc.metadata.get("chunk_id", i)
                with st.expander(f"Rank {i+1} | Chunk {cid}", expanded=i == 0):
                    st.write(doc.page_content)

        with col2:
            st.markdown("**Sparse (BM25)**")
            st.caption(f"{len(bm25_docs)} results")
            for i, (doc, score) in enumerate(zip(bm25_docs, bm25_scores)):
                cid = doc.metadata.get("chunk_id", i)
                with st.expander(f"Rank {i+1} | Chunk {cid} | BM25: {score:.3f}", expanded=i == 0):
                    st.write(doc.page_content)

        with col3:
            st.markdown("**Hybrid (Dense + BM25 via RRF)**")
            st.caption(f"{len(hybrid_docs)} results after fusion")
            for i, doc in enumerate(hybrid_docs):
                cid = doc.metadata.get("chunk_id", i)
                with st.expander(f"Rank {i+1} | Chunk {cid}", expanded=i == 0):
                    st.write(doc.page_content)

        # ---- Reranking results ----
        if show_reranking and reranked:
            st.divider()
            st.subheader("After Cross-Encoder Reranking")
            st.write(
                "Cross-encoder reranking processes each (query, document) pair jointly, "
                "producing more accurate relevance scores than the initial retrieval."
            )

            rerank_data = []
            for rank, (doc, score) in enumerate(reranked):
                cid = doc.metadata.get("chunk_id", rank)
                rerank_data.append({
                    "Rank": rank + 1,
                    "Chunk ID": cid,
                    "Reranker Score": f"{score:.4f}",
                    "Preview": truncate(doc.page_content, 120),
                })

            st.dataframe(pd.DataFrame(rerank_data), use_container_width=True)

            # Show full content of top reranked doc
            if reranked:
                st.write("**Top reranked document (full text):**")
                st.info(reranked[0][0].page_content)

        # ---- Generate answer using hybrid + reranking ----
        st.divider()
        st.subheader("Generated Answer (using best retrieval result)")

        use_docs = [doc for doc, _ in reranked] if (show_reranking and reranked) else hybrid_docs[:3]
        context = "\n\n".join([d.page_content for d in use_docs])

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=(
                "You are a knowledgeable assistant. Answer the question using only the provided context. "
                "If the answer is not in the context, say so honestly."
            )),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {q}")
        ]
        with st.spinner("Generating answer..."):
            answer = llm.invoke(messages).content

        st.write(answer)

        # Overlap analysis
        st.divider()
        st.subheader("Document Overlap Analysis")
        st.write("Which documents appear in multiple retrieval methods?")

        dense_set = {hash(d.page_content) for d in dense_docs}
        bm25_set = {hash(d.page_content) for d in bm25_docs}
        hybrid_set = {hash(d.page_content) for d in hybrid_docs}

        col1, col2, col3 = st.columns(3)
        col1.metric("Dense unique docs", len(dense_set - bm25_set))
        col2.metric("BM25 unique docs", len(bm25_set - dense_set))
        col3.metric("In both Dense + BM25", len(dense_set & bm25_set))

        st.caption(
            "Documents in both dense and sparse retrievals are especially likely to be highly relevant "
            "and naturally bubble to the top with RRF fusion."
        )
