"""
app/tab_knowledge_base.py
--------------------------
Tab 1: Knowledge Base Setup
Allows the user to:
- Upload documents (.txt / .pdf)
- Or use the built-in sample corpus
- Choose a chunking strategy
- Build the FAISS vector store + BM25 index
- View the resulting chunks
"""

import streamlit as st
from langchain_core.documents import Document

from core.models import load_groq_llm, load_embedding_model
from core.rag_engine import (
    recursive_chunking,
    semantic_chunking,
    sentence_window_chunking,
    build_vectorstore,
    build_bm25_index,
)
from utils.helpers import load_documents_from_uploads

# Default sample corpus so users can explore without uploading a file
SAMPLE_CORPUS = """
Retrieval-Augmented Generation (RAG) is an AI framework that retrieves relevant information
from an external knowledge base before generating a response. RAG was introduced by Lewis et al.
in 2020. The core idea is to augment a frozen language model with a retrieval component that
fetches relevant documents at inference time, grounding responses in factual retrieved context.

RAG reduces hallucination by grounding the model's response in factual retrieved context.
Instead of relying solely on parametric knowledge (weights), the model uses retrieved documents
as grounding evidence. This allows the model to cite sources and produce verifiable outputs.
Hallucination is the tendency of LLMs to generate plausible-sounding but factually incorrect
information.

The RAG pipeline consists of four main components: the document store, the retriever, the reader,
and the generator. The document store indexes documents for retrieval. The retriever finds relevant
documents using embedding similarity or keyword search. The reader processes retrieved documents.
The generator uses the retrieved context to produce the final answer.

FAISS (Facebook AI Similarity Search) is an open-source library developed by Meta Research.
It enables efficient similarity search over large collections of dense vectors. FAISS supports
both exact and approximate nearest neighbor search. It is widely used in RAG systems as the
vector store backend due to its speed and scalability. FAISS can handle billions of vectors.

Chunking is the process of splitting documents into smaller pieces before embedding them.
Common chunking strategies include fixed-size chunking, recursive character splitting, semantic
chunking, and hierarchical chunking. Chunk size and overlap are hyperparameters that significantly
affect retrieval quality. Too small chunks lose context; too large chunks reduce precision.

Embeddings are dense vector representations of text that capture semantic meaning. Similar texts
have embeddings that are close in vector space, allowing cosine similarity or dot product to
measure relevance. Popular embedding models include Sentence-BERT, all-MiniLM-L6-v2, and BGE.

Reranking is a post-retrieval technique that uses a cross-encoder model to rescore retrieved
documents. Unlike bi-encoders that embed query and document separately, cross-encoders process
the (query, document) pair together, enabling more accurate relevance scoring.

Hybrid search combines dense retrieval (semantic vector search) with sparse retrieval (keyword
matching like BM25). Dense methods capture semantic meaning while sparse methods handle exact
keyword matches. Reciprocal Rank Fusion (RRF) is commonly used to merge ranked lists.

RAGAS is an evaluation framework for RAG systems providing reference-free metrics: faithfulness,
answer relevancy, context precision, and context recall.

LangChain is a framework for building LLM applications. It provides abstractions for chains,
agents, retrievers, and memory. It integrates with many LLM providers and vector stores,
simplifying the construction of RAG pipelines with ready-made composable components.

Autonomous agents are AI systems that can perceive their environment, make decisions, and take
actions to achieve goals without continuous human intervention. Modern AI agents use LLMs as
their reasoning core and are equipped with tools like web search and code execution.

Natural Language Processing (NLP) is a subfield of AI concerned with interactions between
computers and human language. It enables computers to process and analyze large amounts of
natural language data. Applications include sentiment analysis, machine translation, named
entity recognition, and question answering systems.
"""


def render():
    """Render the Knowledge Base tab."""
    st.header("Knowledge Base Setup")
    st.write(
        "Upload your documents or use the built-in sample corpus. "
        "The knowledge base is shared across all other tabs."
    )

    embedding_model = load_embedding_model()

    # --- Source selection ---
    source_choice = st.radio(
        "Document source",
        ["Use sample corpus", "Upload your own files"],
        horizontal=True,
    )

    if source_choice == "Upload your own files":
        uploaded_files = st.file_uploader(
            "Upload .txt or .pdf files",
            type=["txt", "pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            with st.spinner("Loading documents..."):
                documents = load_documents_from_uploads(uploaded_files)
            st.success(f"Loaded {len(documents)} document(s).")
        else:
            documents = []
    else:
        # Wrap sample text as a LangChain Document
        documents = [Document(page_content=SAMPLE_CORPUS, metadata={"source": "sample_corpus"})]
        st.info("Using built-in sample corpus (RAG and AI concepts).")

    st.divider()

    # --- Chunking configuration ---
    st.subheader("Chunking Strategy")
    st.write(
        "The chunking strategy determines how your documents are split before embedding. "
        "Each strategy has different trade-offs between precision and context coverage."
    )

    strategy = st.selectbox(
        "Select default chunking strategy for the knowledge base",
        ["Recursive", "Semantic", "Sentence Window"],
        help=(
            "Recursive: Fixed-size splits on natural boundaries. "
            "Semantic: Splits on meaning shifts. "
            "Sentence Window: Sentence-level splits with surrounding context. "
            "(Hierarchical is available in the Chunking Comparison tab.)"
        ),
    )

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk size (chars)", 200, 1000, 500, 50,
                               disabled=(strategy != "Recursive"))
    with col2:
        chunk_overlap = st.slider("Chunk overlap (chars)", 0, 300, 100, 25,
                                  disabled=(strategy != "Recursive"))

    st.divider()

    # --- Build button ---
    if st.button("Build Knowledge Base", type="primary", disabled=not documents):
        if not documents:
            st.warning("No documents to process.")
            return

        with st.spinner("Building knowledge base..."):

            # Apply selected chunking strategy
            if strategy == "Recursive":
                chunks = recursive_chunking(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            elif strategy == "Semantic":
                chunks = semantic_chunking(documents, embedding_model)
            else:  # Sentence Window
                chunks = sentence_window_chunking(documents, window_size=2)

            # Build FAISS vector store
            vectorstore = build_vectorstore(chunks, embedding_model)

            # Build BM25 index (for hybrid search)
            bm25_index, bm25_chunks = build_bm25_index(chunks)

            # Save everything to session state so other tabs can use it
            st.session_state["documents"] = documents
            st.session_state["chunks"] = chunks
            st.session_state["vectorstore"] = vectorstore
            st.session_state["bm25_index"] = bm25_index
            st.session_state["bm25_chunks"] = bm25_chunks
            st.session_state["kb_ready"] = True

        st.success(f"Knowledge base ready. {len(chunks)} chunks indexed.")

    # --- Display chunks if KB is built ---
    if st.session_state.get("kb_ready"):
        st.divider()
        st.subheader("Indexed Chunks")

        chunks = st.session_state["chunks"]
        st.write(f"Total chunks: **{len(chunks)}** | Strategy: **{strategy}**")

        # Chunk statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Chunks", len(chunks))
        col2.metric("Avg Chunk Size", f"{sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)} chars")
        col3.metric("Documents", len(st.session_state["documents"]))

        st.write("Preview of first 5 chunks:")
        for i, chunk in enumerate(chunks[:5]):
            cid = chunk.metadata.get("chunk_id", i)
            source = chunk.metadata.get("source", "unknown")
            with st.expander(f"Chunk {cid} | {source} | {len(chunk.page_content)} chars"):
                st.text(chunk.page_content)
                if chunk.metadata.get("window"):
                    st.caption(f"Window context: {chunk.metadata['window'][:200]}...")
