"""
core/rag_engine.py
------------------
The heart of the application. Contains all RAG logic:
- Document ingestion and chunking (4 strategies)
- Vector store management (build, save, load, update)
- Retrieval (dense, sparse BM25, hybrid)
- Reranking (cross-encoder)
- Generation (simple RAG, HyDE, multi-query, branched, conversational)
All functions are pure (no Streamlit calls) so they can be unit-tested independently.
"""

import os
import re
import numpy as np
from typing import List, Tuple, Dict, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# ---------------------------------------------------------------------------
# SECTION 1: Chunking Strategies
# ---------------------------------------------------------------------------

def recursive_chunking(documents: List[Document], chunk_size=500, chunk_overlap=100) -> List[Document]:
    """
    Recursive Character Text Splitting.
    Splits on paragraph -> sentence -> word boundaries in order.
    Most common, general-purpose chunking strategy.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        c.metadata["chunk_strategy"] = "recursive"
    return chunks


def semantic_chunking(documents: List[Document], embedding_model) -> List[Document]:
    """
    Semantic Chunking using embeddings.
    Detects topic shifts by measuring cosine similarity between consecutive sentences.
    Splits where similarity drops significantly (a topic change).
    """
    from langchain_experimental.text_splitter import SemanticChunker
    splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        c.metadata["chunk_strategy"] = "semantic"
    return chunks


def hierarchical_chunking(documents: List[Document], embedding_model) -> Tuple[object, List[Document]]:
    """
    Hierarchical / Parent-Document Chunking.
    - Child chunks (small, 200 chars) are embedded for precise retrieval.
    - Parent chunks (large, 1000 chars) are returned as context for generation.
    Returns the retriever object and the parent-level documents for display.
    """
    from langchain_classic.retrievers import ParentDocumentRetriever
    from langchain_core.stores import InMemoryStore

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    # Placeholder FAISS to initialize the retriever
    child_vectorstore = FAISS.from_texts(["__placeholder__"], embedding=embedding_model)
    parent_docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=child_vectorstore,
        docstore=parent_docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(documents)

    # Also produce flat parent chunks for display purposes
    parent_chunks = parent_splitter.split_documents(documents)
    for i, c in enumerate(parent_chunks):
        c.metadata["chunk_id"] = i
        c.metadata["chunk_strategy"] = "hierarchical"

    return retriever, parent_chunks


def sentence_window_chunking(documents: List[Document], window_size=2) -> List[Document]:
    """
    Sentence-Window Chunking.
    - Each sentence is embedded individually (precise matching).
    - Each sentence stores its surrounding window in metadata (richer context).
    At retrieval time, the window text is used instead of just the sentence.
    """
    all_chunks = []
    for doc in documents:
        text = doc.page_content.strip()
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for i, sentence in enumerate(sentences):
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size + 1)
            window_text = " ".join(sentences[start:end])

            chunk = Document(
                page_content=sentence,   # Only sentence is embedded
                metadata={
                    **doc.metadata,
                    "chunk_id": len(all_chunks),
                    "chunk_strategy": "sentence_window",
                    "window": window_text,     # Window stored for generation
                    "sentence_index": i,
                }
            )
            all_chunks.append(chunk)
    return all_chunks


# ---------------------------------------------------------------------------
# SECTION 2: Vector Store Management
# ---------------------------------------------------------------------------

def build_vectorstore(chunks: List[Document], embedding_model) -> FAISS:
    """Build a FAISS vector store from a list of Document chunks."""
    return FAISS.from_documents(chunks, embedding_model)


def save_vectorstore(vs: FAISS, path: str):
    """Persist FAISS index to disk. Avoids re-embedding on each app restart."""
    vs.save_local(path)


def load_vectorstore(path: str, embedding_model) -> FAISS:
    """Load a saved FAISS index from disk."""
    return FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True
    )


def add_documents_to_vectorstore(vs: FAISS, new_chunks: List[Document], path: str) -> FAISS:
    """Incrementally add new document chunks to an existing vector store."""
    vs.add_documents(new_chunks)
    save_vectorstore(vs, path)
    return vs


# ---------------------------------------------------------------------------
# SECTION 3: Retrieval Methods
# ---------------------------------------------------------------------------

def dense_retrieve(query: str, vectorstore: FAISS, k: int = 4) -> List[Document]:
    """Standard FAISS semantic (dense) retrieval."""
    return vectorstore.similarity_search(query, k=k)


def build_bm25_index(chunks: List[Document]):
    """
    Build BM25 sparse index from document chunks.
    BM25 is a classic keyword-frequency retrieval algorithm.
    Returns the BM25 model and the corresponding chunks list.
    """
    from rank_bm25 import BM25Okapi
    corpus = [c.page_content for c in chunks]
    tokenized = [text.lower().split() for text in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunks


def bm25_retrieve(query: str, bm25_index, chunks: List[Document], k: int = 4) -> List[Tuple[Document, float]]:
    """BM25 sparse keyword retrieval. Returns (doc, score) tuples."""
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top_indices]


def reciprocal_rank_fusion(ranked_lists: List[List[Tuple[Document, float]]], k: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF).
    Merges multiple ranked document lists into a single ranking.
    Formula: score(doc) = sum of 1/(k + rank) across all lists.
    Documents appearing high in multiple lists rise to the top.
    """
    rrf_scores: Dict[int, float] = {}
    doc_map: Dict[int, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _) in enumerate(ranked_list):
            key = hash(doc.page_content)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

    sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [doc_map[key] for key in sorted_keys]


def hybrid_retrieve(query: str, vectorstore: FAISS, bm25_index, chunks: List[Document], k: int = 5) -> List[Document]:
    """
    Hybrid retrieval: Dense (FAISS) + Sparse (BM25) merged with RRF.
    Gets the best of semantic understanding and keyword matching.
    """
    dense_docs = vectorstore.similarity_search(query, k=k)
    dense_ranked = [(doc, 1.0) for doc in dense_docs]

    bm25_ranked = bm25_retrieve(query, bm25_index, chunks, k=k)

    return reciprocal_rank_fusion([dense_ranked, bm25_ranked])


def rerank_documents(query: str, docs: List[Document], reranker, top_k: int = 3) -> List[Tuple[Document, float]]:
    """
    Cross-encoder reranking.
    Scores each (query, document) pair jointly — more accurate than bi-encoder alone.
    Returns documents sorted by reranker score with their scores.
    """
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [(doc, float(score)) for score, doc in scored[:top_k]]


# ---------------------------------------------------------------------------
# SECTION 4: Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the question using only the provided context. "
    "If the answer is not in the context, say clearly that you don't know based on the provided information. "
    "Do not fabricate information."
)

CITATION_PROMPT = (
    "You are a knowledgeable assistant. Answer the question using only the provided context. "
    "At the end of your answer, list the chunk IDs you relied on under a 'Sources:' heading. "
    "If the answer is not in the context, say so honestly."
)


# ---------------------------------------------------------------------------
# SECTION 5: Generation Patterns
# ---------------------------------------------------------------------------

def simple_rag(query: str, vectorstore: FAISS, llm, k: int = 3) -> Dict:
    """
    Basic RAG: retrieve top-k chunks, build context, generate answer.
    Returns answer + retrieved docs for display.
    """
    docs = dense_retrieve(query, vectorstore, k=k)
    context = "\n\n".join([d.page_content for d in docs])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]
    response = llm.invoke(messages)
    return {"answer": response.content, "docs": docs, "context": context}


def rag_with_citations(query: str, vectorstore: FAISS, llm, k: int = 3) -> Dict:
    """
    RAG with source citations. Chunks have chunk_id metadata.
    The LLM is prompted to cite which chunk IDs support its answer.
    """
    docs = dense_retrieve(query, vectorstore, k=k)
    context_parts = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id", "?")
        context_parts.append(f"[Chunk {cid}]:\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=CITATION_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]
    response = llm.invoke(messages)
    return {"answer": response.content, "docs": docs, "context": context}


def hyde_rag(query: str, vectorstore: FAISS, llm, k: int = 3) -> Dict:
    """
    HyDE (Hypothetical Document Embeddings).
    1. Ask LLM to write a hypothetical answer to the query.
    2. Embed that hypothetical answer (closer to real docs than raw query).
    3. Use the hypothetical answer embedding for retrieval.
    4. Generate final answer from the real retrieved docs.
    """
    # Step 1: Generate hypothetical document
    hyde_messages = [
        SystemMessage(content=(
            "Write a short factual paragraph (2-3 sentences) that answers the following question. "
            "Write as if from a technical document. No preamble — just the paragraph."
        )),
        HumanMessage(content=query)
    ]
    hypothetical_doc = llm.invoke(hyde_messages).content

    # Step 2 & 3: Retrieve using hypothetical doc embedding
    docs = vectorstore.similarity_search(hypothetical_doc, k=k)
    context = "\n\n".join([d.page_content for d in docs])

    # Step 4: Generate real answer from retrieved context
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]
    answer = llm.invoke(messages).content
    return {"answer": answer, "docs": docs, "hypothetical_doc": hypothetical_doc, "context": context}


def multi_query_rag(query: str, vectorstore: FAISS, llm, n_variations: int = 3, k: int = 3) -> Dict:
    """
    Multi-Query RAG.
    1. Generate N alternative phrasings of the query.
    2. Retrieve documents for each phrasing.
    3. Deduplicate across all retrievals.
    4. Generate answer from the merged, unique document set.
    """
    # Step 1: Generate query variations
    var_messages = [
        SystemMessage(content=(
            f"Generate {n_variations} different phrasings of the following question. "
            "Each should approach the topic from a slightly different angle. "
            f"Output exactly {n_variations} lines, one per line, no numbering, no extra text."
        )),
        HumanMessage(content=query)
    ]
    variations_text = llm.invoke(var_messages).content
    variations = [v.strip() for v in variations_text.strip().split("\n") if v.strip()][:n_variations]
    all_queries = [query] + variations

    # Step 2 & 3: Retrieve and deduplicate
    seen = set()
    unique_docs = []
    for q in all_queries:
        for doc in vectorstore.similarity_search(q, k=k):
            h = hash(doc.page_content)
            if h not in seen:
                seen.add(h)
                unique_docs.append(doc)

    # Step 4: Generate
    context = "\n\n".join([d.page_content for d in unique_docs])
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]
    answer = llm.invoke(messages).content
    return {"answer": answer, "docs": unique_docs, "query_variations": all_queries, "context": context}


def route_query(query: str, llm) -> str:
    """
    Classify a query as 'factual', 'analytical', or 'general'.
    Used by branched RAG to decide which retrieval pipeline to use.
    """
    messages = [
        SystemMessage(content=(
            "Classify the following question into exactly one category: factual, analytical, or general. "
            "factual = specific fact/definition/named entity. "
            "analytical = comparison, cause/effect, reasoning. "
            "general = broad or open-ended. "
            "Reply with ONLY the category word."
        )),
        HumanMessage(content=query)
    ]
    category = llm.invoke(messages).content.strip().lower()
    return category if category in ["factual", "analytical", "general"] else "general"


def branched_rag(query: str, vectorstore: FAISS, llm, k: int = 3) -> Dict:
    """
    Branched RAG with query routing.
    Routes query to different retrieval strategies based on query type:
    - factual  -> precise dense retrieval (k=2)
    - analytical -> HyDE for richer retrieval
    - general  -> standard RAG (k=4)
    """
    query_type = route_query(query, llm)

    if query_type == "factual":
        docs = dense_retrieve(query, vectorstore, k=2)
        context = "\n\n".join([d.page_content for d in docs])
        messages = [
            SystemMessage(content="Answer this factual question concisely using only the context."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        ]
        answer = llm.invoke(messages).content

    elif query_type == "analytical":
        result = hyde_rag(query, vectorstore, llm, k=k)
        return {**result, "query_type": query_type, "routing_path": "HyDE retrieval"}

    else:
        docs = dense_retrieve(query, vectorstore, k=4)
        context = "\n\n".join([d.page_content for d in docs])
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        ]
        answer = llm.invoke(messages).content

    return {
        "answer": answer,
        "docs": docs,
        "query_type": query_type,
        "routing_path": {
            "factual": "Dense retrieval (k=2)",
            "general": "Dense retrieval (k=4)",
        }.get(query_type, "HyDE retrieval"),
        "context": context,
    }


def conversational_rag(
    query: str,
    vectorstore: FAISS,
    llm,
    chat_history: List[Tuple[str, str]],
    k: int = 3,
    max_history: int = 6
) -> Dict:
    """
    Conversational RAG with memory.
    Includes recent chat history in the prompt so the LLM can handle follow-up questions.
    """
    docs = dense_retrieve(query, vectorstore, k=k)
    context = "\n\n".join([d.page_content for d in docs])

    messages = [
        SystemMessage(content=(
            "You are a knowledgeable assistant with access to a knowledge base. "
            "Use the provided context to answer questions. "
            "If a question refers to earlier conversation, use the history. "
            "Never fabricate information."
        ))
    ]

    # Add recent history (limit to avoid token overflow)
    for human_msg, ai_msg in chat_history[-max_history:]:
        messages.append(HumanMessage(content=human_msg))
        messages.append(AIMessage(content=ai_msg))

    messages.append(HumanMessage(
        content=f"Knowledge Base Context:\n{context}\n\nQuestion: {query}"
    ))

    response = llm.invoke(messages)
    return {"answer": response.content, "docs": docs, "context": context}


# ---------------------------------------------------------------------------
# SECTION 6: Chunking Comparison
# ---------------------------------------------------------------------------

def compare_chunking_strategies(
    query: str,
    documents: List[Document],
    embedding_model,
    llm,
    k: int = 3,
) -> Dict[str, Dict]:
    """
    Run all four chunking strategies and return the answer from each.
    Used for side-by-side comparison in the UI.
    Each strategy builds its own FAISS index and runs the full RAG pipeline.
    """
    results = {}

    # 1. Recursive
    try:
        r_chunks = recursive_chunking(documents)
        r_vs = build_vectorstore(r_chunks, embedding_model)
        r_result = simple_rag(query, r_vs, llm, k=k)
        results["Recursive"] = {
            "answer": r_result["answer"],
            "num_chunks": len(r_chunks),
            "retrieved": len(r_result["docs"]),
            "strategy_note": "Splits on paragraph -> sentence -> word. Fixed size with overlap.",
        }
    except Exception as e:
        results["Recursive"] = {"answer": f"Error: {e}", "num_chunks": 0, "retrieved": 0, "strategy_note": ""}

    # 2. Semantic
    try:
        s_chunks = semantic_chunking(documents, embedding_model)
        s_vs = build_vectorstore(s_chunks, embedding_model)
        s_result = simple_rag(query, s_vs, llm, k=k)
        results["Semantic"] = {
            "answer": s_result["answer"],
            "num_chunks": len(s_chunks),
            "retrieved": len(s_result["docs"]),
            "strategy_note": "Splits on meaning shifts detected via embedding similarity drops.",
        }
    except Exception as e:
        results["Semantic"] = {"answer": f"Error: {e}", "num_chunks": 0, "retrieved": 0, "strategy_note": ""}

    # 3. Hierarchical
    try:
        h_retriever, h_parent_chunks = hierarchical_chunking(documents, embedding_model)
        h_docs = h_retriever.invoke(query)[:k]
        h_context = "\n\n".join([d.page_content for d in h_docs])
        h_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{h_context}\n\nQuestion: {query}")
        ]
        h_answer = llm.invoke(h_messages).content
        results["Hierarchical"] = {
            "answer": h_answer,
            "num_chunks": len(h_parent_chunks),
            "retrieved": len(h_docs),
            "strategy_note": "Child chunks embedded for retrieval; full parent chunks returned for context.",
        }
    except Exception as e:
        results["Hierarchical"] = {"answer": f"Error: {e}", "num_chunks": 0, "retrieved": 0, "strategy_note": ""}

    # 4. Sentence Window
    try:
        sw_chunks = sentence_window_chunking(documents, window_size=2)
        sw_vs = build_vectorstore(sw_chunks, embedding_model)
        sw_docs = dense_retrieve(query, sw_vs, k=k)
        # Use window context instead of raw sentence
        sw_context_parts = [d.metadata.get("window", d.page_content) for d in sw_docs]
        sw_context = "\n\n".join(sw_context_parts)
        sw_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{sw_context}\n\nQuestion: {query}")
        ]
        sw_answer = llm.invoke(sw_messages).content
        results["Sentence Window"] = {
            "answer": sw_answer,
            "num_chunks": len(sw_chunks),
            "retrieved": len(sw_docs),
            "strategy_note": "Sentences embedded individually; surrounding window returned as context.",
        }
    except Exception as e:
        results["Sentence Window"] = {"answer": f"Error: {e}", "num_chunks": 0, "retrieved": 0, "strategy_note": ""}

    return results
