"""
app/tab_learning.py
--------------------
Tab: Learning
Covers all concepts from the three RAG notebooks:
  Notebook 1 - Basic RAG (setup, chunking, FAISS, pipeline, prompt engineering, citations)
  Notebook 2 - Advanced RAG (chunking strategies, RAG patterns, hybrid search, reranking)
  Notebook 3 - RAG Evaluation (RAGAS, DeepEval, production chatbot)
Organized as collapsible sections so the reader can go at their own pace.
"""

import streamlit as st


def render():
    st.header("Learning")
    st.write(
        "This section walks through every RAG concept covered in the notebooks, "
        "with explanations, code examples, and key takeaways. "
        "Use the sections below to study at your own pace."
    )

    # -------------------------------------------------------------------------
    # TOP-LEVEL MODULE SELECTOR
    # -------------------------------------------------------------------------
    module = st.radio(
        "Select module",
        [
            "Notebook 1 — Basic RAG",
            "Notebook 2 — Advanced RAG",
            "Notebook 3 — Evaluation and Production Chatbot",
        ],
        horizontal=True,
    )

    st.divider()

    # =========================================================================
    # NOTEBOOK 1
    # =========================================================================
    if module == "Notebook 1 — Basic RAG":

        st.subheader("Notebook 1 — Basic RAG: From Setup to Citations")
        st.write(
            "This module covers the foundations of RAG: why it exists, how to build a pipeline "
            "from scratch, how to improve it with better chunking and prompts, and how to make "
            "the system explainable with source citations."
        )

        # ------------------------------------------------------------------
        # 1.1 Introduction to RAG
        # ------------------------------------------------------------------
        with st.expander("1.1 — What is RAG and Why Does It Exist?", expanded=True):
            st.markdown("#### The Problem With Plain LLMs")
            st.write(
                "Large Language Models (LLMs) are trained on a fixed snapshot of the internet. "
                "Once training is done, their knowledge is frozen. This creates three practical problems:"
            )
            st.write(
                "- They may generate plausible-sounding but factually incorrect information (hallucination).\n"
                "- They have no access to private or domain-specific data.\n"
                "- Their knowledge becomes stale as the world changes."
            )

            st.markdown("#### What RAG Does")
            st.write(
                "RAG stands for Retrieval-Augmented Generation. "
                "It solves these problems by giving the LLM access to an external knowledge base at query time. "
                "Instead of relying purely on what the model memorized during training, "
                "the system retrieves relevant documents first, then passes them as context to the LLM."
            )

            st.markdown("#### The RAG Pipeline (5 Steps)")
            st.code(
                """
User Query
    |
    v
Step 1: Embed the query into a vector
    |
    v
Step 2: Search vector database for similar chunks
    |
    v
Step 3: Retrieve top-k relevant chunks
    |
    v
Step 4: Build a prompt with query + retrieved context
    |
    v
Step 5: LLM generates the final answer
""",
                language="text",
            )

            st.markdown("#### Key Terminology")
            st.write(
                "- **Parametric knowledge**: what the LLM learned during training (stored in weights).\n"
                "- **Non-parametric knowledge**: external documents retrieved at inference time.\n"
                "- **Grounding**: anchoring LLM output to retrieved facts to reduce hallucination.\n"
                "- **Hallucination**: when an LLM generates confident but incorrect information."
            )

        # ------------------------------------------------------------------
        # 1.2 Setup
        # ------------------------------------------------------------------
        with st.expander("1.2 — Setup: LLM, Embeddings, and FAISS"):
            st.markdown("#### The LLM")
            st.write(
                "We use `llama-3.1-8b-instant` via Groq. Groq provides a free API for fast inference. "
                "The LLM is responsible for generating the final answer from the retrieved context."
            )
            st.code(
                """
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,   # Low temperature = more factual, less creative
    max_tokens=1000,
)

# Send a message
response = llm.invoke([HumanMessage(content="What is RAG?")])
print(response.content)
""",
                language="python",
            )

            st.markdown("#### The Embedding Model")
            st.write(
                "Embeddings convert text into numerical vectors. "
                "Texts that are semantically similar end up with vectors that are close together in vector space. "
                "This allows us to search for relevant documents by comparing vector distances rather than exact keywords."
            )
            st.write(
                "We use `all-MiniLM-L6-v2` from Sentence Transformers. "
                "It runs fully locally (no API key needed), produces 384-dimensional vectors, "
                "and is fast enough for real-time retrieval."
            )
            st.code(
                """
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed three sentences
sentences = [
    "RAG improves accuracy",
    "Dogs are animals",
    "RAG helps reduce hallucination"
]
embeddings = embedding_model.encode(sentences)

print("Shape:", embeddings.shape)
# Output: (3, 384) — 3 sentences, each a 384-dimensional vector
""",
                language="python",
            )

            st.markdown("#### FAISS Vector Store")
            st.write(
                "FAISS (Facebook AI Similarity Search) is an open-source library by Meta Research. "
                "It stores embedding vectors and lets you search for the most similar ones very quickly, "
                "even across millions of vectors. In RAG, FAISS is where all document chunk embeddings are stored."
            )
            st.code(
                """
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build FAISS from documents
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Similarity search
results = vectorstore.similarity_search("How does RAG work?", k=3)
# Returns the 3 most semantically similar chunks
""",
                language="python",
            )

        # ------------------------------------------------------------------
        # 1.3 Document Loading
        # ------------------------------------------------------------------
        with st.expander("1.3 — Loading Documents"):
            st.write(
                "Before indexing, you need to load your documents into LangChain `Document` objects. "
                "A `Document` has two parts: `page_content` (the text) and `metadata` (source info, page number, etc.)."
            )
            st.code(
                """
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document

# Load a .txt file
loader = TextLoader("my_document.txt", encoding="utf-8")
docs = loader.load()

# Load a PDF
loader = PyPDFLoader("my_report.pdf")
docs = loader.load()  # One Document per page

# Create a Document manually from raw text
doc = Document(
    page_content="RAG reduces hallucination by grounding responses.",
    metadata={"source": "notes.txt", "page": 1}
)
""",
                language="python",
            )

        # ------------------------------------------------------------------
        # 1.4 Basic Chunking
        # ------------------------------------------------------------------
        with st.expander("1.4 — Basic Chunking"):
            st.markdown("#### Why Chunk?")
            st.write(
                "LLMs have a context window limit — you cannot pass an entire book as context. "
                "Chunking splits documents into smaller pieces that fit in the context window. "
                "Each chunk is embedded separately and stored in the vector database."
            )

            st.markdown("#### No Overlap vs Overlap")
            st.write(
                "Without overlap, a sentence that falls at a chunk boundary gets split apart. "
                "The first half is in chunk N, the second half is in chunk N+1. "
                "Retrieval then returns an incomplete chunk that cuts off mid-thought."
            )
            st.write(
                "With overlap, the end of chunk N is repeated at the start of chunk N+1. "
                "This ensures boundary content appears fully in at least one chunk."
            )
            st.code(
                """
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Basic chunking — no overlap
basic_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
)

# Improved chunking — with overlap
improved_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,  # 100 chars of overlap between consecutive chunks
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = improved_splitter.split_documents(docs)
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk: {chunks[0].page_content[:200]}")
""",
                language="python",
            )

            st.markdown("#### How RecursiveCharacterTextSplitter Works")
            st.write(
                "It tries to split on separators in order of priority: "
                "double newline (paragraph) first, then single newline (line), "
                "then period+space (sentence), then space (word), then character. "
                "It keeps chunks as close to `chunk_size` as possible without exceeding it."
            )

        # ------------------------------------------------------------------
        # 1.5 Basic RAG Pipeline
        # ------------------------------------------------------------------
        with st.expander("1.5 — Building the Basic RAG Pipeline"):
            st.write(
                "Once we have chunks in FAISS, we can build the core RAG function. "
                "It retrieves relevant chunks for a query and passes them as context to the LLM."
            )
            st.code(
                """
from langchain_core.messages import HumanMessage, SystemMessage

def rag_pipeline(query, vectorstore, llm, k=3):
    # Step 1: Retrieve top-k similar chunks
    docs = vectorstore.similarity_search(query, k=k)

    # Step 2: Build context string from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 3: Build the prompt
    messages = [
        SystemMessage(content="Answer using only the provided context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]

    # Step 4: Generate the answer
    response = llm.invoke(messages)
    return response.content, docs

answer, sources = rag_pipeline("How does RAG reduce hallucination?", vectorstore, llm)
print(answer)
""",
                language="python",
            )

        # ------------------------------------------------------------------
        # 1.6 Prompt Engineering (Basic)
        # ------------------------------------------------------------------
        with st.expander("1.6 — Basic Prompt Engineering for RAG"):
            st.write(
                "The quality of the LLM's answer depends heavily on how you write the prompt. "
                "For RAG specifically, a poorly written prompt can cause the model to ignore the context "
                "and answer from its training data — which defeats the whole purpose."
            )

            st.markdown("#### Key Prompt Improvements for RAG")

            st.write("**1. Tell the model to only use the context**")
            st.code(
                """
# Without constraint — model might use its own training data
SystemMessage(content="You are a helpful assistant.")

# With constraint — model is grounded to the retrieved context
SystemMessage(content=(
    "Answer the question using ONLY the provided context. "
    "Do not use any external knowledge."
))
""",
                language="python",
            )

            st.write("**2. Add 'I don't know' handling**")
            st.code(
                """
SystemMessage(content=(
    "Answer the question using only the provided context. "
    "If the answer is not in the context, say: "
    "'I don't know based on the provided information.' "
    "Never fabricate information."
))
""",
                language="python",
            )

            st.write("**3. Role prompting**")
            st.code(
                """
# Assign an expert persona to improve response quality
SystemMessage(content=(
    "You are a senior AI researcher. "
    "Answer precisely and technically based only on the provided context."
))
""",
                language="python",
            )

            st.markdown("#### Effect on Hallucination")
            st.write(
                "A constrained prompt reduces hallucination because the model is explicitly "
                "told that its only valid information source is the retrieved context. "
                "Without this constraint, models often blend context with training data, "
                "making it hard to tell which parts of the answer are grounded."
            )

        # ------------------------------------------------------------------
        # 1.7 Citations
        # ------------------------------------------------------------------
        with st.expander("1.7 — Adding Source Citations"):
            st.write(
                "Citations make the RAG system explainable. "
                "Instead of showing only the final answer, you also show which document chunks "
                "were used to generate it. This builds trust and helps debug retrieval problems."
            )
            st.code(
                """
# Step 1: Add chunk_id metadata to each chunk
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i

# Step 2: Label chunks in the context with their IDs
context_parts = []
for doc in docs:
    cid = doc.metadata.get("chunk_id", "?")
    context_parts.append(f"[Chunk {cid}]:\n{doc.page_content}")
context = "\n\n".join(context_parts)

# Step 3: Prompt the LLM to cite chunk IDs
system_prompt = (
    "Answer the question using only the context. "
    "At the end, list the Chunk IDs you used under 'Sources:'."
)

# Example output:
# Answer: RAG reduces hallucination by grounding responses in retrieved context.
# Sources:
# - Chunk 3
# - Chunk 7
""",
                language="python",
            )

            st.markdown("#### Why Citations Matter")
            st.write(
                "- **Debugging**: If the answer is wrong, you can inspect which chunks were retrieved "
                "and whether the issue is in retrieval (wrong chunks) or generation (correct chunks, bad answer).\n"
                "- **Trust**: Users can verify the answer against the original source.\n"
                "- **Explainability**: Required for many production use cases in legal, medical, and finance domains."
            )

    # =========================================================================
    # NOTEBOOK 2
    # =========================================================================
    elif module == "Notebook 2 — Advanced RAG":

        st.subheader("Notebook 2 — Advanced RAG: Chunking, Patterns, Hybrid Search, Reranking")
        st.write(
            "This module goes beyond the basics. It covers four professional chunking strategies, "
            "five RAG architectural patterns, hybrid search, and cross-encoder reranking."
        )

        # ------------------------------------------------------------------
        # 2.1 Advanced Chunking
        # ------------------------------------------------------------------
        with st.expander("2.1 — Advanced Chunking Strategies", expanded=True):
            st.write(
                "Chunking strategy is one of the highest-leverage decisions in RAG. "
                "Poor chunking causes context fragmentation — the key information gets split "
                "across two chunks and neither retrieved chunk is complete enough to answer the question."
            )

            st.markdown("#### Strategy Comparison")
            import pandas as pd
            df = pd.DataFrame({
                "Strategy": ["Recursive", "Semantic", "Hierarchical", "Sentence Window"],
                "Split Basis": ["Character count", "Meaning shift (embedding similarity)", "Two levels: parent + child", "Individual sentences"],
                "Best For": ["General purpose", "Topic-structured documents", "Long documents needing precision + richness", "High-precision retrieval"],
                "Main Advantage": ["Predictable chunk size", "Keeps topics together", "Precise retrieval, rich context", "Exact sentence matching"],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### Semantic Chunking")
            st.write(
                "Instead of splitting at fixed character counts, semantic chunking embeds each sentence "
                "and measures the cosine similarity between consecutive sentences. "
                "Where similarity drops sharply, a topic change has occurred — and that is where the split happens."
            )
            st.code(
                """
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# breakpoint_threshold_amount=95 means:
# split where similarity drops are in the top 5% largest drops
semantic_splitter = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)

chunks = semantic_splitter.split_documents(docs)
# Chunks are fewer but larger, each covering a single coherent topic
""",
                language="python",
            )

            st.markdown("---")
            st.markdown("#### Hierarchical (Parent-Document) Chunking")
            st.write(
                "The core insight: small chunks are better for retrieval (precise semantic match), "
                "but large chunks are better for generation (more context around the answer). "
                "Hierarchical chunking separates these two concerns."
            )
            st.write(
                "- Child chunks (200 chars) are embedded and stored in FAISS for retrieval.\n"
                "- Parent chunks (1000 chars) are stored in a docstore.\n"
                "- When a child chunk is retrieved, the system returns its full parent chunk as context."
            )
            st.code(
                """
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

parent_docstore = InMemoryStore()
child_vectorstore = FAISS.from_texts(["placeholder"], embedding=embedding_model)

retriever = ParentDocumentRetriever(
    vectorstore=child_vectorstore,
    docstore=parent_docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(docs)

# Retrieve: matches on child chunks, returns full parent
results = retriever.invoke("What is RAG?")
""",
                language="python",
            )

            st.markdown("---")
            st.markdown("#### Sentence-Window Chunking")
            st.write(
                "Each sentence is embedded individually. "
                "But each chunk also stores a 'window' — the 2 sentences before and after it — in metadata. "
                "At retrieval time, the window text is used as context instead of just the single sentence. "
                "This gives precise matching with richer context."
            )
            st.code(
                """
import re
from langchain.schema import Document

def build_sentence_window_chunks(text, window_size=2):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    docs = []
    for i, sentence in enumerate(sentences):
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)
        window_text = " ".join(sentences[start:end])

        doc = Document(
            page_content=sentence,           # Only sentence is embedded
            metadata={
                "sentence_index": i,
                "window": window_text,       # Window is stored for generation
            }
        )
        docs.append(doc)
    return docs

# At retrieval time, use the window instead of the sentence:
retrieved_docs = vectorstore.similarity_search(query, k=3)
context = "\n\n".join([
    d.metadata.get("window", d.page_content) for d in retrieved_docs
])
""",
                language="python",
            )

        # ------------------------------------------------------------------
        # 2.2 RAG Patterns
        # ------------------------------------------------------------------
        with st.expander("2.2 — Advanced RAG Patterns"):
            st.write(
                "Beyond the basic retrieve-then-generate pattern, there are several architectural "
                "variants that address specific weaknesses of simple RAG."
            )

            st.markdown("---")
            st.markdown("#### RAG with Conversation Memory")
            st.write(
                "Basic RAG treats every question as independent. "
                "A conversational chatbot needs to handle follow-up questions like 'Tell me more about that.' "
                "We maintain a list of previous (user, assistant) pairs and include them in the prompt."
            )
            st.code(
                """
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

chat_history = []  # List of (user_text, assistant_text) tuples

def conversational_rag(query, vectorstore, llm, chat_history, k=3):
    # Retrieve context for the current query
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([d.page_content for d in docs])

    messages = [SystemMessage(content="Answer using the context and chat history.")]

    # Add recent conversation turns
    for human_msg, ai_msg in chat_history[-6:]:  # Keep last 6 turns
        messages.append(HumanMessage(content=human_msg))
        messages.append(AIMessage(content=ai_msg))

    messages.append(HumanMessage(
        content=f"Context:\n{context}\n\nQuestion: {query}"
    ))

    response = llm.invoke(messages)
    chat_history.append((query, response.content))  # Save to history
    return response.content
""",
                language="python",
            )

            st.markdown("---")
            st.markdown("#### HyDE — Hypothetical Document Embeddings")
            st.write(
                "Problem: A user's question and a relevant document may be phrased very differently. "
                "The question 'Why does RAG reduce hallucination?' and the document passage "
                "'RAG grounds responses in retrieved context' may not be close in embedding space."
            )
            st.write(
                "HyDE Solution: Before retrieval, ask the LLM to write a hypothetical answer to the question. "
                "Embed that hypothetical answer (not the original question). "
                "The hypothetical answer uses the same vocabulary and style as real documents, "
                "so it lands closer to them in embedding space."
            )
            st.code(
                """
def hyde_rag(query, vectorstore, llm, k=3):
    # Step 1: Generate a hypothetical answer
    hyde_messages = [
        SystemMessage(content=(
            "Write a short factual paragraph that answers this question. "
            "Write as if from a technical document. No preamble."
        )),
        HumanMessage(content=query)
    ]
    hypothetical_doc = llm.invoke(hyde_messages).content

    # Step 2: Use hypothetical doc embedding for retrieval
    docs = vectorstore.similarity_search(hypothetical_doc, k=k)
    context = "\n\n".join([d.page_content for d in docs])

    # Step 3: Generate real answer from retrieved context
    messages = [
        SystemMessage(content="Answer using only the provided context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]
    return llm.invoke(messages).content
""",
                language="python",
            )

            st.markdown("---")
            st.markdown("#### Multi-Query RAG")
            st.write(
                "A single query phrasing may miss documents that use different terminology. "
                "Multi-Query RAG generates N alternative phrasings, retrieves documents for each, "
                "deduplicates, and generates the final answer from the merged unique document set. "
                "This increases recall — more relevant documents are found."
            )
            st.code(
                """
def multi_query_rag(query, vectorstore, llm, n=3, k=3):
    # Generate alternative phrasings
    messages = [
        SystemMessage(content=(
            f"Generate {n} different phrasings of this question. "
            f"Output exactly {n} lines, no numbering."
        )),
        HumanMessage(content=query)
    ]
    variations_text = llm.invoke(messages).content
    variations = [v.strip() for v in variations_text.split("\n") if v.strip()][:n]
    all_queries = [query] + variations

    # Retrieve and deduplicate
    seen = set()
    unique_docs = []
    for q in all_queries:
        for doc in vectorstore.similarity_search(q, k=k):
            h = hash(doc.page_content)
            if h not in seen:
                seen.add(h)
                unique_docs.append(doc)

    # Generate from merged context
    context = "\n\n".join([d.page_content for d in unique_docs])
    messages = [
        SystemMessage(content="Answer using only the context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]
    return llm.invoke(messages).content
""",
                language="python",
            )

            st.markdown("---")
            st.markdown("#### Branched RAG (Query Routing)")
            st.write(
                "Not all queries are the same type. A specific factual lookup ('What is FAISS?') "
                "is best served by precise dense retrieval. An analytical question ('Why is RAG better than fine-tuning?') "
                "benefits from HyDE. A broad question works with standard RAG. "
                "A router LLM classifies the query and dispatches it to the right pipeline."
            )
            st.code(
                """
def route_query(query, llm):
    messages = [
        SystemMessage(content=(
            "Classify this question as: factual, analytical, or general. "
            "Reply with ONLY the category word."
        )),
        HumanMessage(content=query)
    ]
    return llm.invoke(messages).content.strip().lower()

def branched_rag(query, vectorstore, llm):
    query_type = route_query(query, llm)

    if query_type == "factual":
        # Precise dense retrieval, fewer chunks
        docs = vectorstore.similarity_search(query, k=2)
    elif query_type == "analytical":
        # HyDE for richer retrieval
        return hyde_rag(query, vectorstore, llm)
    else:
        # Standard RAG
        docs = vectorstore.similarity_search(query, k=4)

    context = "\n\n".join([d.page_content for d in docs])
    messages = [
        SystemMessage(content="Answer using only the context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]
    return llm.invoke(messages).content
""",
                language="python",
            )

            st.markdown("---")
            st.markdown("#### Multimodal RAG")
            st.write(
                "Standard RAG only handles text. Real documents often contain images, charts, and diagrams. "
                "Multimodal RAG adds image understanding by using a vision-capable LLM (Gemini 2.0 Flash)."
            )
            st.write(
                "The pipeline works in two phases:\n"
                "- **Ingestion**: For each image, Gemini Vision generates a text caption describing its content. "
                "The caption is stored in FAISS alongside text chunks.\n"
                "- **Retrieval**: If an image chunk is retrieved, the image is sent directly to Gemini Vision "
                "for question-answering. If a text chunk is retrieved, it is handled normally."
            )
            st.code(
                """
import google.generativeai as genai
from PIL import Image
from langchain.schema import Document

genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")

def caption_image(image_path):
    image = Image.open(image_path)
    response = gemini.generate_content([
        "Describe all visual elements and text in this image in detail.",
        image
    ])
    return response.text

# Store caption in vector DB
image_doc = Document(
    page_content=caption,
    metadata={"source": "chart.png", "content_type": "image", "image_path": image_path}
)

# At retrieval time:
if doc.metadata.get("content_type") == "image":
    # Use Gemini Vision directly on the image
    answer = gemini.generate_content([question, Image.open(doc.metadata["image_path"])]).text
else:
    # Normal text-based RAG
    context.append(doc.page_content)
""",
                language="python",
            )

        # ------------------------------------------------------------------
        # 2.3 Hybrid Search
        # ------------------------------------------------------------------
        with st.expander("2.3 — Hybrid Search (Dense + Sparse + RRF)"):
            st.markdown("#### Two Types of Retrieval")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Dense Retrieval (FAISS)**")
                st.write(
                    "Converts text to vectors using an embedding model. "
                    "Finds documents semantically similar to the query. "
                    "Works even when the exact words are different. "
                    "Example: 'vector database' matches 'FAISS' even though neither word appears in the other."
                )
            with col2:
                st.markdown("**Sparse Retrieval (BM25)**")
                st.write(
                    "Classic keyword-frequency based ranking (same idea as traditional search engines). "
                    "Scores documents by how often query terms appear and how rare those terms are. "
                    "Excellent at exact term matching. "
                    "Example: searching 'FAISS' will prioritize documents containing that exact word."
                )

            st.markdown("#### Why Hybrid?")
            st.write(
                "Neither method is universally better. Dense misses exact keyword matches. "
                "Sparse misses semantic relationships. Combining them covers both cases."
            )

            st.markdown("#### BM25 Implementation")
            st.code(
                """
from rank_bm25 import BM25Okapi
import numpy as np

# Tokenize the corpus
corpus = [chunk.page_content for chunk in chunks]
tokenized = [text.lower().split() for text in corpus]
bm25 = BM25Okapi(tokenized)

def bm25_search(query, k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top_indices]
""",
                language="python",
            )

            st.markdown("#### Reciprocal Rank Fusion (RRF)")
            st.write(
                "RRF merges the ranked lists from dense and sparse retrieval into one unified ranking. "
                "Formula: for each document, its RRF score = sum of 1/(k + rank) across all lists, "
                "where k=60 is a smoothing constant. "
                "Documents that rank high in both lists rise to the top. "
                "Documents that only appear in one list still get credit but rank lower."
            )
            st.code(
                """
def reciprocal_rank_fusion(ranked_lists, k=60):
    rrf_scores = {}
    doc_map = {}

    for ranked_list in ranked_lists:
        for rank, (doc, score) in enumerate(ranked_list):
            key = hash(doc.page_content)
            # Higher rank (rank=0) contributes more: 1/(60+1) = 0.0164
            # Lower rank (rank=9) contributes less: 1/(60+10) = 0.0143
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

    sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [doc_map[key] for key in sorted_keys]

def hybrid_search(query, vectorstore, bm25, chunks, k=5):
    dense_docs = vectorstore.similarity_search(query, k=k)
    dense_ranked = [(doc, 1.0) for doc in dense_docs]

    bm25_ranked = bm25_search(query, k=k)

    return reciprocal_rank_fusion([dense_ranked, bm25_ranked])
""",
                language="python",
            )

        # ------------------------------------------------------------------
        # 2.4 Reranking
        # ------------------------------------------------------------------
        with st.expander("2.4 — Cross-Encoder Reranking"):
            st.markdown("#### The Retrieval Accuracy Problem")
            st.write(
                "FAISS retrieval is fast because it encodes query and documents independently "
                "and compares their vectors. But this independence is also a limitation — "
                "the model does not see how query and document interact with each other. "
                "Results are approximate. The top-5 retrieved documents may not be the actual top-5 most relevant."
            )

            st.markdown("#### Bi-Encoder vs Cross-Encoder")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Bi-Encoder (FAISS)**")
                st.write(
                    "Encodes query and document separately into vectors. "
                    "Fast — can search millions of docs in milliseconds. "
                    "Used for the initial broad retrieval."
                )
            with col2:
                st.markdown("**Cross-Encoder (Reranker)**")
                st.write(
                    "Takes (query, document) as a pair and processes them together. "
                    "Can model how the query and document interact. "
                    "Much more accurate but too slow to run on millions of docs. "
                    "Used on the small set of candidates from bi-encoder retrieval."
                )

            st.markdown("#### Two-Stage Pipeline")
            st.code(
                """
Query
  |
  v
Stage 1: FAISS retrieval -> top 10-15 candidate documents (fast, broad)
  |
  v
Stage 2: Cross-encoder reranking -> top 3 documents (slow, accurate)
  |
  v
LLM generation using top 3 reranked documents
""",
                language="text",
            )

            st.markdown("#### Implementation")
            st.code(
                """
from sentence_transformers import CrossEncoder

# ms-marco-MiniLM-L-6-v2: trained on MS MARCO passage ranking dataset
# Runs locally, no API key needed
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(query, vectorstore, initial_k=10, final_k=3):
    # Stage 1: Broad retrieval
    initial_docs = vectorstore.similarity_search(query, k=initial_k)

    # Stage 2: Score each (query, document) pair
    pairs = [(query, doc.page_content) for doc in initial_docs]
    scores = reranker.predict(pairs)

    # Sort by reranker score
    scored = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)

    # Return top final_k
    return [doc for score, doc in scored[:final_k]]

# The reranker assigns higher scores to documents that better answer the query
# Example scores: [3.2, 2.8, 1.1, 0.3, -0.5, ...]  (higher is better)
""",
                language="python",
            )

            st.markdown("#### Full Ultimate Pipeline")
            st.write(
                "The strongest RAG pipeline combines all three retrieval improvements: "
                "hybrid search (dense + BM25) for broad, diverse candidates, "
                "then cross-encoder reranking for accurate final selection."
            )
            st.code(
                """
def ultimate_rag(query, vectorstore, bm25, chunks, reranker, llm):
    # Step 1: Hybrid retrieval (diverse candidates)
    hybrid_docs = hybrid_search(query, vectorstore, bm25, chunks, k=10)

    # Step 2: Cross-encoder reranking (accurate selection)
    pairs = [(query, doc.page_content) for doc in hybrid_docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, hybrid_docs), reverse=True)
    top_docs = [doc for _, doc in scored[:3]]

    # Step 3: Generate
    context = "\n\n".join([d.page_content for d in top_docs])
    messages = [
        SystemMessage(content="Answer using only the context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]
    return llm.invoke(messages).content
""",
                language="python",
            )

    # =========================================================================
    # NOTEBOOK 3
    # =========================================================================
    elif module == "Notebook 3 — Evaluation and Production Chatbot":

        st.subheader("Notebook 3 — Evaluation and Production Chatbot")
        st.write(
            "This module covers how to measure RAG quality objectively using RAGAS, "
            "how to write structured tests with DeepEval, "
            "and how to build a production-ready chatbot with persistence, streaming, and citations."
        )

        # ------------------------------------------------------------------
        # 3.1 Why Evaluation Matters
        # ------------------------------------------------------------------
        with st.expander("3.1 — Why RAG Evaluation Matters", expanded=True):
            st.write(
                "Without evaluation, you are guessing. You cannot tell whether your chunking strategy "
                "is helping or hurting, whether the retriever is finding the right documents, "
                "or whether the LLM is hallucinating. Evaluation gives you numbers you can act on."
            )

            st.markdown("#### The Four Dimensions of RAG Quality")
            import pandas as pd
            
            df = pd.DataFrame({
                "Dimension": ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"],
                "Question It Answers": [
                    "Are all claims in the answer supported by retrieved context?",
                    "Does the answer actually address the question?",
                    "Are the retrieved chunks relevant to the question?",
                    "Did retrieval find all the information needed to answer?",
                ],
                "Score Direction": ["Higher is better (1 = no hallucination)", "Higher is better", "Higher is better", "Higher is better (needs ground truth)"],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("#### What You Can Do With Evaluation Scores")
            st.write(
                "- Low Faithfulness -> the LLM is hallucinating. Fix: tighten the system prompt, "
                "reduce temperature, or add a fact-check step.\n"
                "- Low Context Precision -> the retriever is pulling irrelevant chunks. "
                "Fix: reduce k, use reranking, or improve chunking.\n"
                "- Low Context Recall -> the retriever is missing relevant documents. "
                "Fix: increase k, use hybrid search, or use multi-query RAG.\n"
                "- Low Answer Relevancy -> the answer is off-topic or incomplete. "
                "Fix: improve the system prompt or retrieval quality."
            )

        # ------------------------------------------------------------------
        # 3.2 RAGAS
        # ------------------------------------------------------------------
        with st.expander("3.2 — RAGAS Evaluation Framework"):
            st.write(
                "RAGAS (Retrieval-Augmented Generation Assessment) is the most widely used "
                "open-source framework for evaluating RAG pipelines. "
                "It uses an LLM as a judge, so no manually labeled data is needed for most metrics."
            )

            st.markdown("#### How Each Metric Is Computed")
            st.write(
                "**Faithfulness**: RAGAS asks the judge LLM to extract all atomic claims from the answer "
                "(e.g. 'FAISS was created by Meta', 'FAISS supports GPU'). "
                "Then it checks each claim against the retrieved context. "
                "Score = supported claims / total claims."
            )
            st.write(
                "**Answer Relevancy**: RAGAS generates several questions from the answer, "
                "then measures how similar those reverse-engineered questions are to the original query. "
                "A high score means the answer is directly on-topic."
            )
            st.write(
                "**Context Precision**: For each retrieved chunk, the judge LLM decides whether it is relevant "
                "to the question. Score = relevant chunks / total retrieved chunks."
            )
            st.write(
                "**Context Recall**: The judge checks whether each sentence in the ground truth answer "
                "can be attributed to the retrieved context. Requires a reference answer."
            )

            st.markdown("#### RAGAS Setup and Evaluation")
            st.code(
                """
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

# Wrap our LLM and embedding model for RAGAS
ragas_llm = LangchainLLMWrapper(gemini_llm)       # Gemini as judge
ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

# Build the evaluation dataset
# Each sample needs: question, answer, contexts, ground_truth
eval_samples = [
    {
        "question": "What is FAISS?",
        "answer": rag_pipeline("What is FAISS?"),
        "contexts": [retrieved_chunk_text_1, retrieved_chunk_text_2],
        "ground_truth": "FAISS is an open-source library by Meta for fast similarity search."
    },
    # ... more samples
]

dataset = Dataset.from_list(eval_samples)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

# View results
df = results.to_pandas()
print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])
""",
                language="python",
            )

            st.markdown("#### Interpreting Scores")
            st.write(
                "All RAGAS scores are between 0 and 1. A rough guide:\n"
                "- 0.8 and above: strong performance.\n"
                "- 0.6 to 0.8: acceptable, room for improvement.\n"
                "- Below 0.6: significant issues that need addressing."
            )

        # ------------------------------------------------------------------
        # 3.3 DeepEval
        # ------------------------------------------------------------------
        with st.expander("3.3 — DeepEval Testing Framework"):
            st.write(
                "DeepEval provides a pytest-style testing interface for LLM applications. "
                "The key difference from RAGAS: DeepEval uses explicit pass/fail thresholds. "
                "If a score drops below a threshold, the test fails — just like a unit test. "
                "This is designed for CI/CD: run the test suite after every RAG system change "
                "to catch quality regressions before they reach production."
            )

            st.markdown("#### Core Concepts")
            st.write(
                "- **LLMTestCase**: Wraps one interaction (input, actual_output, retrieval_context).\n"
                "- **Metric**: Scores the test case (HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric).\n"
                "- **Threshold**: The minimum (or maximum for Hallucination) acceptable score.\n"
                "- **assert_test()**: Raises an error if any metric fails its threshold."
            )

            st.markdown("#### DeepEval Metrics")
            st.write(
                "**HallucinationMetric** (lower is better, max threshold): "
                "Detects whether the answer contains information not present in the retrieval context. "
                "A score of 0 means no hallucination. A score of 1 means the answer is entirely hallucinated.\n\n"
                "**AnswerRelevancyMetric** (higher is better, min threshold): "
                "Checks if the answer addresses the question.\n\n"
                "**FaithfulnessMetric** (higher is better, min threshold): "
                "Checks if the answer is consistent with the retrieved context."
            )

            st.code(
                """
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric

# Define metrics with thresholds
# Tests fail if these thresholds are violated
hallucination_metric = HallucinationMetric(
    threshold=0.5,    # Fail if hallucination > 50%
    model=judge_llm,
    include_reason=True
)
relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,    # Fail if relevancy < 70%
    model=judge_llm,
    include_reason=True
)
faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,    # Fail if faithfulness < 70%
    model=judge_llm,
    include_reason=True
)

# Create a test case
test_case = LLMTestCase(
    input="What is RAG?",
    actual_output="RAG is a framework that retrieves external documents to ground LLM responses.",
    retrieval_context=["RAG combines retrieval with generation to reduce hallucination."],
)

# Measure metrics
hallucination_metric.measure(test_case)
print(f"Hallucination score: {hallucination_metric.score}")
print(f"Reason: {hallucination_metric.reason}")

# Run all tests at once with a report
evaluate(
    test_cases=[test_case],
    metrics=[hallucination_metric, relevancy_metric, faithfulness_metric]
)
""",
                language="python",
            )

            st.markdown("#### When to Use RAGAS vs DeepEval")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Use RAGAS when:**")
                st.write(
                    "- Exploring metrics during development.\n"
                    "- Doing batch evaluation on a dataset.\n"
                    "- Comparing two RAG configurations."
                )
            with col2:
                st.markdown("**Use DeepEval when:**")
                st.write(
                    "- Writing automated regression tests.\n"
                    "- Integrating quality checks into CI/CD.\n"
                    "- You need explicit pass/fail with human-readable reasons."
                )

        # ------------------------------------------------------------------
        # 3.4 Production Chatbot
        # ------------------------------------------------------------------
        with st.expander("3.4 — Building a Production RAG Chatbot"):
            st.markdown("#### What Makes a Chatbot 'Production-Ready'?")
            st.write(
                "A notebook prototype is not production-ready. "
                "A production chatbot needs to handle persistence, multi-turn conversation, "
                "multiple document types, streaming output, and source attribution."
            )

            st.markdown("#### 1. Persistent Vector Store")
            st.write(
                "In a prototype, the FAISS index is rebuilt from scratch every time. "
                "In production, you embed documents once, save the index to disk, "
                "and load it on startup. No re-embedding needed."
            )
            st.code(
                """
from langchain_community.vectorstores import FAISS

# Save to disk after first build
vectorstore.save_local("faiss_index/")

# Load on subsequent startups
vectorstore = FAISS.load_local(
    "faiss_index/",
    embedding_model,
    allow_dangerous_deserialization=True
)
""",
                language="python",
            )

            st.markdown("#### 2. Multi-Document Ingestion")
            st.write(
                "A production system ingests multiple files from different sources. "
                "Each file needs to be loaded with the right loader, chunked, embedded, "
                "and tagged with source metadata."
            )
            st.code(
                """
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def ingest_documents(file_paths):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)

    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        loader = TextLoader(path) if ext == ".txt" else PyPDFLoader(path)
        docs = loader.load()

        chunks = splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = os.path.basename(path)
            chunk.metadata["chunk_id"] = f"{os.path.basename(path)}_{i}"

        all_chunks.extend(chunks)
    return all_chunks
""",
                language="python",
            )

            st.markdown("#### 3. Streaming Output")
            st.write(
                "Streaming sends tokens to the user as they are generated instead of waiting for the "
                "full response. This dramatically improves perceived responsiveness — the user sees output "
                "appearing in real time rather than a blank screen for several seconds."
            )
            st.code(
                """
from langchain_groq import ChatGroq

streaming_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    streaming=True,    # Enable streaming
)

# stream() yields AIMessageChunk objects one by one
for chunk in streaming_llm.stream(messages):
    print(chunk.content, end="", flush=True)  # Print each token immediately
""",
                language="python",
            )

            st.markdown("#### 4. Incremental Knowledge Base Updates")
            st.write(
                "When new documents arrive, you do not need to rebuild the entire index. "
                "FAISS supports adding new vectors to an existing index incrementally."
            )
            st.code(
                """
# Add new documents to existing vector store without rebuilding
new_chunks = ingest_documents(["new_document.pdf"])
vectorstore.add_documents(new_chunks)

# Save the updated index
vectorstore.save_local("faiss_index/")
""",
                language="python",
            )

            st.markdown("#### 5. Full Chatbot Class")
            st.write(
                "A clean chatbot class encapsulates all the above: "
                "retrieval, conversation memory, citation tracking, and streaming."
            )
            st.code(
                """
class RAGChatbot:
    def __init__(self, vectorstore, llm, k=3, max_history=6):
        self.vectorstore = vectorstore
        self.llm = llm
        self.k = k
        self.max_history = max_history
        self.chat_history = []

    def chat(self, user_query):
        # Retrieve
        docs = self.vectorstore.similarity_search(user_query, k=self.k)
        context = "\n\n".join([d.page_content for d in docs])

        # Build messages with history
        messages = [SystemMessage(content="Answer using the context and history.")]
        for human_msg, ai_msg in self.chat_history[-self.max_history:]:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))
        messages.append(HumanMessage(
            content=f"Context:\n{context}\n\nQuestion: {user_query}"
        ))

        # Generate
        response = self.llm.invoke(messages)
        answer = response.content

        # Track history
        self.chat_history.append((user_query, answer))

        # Return answer + sources
        sources = [doc.metadata.get("chunk_id", i) for i, doc in enumerate(docs)]
        return {"answer": answer, "sources": sources, "docs": docs}

    def reset(self):
        self.chat_history = []

chatbot = RAGChatbot(vectorstore, llm)
result = chatbot.chat("What is FAISS?")
print(result["answer"])
print("Sources:", result["sources"])
""",
                language="python",
            )

        # ------------------------------------------------------------------
        # 3.5 Concept Glossary
        # ------------------------------------------------------------------
        with st.expander("3.5 — Full Glossary of Key Terms"):
            terms = {
                "RAG (Retrieval-Augmented Generation)": (
                    "An AI framework that retrieves relevant external documents before generating a response. "
                    "Grounds LLM output in factual retrieved context."
                ),
                "Hallucination": (
                    "When an LLM generates plausible-sounding but factually incorrect information. "
                    "RAG reduces this by constraining the model to retrieved context."
                ),
                "Embedding": (
                    "A numerical vector that represents the semantic meaning of text. "
                    "Similar texts have embeddings that are close together in vector space."
                ),
                "Vector Store / Vector Database": (
                    "A database optimized for storing and searching embedding vectors. "
                    "FAISS, Chroma, Pinecone, and Qdrant are examples."
                ),
                "FAISS": (
                    "Facebook AI Similarity Search. An open-source library by Meta Research "
                    "for fast approximate nearest-neighbor search over dense vectors."
                ),
                "Chunking": (
                    "Splitting documents into smaller pieces before embedding. "
                    "Chunk size and overlap are critical hyperparameters."
                ),
                "Chunk Overlap": (
                    "The number of characters shared between consecutive chunks. "
                    "Prevents important context from being cut off at chunk boundaries."
                ),
                "Semantic Chunking": (
                    "A chunking strategy that uses embeddings to detect topic shifts and splits there, "
                    "rather than splitting on fixed character counts."
                ),
                "Hierarchical Chunking": (
                    "Uses small child chunks for retrieval and larger parent chunks for context generation. "
                    "Separates the concerns of retrieval precision and generation richness."
                ),
                "Sentence-Window Chunking": (
                    "Embeds individual sentences for retrieval, but returns the surrounding window of sentences as context."
                ),
                "Dense Retrieval": (
                    "Embedding-based retrieval that finds semantically similar documents. "
                    "Used by FAISS. Understands meaning even with different wording."
                ),
                "Sparse Retrieval (BM25)": (
                    "Keyword-frequency based retrieval. Excels at exact term matching."
                ),
                "Hybrid Search": (
                    "Combining dense and sparse retrieval results using Reciprocal Rank Fusion."
                ),
                "Reciprocal Rank Fusion (RRF)": (
                    "A rank fusion algorithm that merges multiple ranked lists. "
                    "Score = sum of 1/(k + rank) for each list the document appears in."
                ),
                "Reranking": (
                    "Post-retrieval step using a cross-encoder to accurately rescore a small set of candidates. "
                    "More accurate than bi-encoder retrieval but too slow to run on the full corpus."
                ),
                "Bi-Encoder": (
                    "An embedding model that encodes query and document independently. "
                    "Used for fast retrieval (FAISS). Less accurate than cross-encoder."
                ),
                "Cross-Encoder": (
                    "A model that processes (query, document) as a pair. "
                    "More accurate than bi-encoder. Used for reranking."
                ),
                "HyDE (Hypothetical Document Embeddings)": (
                    "A retrieval technique where the LLM first generates a hypothetical answer, "
                    "then that hypothetical answer is embedded for retrieval. "
                    "Bridges the semantic gap between queries and documents."
                ),
                "Multi-Query RAG": (
                    "Generates N phrasings of the query, retrieves documents for each, "
                    "deduplicates, and generates from the merged set. Increases recall."
                ),
                "Branched RAG": (
                    "A router LLM classifies the query type and dispatches to the most appropriate "
                    "retrieval pipeline for that type."
                ),
                "RAGAS": (
                    "Retrieval-Augmented Generation Assessment. An open-source evaluation framework "
                    "providing automated metrics: faithfulness, answer relevancy, context precision, context recall."
                ),
                "Faithfulness": (
                    "RAGAS metric. Fraction of claims in the answer that are supported by retrieved context."
                ),
                "Answer Relevancy": (
                    "RAGAS metric. How well the answer addresses the original question."
                ),
                "Context Precision": (
                    "RAGAS metric. Fraction of retrieved chunks that are relevant to the question."
                ),
                "Context Recall": (
                    "RAGAS metric. Whether all information needed to answer the question was retrieved."
                ),
                "DeepEval": (
                    "A production LLM testing framework with pytest-style pass/fail thresholds. "
                    "Designed for CI/CD regression testing of RAG systems."
                ),
                "LLM as Judge": (
                    "Using a separate LLM (Gemini, GPT-4) to score the quality of another LLM's output. "
                    "Eliminates the need for manual human evaluation at scale."
                ),
                "Parametric Memory": (
                    "Knowledge stored in LLM model weights from training. Fixed and may be outdated."
                ),
                "Non-Parametric Memory": (
                    "External knowledge retrieved at inference time. Fresh, updatable, domain-specific."
                ),
            }

            # Render as a searchable table
            import pandas as pd
            glossary_df = pd.DataFrame(
                [(term, definition) for term, definition in terms.items()],
                columns=["Term", "Definition"]
            )
            st.dataframe(glossary_df, use_container_width=True, hide_index=True)