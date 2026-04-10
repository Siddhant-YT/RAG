"""
utils/helpers.py
----------------
Utility functions used across multiple Streamlit pages/tabs:
- Document loading from uploaded files
- Session state initialization
- Text display helpers
"""

import os
import tempfile
from typing import List
import streamlit as st
from langchain_core.documents import Document


def load_documents_from_uploads(uploaded_files) -> List[Document]:
    """
    Load and return LangChain Documents from Streamlit UploadedFile objects.
    Supports .txt and .pdf files.
    Uses temporary files on disk because loaders need file paths.
    """
    documents = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        # Write the uploaded file to a temp path so LangChain loaders can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            if suffix == ".txt":
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif suffix == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(tmp_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                continue

            docs = loader.load()
            # Tag each document with the original filename
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            documents.extend(docs)

        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
        finally:
            os.unlink(tmp_path)  # Clean up temp file

    return documents


def init_session_state():
    """
    Initialize all session state keys used across the app.
    Called once at the top of app.py. Safe to call multiple times
    (only sets keys that don't already exist).
    """
    defaults = {
        # Shared knowledge base
        "documents": [],              # Raw loaded LangChain Documents
        "chunks": [],                 # Chunked documents (recursive by default)
        "vectorstore": None,          # FAISS vector store
        "bm25_index": None,           # BM25 sparse index
        "bm25_chunks": [],            # Chunks corresponding to BM25 index
        "kb_ready": False,            # Whether KB has been built

        # Chatbot conversation history (list of (user, assistant) tuples)
        "chat_history": [],

        # Evaluation samples collected during chatbot session
        "eval_samples": [],

        # Current active tab (for internal tracking)
        "active_tab": "Knowledge Base",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def display_retrieved_docs(docs, collapsed: bool = True):
    """
    Display retrieved document chunks in Streamlit expanders.
    collapsed=True means expanders are closed by default.
    """
    if not docs:
        st.info("No documents retrieved.")
        return

    for i, doc in enumerate(docs):
        cid = doc.metadata.get("chunk_id", i)
        source = doc.metadata.get("source", "unknown")
        label = f"Chunk {cid} — {source}"
        with st.expander(label, expanded=not collapsed):
            st.text(doc.page_content)


def truncate(text: str, max_chars: int = 300) -> str:
    """Truncate text for display in tables or cards."""
    return text[:max_chars] + "..." if len(text) > max_chars else text
