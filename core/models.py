"""
core/models.py
--------------
Central place to initialize all LLMs and embedding models.
Imported by every other module that needs them.
Uses st.cache_resource so models are loaded once per Streamlit session.
"""

import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()


@st.cache_resource(show_spinner=False)
def load_groq_llm():
    """
    Load the Groq LLM (llama-3.1-8b-instant).
    Used for general RAG, chunking, hybrid search, reranking etc.
    cache_resource ensures this is only loaded once across all reruns.
    """
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=1000,
    )
    return llm


@st.cache_resource(show_spinner=False)
def load_groq_streaming_llm():
    """
    Groq LLM with streaming enabled — used for the chatbot tab
    where we want to show tokens as they are generated.
    """
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=800,
        streaming=True,
    )
    return llm


@st.cache_resource(show_spinner=False)
def load_gemini_llm():
    """
    Load Gemini 2.0 Flash via LangChain.
    Used for RAGAS evaluation (as judge LLM) and Multimodal RAG.
    Requires GOOGLE_API_KEY in .env
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )
    return gemini_llm


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """
    Load HuggingFace sentence-transformers embedding model.
    all-MiniLM-L6-v2: 384-dim, runs locally, no API key needed.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


@st.cache_resource(show_spinner=False)
def load_reranker():
    """
    Load the cross-encoder reranking model.
    ms-marco-MiniLM-L-6-v2: scores (query, document) pairs together.
    Runs locally, no API key needed.
    """
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return reranker
