# RAG Studio

A hands-on Streamlit application for learning and experimenting with Retrieval-Augmented Generation (RAG).
Every concept from basic RAG to advanced patterns, evaluation, and prompt engineering is interactive and explorable.

---

## What This App Covers

| Tab | Learnings |
|-----|----------------|
| Knowledge Base | Document ingestion, chunking strategies, FAISS vector store |
| Chunking Comparison | Side-by-side comparison of all 4 chunking strategies on the same query |
| RAG Patterns | Simple RAG, Citations, HyDE, Multi-Query, Branched (Router) RAG |
| Hybrid Search and Reranking | Dense (FAISS) vs Sparse (BM25) vs Hybrid, cross-encoder reranking |
| Multimodal RAG | Image captioning with Gemini Vision, text + image knowledge base |
| Chatbot | Conversational RAG with memory across multiple turns |
| Evaluation | RAGAS metrics, DeepEval pass/fail testing |
| Learning | All the learning that I have gone through i this topic. |

---

## Project Structure

```
RAG/
│
├── app.py                          # Main Streamlit entry point
│
├── core/
│   ├── models.py                   # LLM and embedding model initialization
│   ├── rag_engine.py               # All RAG logic (chunking, retrieval, generation)
│   └── evaluation.py               # RAGAS and DeepEval evaluation
│
├── app/
│   ├── tab_knowledge_base.py       # Tab 1: Document upload and KB setup
│   ├── tab_chunking_comparison.py  # Tab 2: 4-strategy side-by-side comparison
│   ├── tab_rag_patterns.py         # Tab 3: RAG pattern explorer
│   ├── tab_retrieval.py            # Tab 4: Hybrid search and reranking
│   ├── tab_multimodal.py           # Tab 5: Multimodal RAG with Gemini Vision
│   ├── tab_chatbot.py              # Tab 6: Conversational RAG chatbot
│   ├── tab_evaluation.py           # Tab 7: RAGAS and DeepEval evaluation
│   └── tab_learning.py             # Tab 8: Learning 
│
├── utils/
│   └── helpers.py                  # Shared utilities (file loading, session state)
│
├── notebooks/
│   └── notebook1_rag_basiscs.ipynb                 # Basic RAG             
│   └── notebook2_advanced_rag.ipynb                # Chunking types and RAG patterns
│   └── notebook3_rag_evaluation_cahtbot.ipynb      # RAGAS metrics            
│
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone or download the project

```bash
cd RAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

```
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

- **GROQ_API_KEY**: Free at https://console.groq.com — used for the main LLM (llama-3.1-8b-instant)
- **GOOGLE_API_KEY**: Free at https://aistudio.google.com — used for Gemini Vision (Multimodal RAG) and RAGAS evaluation

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## How to Use

### Recommended flow

1. **Knowledge Base tab** — Start here. Use the sample corpus or upload your own `.txt`/`.pdf` files. Choose a chunking strategy and click "Build Knowledge Base".

2. **Chunking Comparison tab** — Enter a query. The app runs all 4 chunking strategies and shows the answer from each, side by side, with chunk counts.

3. **RAG Patterns tab** — Try each pattern (Simple, HyDE, Multi-Query, Branched) on the same question. Notice how the retrieved documents and answers differ.

4. **Hybrid Search and Reranking tab** — Compare what Dense, BM25, and Hybrid retrieval each return. Enable reranking to see how the cross-encoder reorders results.

5. **Multimodal RAG tab** — Upload an image, generate a Gemini caption, build a mixed knowledge base, and ask questions that span text and image content.

6. **Chatbot tab** — Have a multi-turn conversation. Ask follow-up questions that refer to previous answers. Check the sources expander to see which chunks each answer used.

7. **Evaluation tab** — QA pairs are auto-collected from the Chatbot. Add ground truth answers and run RAGAS or DeepEval to get quantitative quality scores.


---

## Models Used

| Purpose | Model | API |
|---------|-------|-----|
| Main LLM | llama-3.1-8b-instant | Groq (free) |
| Embeddings | all-MiniLM-L6-v2 | Local via sentence-transformers |
| Multimodal + Eval | gemini-3.1-flash-previe | Google AI (free tier) |
| Reranking | ms-marco-MiniLM-L-6-v2 | Local via sentence-transformers |

---

## Chunking Strategies

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| Recursive | Splits on paragraph, sentence, word in order. Fixed size + overlap. | General purpose |
| Semantic | Uses embeddings to detect topic shifts and splits there. | Documents with clear topic structure |
| Hierarchical | Small chunks embedded for retrieval; large parent chunks returned for context. | Long documents needing precision + richness |
| Sentence Window | Each sentence embedded; surrounding window returned as context. | High-precision retrieval |

---

## RAG Patterns

| Pattern | Problem It Solves |
|---------|------------------|
| Simple RAG | Baseline |
| RAG with Citations | Explainability — which source was used |
| HyDE | Query-document semantic gap |
| Multi-Query | Single phrasing misses relevant docs |
| Branched RAG | Different question types need different strategies |

---

## Notes

- All models that run locally (embeddings, reranker) are cached with `st.cache_resource` — they load once per session.
- The knowledge base is stored in `st.session_state` — it persists across tab switches but resets on page refresh.
- RAGAS uses an LLM as judge. Each evaluation call makes multiple LLM requests per sample.
- The Multimodal RAG tab writes images to a temporary file, processes them, then deletes the temp file.
