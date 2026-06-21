# SecureQueryRAG

A production-grade Retrieval-Augmented Generation (RAG) system for enterprise Security Operations Centre (SOC) teams. Analysts ask natural language questions against a private knowledge base of security policy and compliance documents and receive accurate, cited answers — without data leaving the organisation.

---

## Key Features

- **Hybrid Search** — Combines dense vector search (ChromaDB) and BM25 keyword search fused with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking** — Scores every retrieved chunk against the query jointly for precision beyond embedding similarity
- **Query Normalisation** — 4-step pipeline: prefix stripping, grammar fixes, multi-part decomposition, informal rewrite
- **Rewrite Fallback** — Automatically retries retrieval with an LLM-rewritten query when reranker confidence is low
- **Multi-turn Sessions** — 5-turn conversation window with pronoun resolution for follow-up queries
- **Guardrails** — PII detection, prompt injection blocking, rate limiting on input; hallucination and sensitive content filtering on output
- **RAGAS Faithfulness** — LLM-as-judge evaluates whether every statement in the answer is supported by retrieved context
- **Async Production Scoring** — Faithfulness runs in a background thread per live query with zero latency impact on the user
- **SQLite Audit Log** — Every query event persisted with latency breakdown, reranker scores, citations, cache status, and faithfulness score
- **REST API + Chat UI** — FastAPI backend with Streamlit frontend

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq — `llama-3.1-8b-instant` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector Store | ChromaDB (persistent, local) |
| Keyword Search | BM25 (`rank-bm25`) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Database | SQLite via SQLAlchemy ORM |
| PDF Parsing | PyPDF |

---

## Architecture

```
User Query
    |
[Input Guard]          PII, prompt injection, rate limiting
    |
[Query Normalisation]  Strip noise, fix grammar, decompose multi-part, informal rewrite
    |
[Multi-turn Rewrite]   Resolve pronouns using session history (turn 2+)
    |
[Hybrid Retrieval]     ChromaDB vector + BM25, fused with RRF (top 10)
    |
[Rerank]               Cross-encoder scores each chunk; keeps top 4
    |
[Rewrite Fallback]     If top score < threshold: rewrite + retry once
    |
[LLM Generation]       Groq LLaMA 3.1, grounded strictly in retrieved context
    |
[Output Guard]         Safety check on generated answer
    |
[Log + Async Faith]    SQLite audit + background faithfulness scoring
    |
Answer returned to user
```

---

## Evaluation Results

| Metric | Score |
|--------|-------|
| Mean Precision@k | 0.6250 |
| Mean Recall@k | 0.8000 |
| MRR | 0.6889 |
| Keyword Hit Rate | 0.5511 |
| Mean Faithfulness (RAGAS) | 0.7564 |

Evaluated on 30 hand-crafted security domain Q&A pairs.

---

## Project Structure

```
enterprise_rag/
├── config/config.yaml          # All tunables
├── data/golden_dataset.json    # 30 evaluation Q&A pairs
├── src/
│   ├── pipeline.py             # Main orchestrator (9 steps)
│   ├── ingestion/              # PDF loading, chunking, embedding
│   ├── retrieval/              # Hybrid search, reranking, query routing
│   ├── generation/             # LLM handler
│   ├── guardrails/             # Input and output safety
│   ├── context/                # Session manager
│   ├── logging/                # SQLite audit logger
│   ├── evaluation/             # Offline metrics + RAGAS faithfulness
│   └── api/app.py              # FastAPI server
├── ui/chat_app.py              # Streamlit UI
├── generate_kb.py              # Build vector index from PDFs
├── requirements.txt
└── test_*.py                   # Test suite
```

---

## Quick Start

```bash
# 1. Setup
python -m venv rag_env
source rag_env/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env

# 2. Add PDFs to data/raw/ then build the knowledge base
python generate_kb.py

# 3. Start the API
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# 4. Start the chat UI
streamlit run ui/chat_app.py
```

You will need a [Groq API key](https://console.groq.com) (free tier available).

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Run full RAG pipeline |
| `GET` | `/health` | Liveness check + config snapshot |
| `GET` | `/logs` | Recent query log with aggregate stats |
| `GET` | `/eval` | Latest evaluation report |
| `POST` | `/cache/clear` | Invalidate query cache after KB update |

---

## License

MIT
