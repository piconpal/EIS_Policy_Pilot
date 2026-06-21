"""
generate_kb.py - Synthetic Knowledge Base PDF Generator for Enterprise RAG

Generates 35 domain-coherent security PDFs using reportlab across 7 domains:
  - IAM (Identity & Access Management)
  - SIEM (Security Information & Event Management)
  - UEBA (User & Entity Behaviour Analytics)
  - Vulnerability Management
  - Data Protection
  - Fraud Detection
  - ASM (Attack Surface Management)

Each PDF contains structured policy content including tables, charts, and
section headings that the RAG pipeline's chunker and embedder can process.

After PDF generation, the script attempts to clear the in-memory query cache
on the running RAG server (POST /cache/clear on localhost:8000) so stale
retrieval results are not served after the knowledge base is updated.

Usage:
    python generate_kb.py

Output:
    data/raw/       — 35 generated PDFs
    data/processed/ — chunked text (created by ingestion pipeline)
    vectorstore/    — ChromaDB embeddings (created by embedder)

Note: PDF content is proprietary. This file is intentionally stubbed.
"""

# PDF generation logic is proprietary and not included in this repository.
# To build your own knowledge base:
#   1. Place your own PDF documents in data/raw/
#   2. Run the ingestion pipeline: python -m src.ingestion.embedder
#   3. Clear cache if server is running: POST http://localhost:8000/cache/clear
