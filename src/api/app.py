"""
app.py — Step 13
FastAPI application serving the Enterprise RAG pipeline over HTTP.

Endpoints:
  POST /query  — full RAG pipeline (input guard → route → retrieve → rerank
                 → generate → output guard → session update → log)
  GET  /health — liveness + config snapshot
  GET  /logs   — recent retrieval log entries (production queries only)
  GET  /eval   — latest evaluation report

Startup:
  Lifespan handler calls retriever.warm_up() to pre-load the embedding model,
  eliminating the cold-start latency spike on the first real request.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from src.retrieval.retriever      import warm_up
from src.logging.retrieval_logger import get_logs, get_log_stats
from src.pipeline                 import run_pipeline, query_cache

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

_config = _load_config()


# ── Lifespan (startup warmup) ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: warming up embedding model...")
    warm_up()
    logger.info("Startup complete — pipeline ready.")
    yield
    logger.info("Shutdown.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise RAG API",
    description="Security Analytics RAG pipeline for enterprise SOC teams.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── GET / ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


# ── Request / Response schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:       str
    session_id:  str           = "anonymous"
    user_id:     str           = "anonymous"
    search_mode: Optional[Literal["vector", "bm25", "hybrid"]] = None   # overrides config if provided


class SourceCited(BaseModel):
    source_file:    str
    page_number:    int
    section_header: str


class QueryResponse(BaseModel):
    query_id:          str
    answer:            str
    sources_cited:     list[SourceCited]
    query_type:        str
    search_mode_used:  str
    is_safe:           bool
    blocked_reason:    Optional[str]
    prompt_tokens:     int
    completion_tokens: int
    latency_ms:        float


# ── POST /query ────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    search_mode = req.search_mode or _config.get("search_mode", "hybrid")
    result = run_pipeline(
        query       = req.query,
        session_id  = req.session_id,
        user_id     = req.user_id,
        search_mode = search_mode,
        config      = _config,
    )
    return QueryResponse(
        **{k: v for k, v in result.items() if k != "sources_cited"},
        sources_cited=[SourceCited(**s) for s in result["sources_cited"]],
    )


# ── POST /cache/clear ──────────────────────────────────────────────────────────

@app.post("/cache/clear")
def cache_clear():
    """
    Wipe the in-memory query cache.
    Call this after ingesting new documents so stale retrieval results
    are not served to users.
    """
    before = query_cache.size()
    query_cache.clear()
    logger.info("Query cache cleared (%d entries removed).", before)
    return {"cleared": True, "entries_removed": before}


# ── GET /health ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model":       _config.get("groq_model"),
        "search_mode": _config.get("search_mode"),
        "top_k":       _config.get("top_k"),
        "reranker_top_n": _config.get("reranker_top_n"),
    }


# ── GET /logs ──────────────────────────────────────────────────────────────────

@app.get("/logs")
def logs_endpoint(limit: int = Query(default=50, ge=1, le=500)):
    entries = get_logs(limit=limit)
    stats   = get_log_stats()
    return {"stats": stats, "logs": entries}


# ── GET /eval ──────────────────────────────────────────────────────────────────

@app.get("/eval")
def eval_endpoint():
    report_path = Path(_config["eval_report_path"])
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation report not found. Run the evaluator first: python -m src.evaluation.evaluator",
        )
    with open(report_path) as f:
        return json.load(f)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
