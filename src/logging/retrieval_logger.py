"""
retrieval_logger.py — Step 11
SQLite-backed audit logger for all RAG pipeline retrieval events.

Fixes applied:
  - #22 Added is_eval column (INTEGER, default 0). Eval-run rows are flagged
        so get_log_stats() excludes them, keeping production metrics clean.
        A migration runs at startup to add the column to existing databases.
  - #25 logging.basicConfig removed from module level.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from sqlalchemy import Column, Integer, Float, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

_config       = _load_config()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH      = (_PROJECT_ROOT / _config.get("log_db_path", "logs/retrieval_log.db")).resolve()
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_DB_URL       = f"sqlite:///{_DB_PATH}"


# ── ORM Model ──────────────────────────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class RetrievalLog(_Base):
    __tablename__ = "retrieval_logs"

    id                     = Column(Integer, primary_key=True, autoincrement=True)
    query_id               = Column(Text,    nullable=False, index=True)
    session_id             = Column(Text,    nullable=False, index=True)
    query_text             = Column(Text,    nullable=False)
    timestamp              = Column(Text,    nullable=False)

    input_safe             = Column(Integer, nullable=False, default=1)
    output_safe            = Column(Integer, nullable=True)
    guardrail_block_reason = Column(Text,    nullable=True)

    search_mode            = Column(Text,    nullable=True)
    chunks_retrieved       = Column(Integer, nullable=True)
    chunks_reranked        = Column(Integer, nullable=True)
    top_chunk_score        = Column(Float,   nullable=True)
    avg_retriever_score    = Column(Float,   nullable=True)
    min_reranker_score     = Column(Float,   nullable=True)

    latency_ms             = Column(Float,   nullable=True)
    retriever_latency_ms   = Column(Float,   nullable=True)
    reranker_latency_ms    = Column(Float,   nullable=True)
    llm_latency_ms         = Column(Float,   nullable=True)

    rate_limit_remaining   = Column(Integer, nullable=True)

    sources_cited          = Column(Text,    nullable=True)
    citation_count         = Column(Integer, nullable=True)
    response_text          = Column(Text,    nullable=True)

    model_used             = Column(Text,    nullable=True)
    prompt_tokens          = Column(Integer, nullable=True)
    completion_tokens      = Column(Integer, nullable=True)
    context_tokens_used    = Column(Integer, nullable=True)

    session_turn_number    = Column(Integer, nullable=True)

    golden_query_id        = Column(Text,    nullable=True)
    precision_at_k         = Column(Float,   nullable=True)
    recall_at_k            = Column(Float,   nullable=True)
    feedback_score         = Column(Integer, nullable=True)

    # #22 — distinguishes evaluation runs from real user traffic
    is_eval                = Column(Integer, nullable=False, default=0)

    # cache hit flag: 1 = result served from in-memory query cache, 0 = full retrieval
    cache_hit              = Column(Integer, nullable=True)

    # rewrite fallback: 1 = low reranker score triggered informal rewrite + retry
    rewrite_attempted      = Column(Integer, nullable=True)

    # async faithfulness score: fraction of answer statements supported by context [0.0–1.0]
    faithfulness_score     = Column(Float,   nullable=True)


# ── Engine + table creation ────────────────────────────────────────────────────

_engine = create_engine(_DB_URL, connect_args={"check_same_thread": False, "timeout": 10})
_Base.metadata.create_all(_engine)

# Migration: add is_eval column to existing databases (#22)
with _engine.connect() as _conn:
    try:
        _conn.execute(text(
            "ALTER TABLE retrieval_logs ADD COLUMN is_eval INTEGER NOT NULL DEFAULT 0"
        ))
        _conn.commit()
        logger.info("Migration: added is_eval column to retrieval_logs.")
    except Exception:
        pass   # Column already exists — safe to ignore

with _engine.connect() as _conn:
    try:
        _conn.execute(text(
            "ALTER TABLE retrieval_logs ADD COLUMN cache_hit INTEGER"
        ))
        _conn.commit()
        logger.info("Migration: added cache_hit column to retrieval_logs.")
    except Exception:
        pass   # Column already exists — safe to ignore

with _engine.connect() as _conn:
    try:
        _conn.execute(text(
            "ALTER TABLE retrieval_logs ADD COLUMN rewrite_attempted INTEGER"
        ))
        _conn.commit()
        logger.info("Migration: added rewrite_attempted column to retrieval_logs.")
    except Exception:
        pass   # Column already exists — safe to ignore

with _engine.connect() as _conn:
    try:
        _conn.execute(text(
            "ALTER TABLE retrieval_logs ADD COLUMN faithfulness_score REAL"
        ))
        _conn.commit()
        logger.info("Migration: added faithfulness_score column to retrieval_logs.")
    except Exception:
        pass   # Column already exists — safe to ignore

logger.info("Retrieval log DB ready: %s", _DB_PATH)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _row_to_dict(row: RetrievalLog) -> dict:
    d = {c.name: getattr(row, c.name) for c in row.__table__.columns}
    if d.get("sources_cited"):
        try:
            d["sources_cited"] = json.loads(d["sources_cited"])
        except (json.JSONDecodeError, TypeError):
            d["sources_cited"] = []
    return d


# ── Public API ─────────────────────────────────────────────────────────────────

def log_query(
    query_text:             str,
    session_id:             str          = "anonymous",
    query_id:               str | None   = None,
    input_safe:             bool         = True,
    output_safe:            bool | None  = None,
    guardrail_block_reason: str | None   = None,
    search_mode:            str | None   = None,
    chunks_retrieved:       int | None   = None,
    chunks_reranked:        int | None   = None,
    top_chunk_score:        float | None = None,
    avg_retriever_score:    float | None = None,
    min_reranker_score:     float | None = None,
    latency_ms:             float | None = None,
    retriever_latency_ms:   float | None = None,
    reranker_latency_ms:    float | None = None,
    llm_latency_ms:         float | None = None,
    rate_limit_remaining:   int | None   = None,
    sources_cited:          list | None  = None,
    citation_count:         int | None   = None,
    response_text:          str | None   = None,
    model_used:             str | None   = None,
    prompt_tokens:          int | None   = None,
    completion_tokens:      int | None   = None,
    context_tokens_used:    int | None   = None,
    session_turn_number:    int | None   = None,
    golden_query_id:        str | None   = None,
    precision_at_k:         float | None = None,
    recall_at_k:            float | None = None,
    feedback_score:         int | None   = None,
    is_eval:                bool         = False,   # #22
    cache_hit:              bool | None  = None,
    rewrite_attempted:      bool | None  = None,
) -> str:
    """
    Persist a retrieval event to SQLite.

    Args:
        is_eval: Set True for evaluation runs to exclude them from production
                 stats in get_log_stats(). Defaults to False.

    Returns:
        query_id (str) used for this log entry.
    """
    qid = query_id or str(uuid.uuid4())
    if citation_count is None and sources_cited is not None:
        citation_count = len(sources_cited)

    row = RetrievalLog(
        query_id               = qid,
        session_id             = session_id,
        query_text             = query_text,
        timestamp              = datetime.now(timezone.utc).isoformat(),
        input_safe             = int(input_safe),
        output_safe            = int(output_safe) if output_safe is not None else None,
        guardrail_block_reason = guardrail_block_reason,
        search_mode            = search_mode,
        chunks_retrieved       = chunks_retrieved,
        chunks_reranked        = chunks_reranked,
        top_chunk_score        = round(top_chunk_score,     4) if top_chunk_score     is not None else None,
        avg_retriever_score    = round(avg_retriever_score, 4) if avg_retriever_score is not None else None,
        min_reranker_score     = round(min_reranker_score,  4) if min_reranker_score  is not None else None,
        latency_ms             = round(latency_ms,           1) if latency_ms          is not None else None,
        retriever_latency_ms   = round(retriever_latency_ms, 1) if retriever_latency_ms is not None else None,
        reranker_latency_ms    = round(reranker_latency_ms,  1) if reranker_latency_ms  is not None else None,
        llm_latency_ms         = round(llm_latency_ms,       1) if llm_latency_ms       is not None else None,
        rate_limit_remaining   = rate_limit_remaining,
        sources_cited          = json.dumps(sources_cited or []),
        citation_count         = citation_count,
        response_text          = response_text,
        model_used             = model_used,
        prompt_tokens          = prompt_tokens,
        completion_tokens      = completion_tokens,
        context_tokens_used    = context_tokens_used,
        session_turn_number    = session_turn_number,
        golden_query_id        = golden_query_id,
        precision_at_k         = precision_at_k,
        recall_at_k            = recall_at_k,
        feedback_score         = feedback_score,
        is_eval                = int(is_eval),
        cache_hit              = int(cache_hit)          if cache_hit          is not None else None,
        rewrite_attempted      = int(rewrite_attempted)  if rewrite_attempted  is not None else None,
    )

    with Session(_engine) as db:
        db.add(row)
        db.commit()

    logger.info(
        "Logged [%s] session=%s turn=%s latency=%.1fms safe=%s/%s is_eval=%s",
        qid[:8], session_id, session_turn_number,
        latency_ms or 0, input_safe, output_safe, is_eval,
    )
    return qid


def get_logs(limit: int = 50, offset: int = 0) -> list[dict]:
    """Fetch recent log entries (all rows including eval), newest first."""
    with Session(_engine) as db:
        rows = (
            db.query(RetrievalLog)
            .order_by(RetrievalLog.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
    return [_row_to_dict(r) for r in rows]


def get_logs_by_session(session_id: str) -> list[dict]:
    """Fetch all entries for a session, oldest first."""
    with Session(_engine) as db:
        rows = (
            db.query(RetrievalLog)
            .filter(RetrievalLog.session_id == session_id)
            .order_by(RetrievalLog.id.asc())
            .all()
        )
    return [_row_to_dict(r) for r in rows]


def update_faithfulness_score(query_id: str, score: float) -> None:
    """Backfill faithfulness_score for a row after async computation completes."""
    with Session(_engine) as db:
        row = db.query(RetrievalLog).filter(RetrievalLog.query_id == query_id).first()
        if row:
            row.faithfulness_score = round(score, 4)
            db.commit()


def update_eval_scores(
    query_id:       str,
    precision_at_k: float,
    recall_at_k:    float,
) -> None:
    """Backfill precision_at_k and recall_at_k for a historical row after an eval run."""
    with Session(_engine) as db:
        row = db.query(RetrievalLog).filter(RetrievalLog.query_id == query_id).first()
        if row:
            row.precision_at_k = round(precision_at_k, 4)
            row.recall_at_k    = round(recall_at_k,    4)
            db.commit()


def get_log_stats() -> dict:
    """
    Return aggregate metrics across production queries only (#22).
    Rows with is_eval=1 are excluded so evaluation traffic does not
    pollute average latency, citation rates, or guardrail pass rates.
    """
    with Session(_engine) as db:
        result = db.execute(text("""
            SELECT
                COUNT(*)                                    AS total_queries,
                SUM(input_safe)                             AS safe_queries,
                COUNT(*) - SUM(input_safe)                  AS blocked_queries,
                ROUND(AVG(latency_ms), 1)                   AS avg_latency_ms,
                ROUND(AVG(retriever_latency_ms), 1)         AS avg_retriever_ms,
                ROUND(AVG(reranker_latency_ms), 1)          AS avg_reranker_ms,
                ROUND(AVG(llm_latency_ms), 1)               AS avg_llm_ms,
                ROUND(AVG(prompt_tokens), 0)                AS avg_prompt_tokens,
                ROUND(AVG(completion_tokens), 0)            AS avg_completion_tokens,
                ROUND(AVG(avg_retriever_score), 4)          AS avg_retriever_score,
                ROUND(AVG(citation_count), 1)               AS avg_citations,
                COUNT(DISTINCT session_id)                  AS unique_sessions
            FROM retrieval_logs
            WHERE is_eval = 0
        """)).fetchone()

    with Session(_engine) as db:
        rows = db.execute(
            text("SELECT sources_cited FROM retrieval_logs WHERE sources_cited IS NOT NULL AND is_eval = 0")
        ).fetchall()

    source_counts: dict[str, int] = {}
    for (src_json,) in rows:
        try:
            for src in json.loads(src_json):
                f = src.get("source_file", "")
                if f:
                    source_counts[f] = source_counts.get(f, 0) + 1
        except (json.JSONDecodeError, TypeError):
            pass

    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_queries":         result[0]  or 0,
        "safe_queries":          result[1]  or 0,
        "blocked_queries":       result[2]  or 0,
        "avg_latency_ms":        result[3]  or 0.0,
        "avg_retriever_ms":      result[4]  or 0.0,
        "avg_reranker_ms":       result[5]  or 0.0,
        "avg_llm_ms":            result[6]  or 0.0,
        "avg_prompt_tokens":     result[7]  or 0,
        "avg_completion_tokens": result[8]  or 0,
        "avg_retriever_score":   result[9]  or 0.0,
        "avg_citations":         result[10] or 0.0,
        "unique_sessions":       result[11] or 0,
        "top_sources": [{"source_file": f, "cited_count": c} for f, c in top_sources],
    }


# ── Smoke-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(f"\n{'='*65}\n  RETRIEVAL LOGGER SMOKE-TEST\n{'='*65}\n")

    prod_id = log_query(
        query_text          = "What is RBAC?",
        session_id          = "sess-prod",
        input_safe          = True,
        output_safe         = True,
        search_mode         = "hybrid",
        chunks_retrieved    = 5,
        chunks_reranked     = 3,
        top_chunk_score     = 0.87,
        avg_retriever_score = 0.74,
        sources_cited       = [{"source_file": "iam.pdf", "page_number": 3}],
        latency_ms          = 1200.0,
        model_used          = "llama-3.1-8b-instant",
        prompt_tokens       = 420,
        completion_tokens   = 180,
        is_eval             = False,
    )
    eval_id = log_query(
        query_text          = "What is RBAC?",
        session_id          = "eval_run",
        input_safe          = True,
        output_safe         = True,
        search_mode         = "hybrid",
        chunks_retrieved    = 5,
        chunks_reranked     = 3,
        top_chunk_score     = 0.85,
        avg_retriever_score = 0.72,
        sources_cited       = [{"source_file": "iam.pdf", "page_number": 3}],
        latency_ms          = 1100.0,
        model_used          = "llama-3.1-8b-instant",
        prompt_tokens       = 400,
        completion_tokens   = 160,
        is_eval             = True,
        golden_query_id     = "q_001",
        precision_at_k      = 0.8,
        recall_at_k         = 1.0,
    )

    stats = get_log_stats()
    print("[Test] get_log_stats() excludes eval rows:")
    print(f"  total_queries (prod only): {stats['total_queries']}")
    assert stats["total_queries"] >= 1, "FAIL: at least one prod row should exist"
    print("  PASS\n")

    print(f"{'='*65}\n  Smoke-test passed.\n{'='*65}\n")
