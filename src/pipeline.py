"""
pipeline.py
RAG pipeline orchestration with in-memory query cache.

Cache policy:
  - Only queries where session_turn_number == 1 are eligible.
    Turn 1 = no prior context, so route_and_retrieve runs on the raw query with
    no contextualization rewrite applied — making results safely cacheable across sessions.
  - cache_key = sha256(normalized_query + "|" + search_mode)
    sha256 is used instead of Python's built-in hash() for cross-process stability.
  - TTL      = 3600 s
  - maxsize  = 200 entries (LRU eviction when exceeded)
  - Cached   : (chunks, query_type, retrieval_query, reranked)
  - NOT cached: generate_response — non-deterministic + uses per-session context.

Public API:
    run_pipeline(query, session_id, user_id, search_mode, config) → dict
    query_cache   — module-level _QueryCache singleton (accessible for testing)
"""

import hashlib
import logging
import time
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import yaml

from src.guardrails.input_guard   import check_input
from src.guardrails.output_guard  import check_output
from src.retrieval.query_router        import route_and_retrieve, _informal_rewrite
from src.evaluation.faithfulness       import compute_faithfulness_async
from src.retrieval.reranker       import rerank
from src.generation.llm_handler   import generate_response
from src.context.session_manager  import add_turn, get_context
from src.logging.retrieval_logger import log_query

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

_config        = _load_config()
_CACHE_TTL     = int(_config.get("cache_ttl_seconds", 3600))
_CACHE_MAXSIZE = int(_config.get("cache_maxsize", 200))


# ── In-memory LRU + TTL cache ──────────────────────────────────────────────────

class _QueryCache:
    """
    Thread-safe LRU cache with TTL for RAG retrieval results.

    Keys   : sha256(normalized_query|search_mode)
    Values : (chunks, query_type, retrieval_query, reranked)
    """

    def __init__(self, maxsize: int = _CACHE_MAXSIZE, ttl: float = _CACHE_TTL):
        self._maxsize = maxsize
        self._ttl     = ttl
        self._store: OrderedDict[str, tuple] = OrderedDict()
        self._lock    = threading.Lock()

    @staticmethod
    def make_key(query: str, search_mode: str) -> str:
        normalized = query.strip().lower()
        return hashlib.sha256(f"{normalized}|{search_mode}".encode()).hexdigest()

    def get(self, query: str, search_mode: str):
        """Return cached value or None (on miss or expired entry)."""
        key = self.make_key(query, search_mode)
        with self._lock:
            if key not in self._store:
                return None
            value, ts = self._store[key]
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)   # mark as most-recently used
            return value

    def set(self, query: str, search_mode: str, value) -> None:
        """Store value; evict LRU entry if maxsize exceeded."""
        key = self.make_key(query, search_mode)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.monotonic())
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)   # evict oldest

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# Module-level singleton — shared across all requests
query_cache = _QueryCache()


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(
    query:       str,
    session_id:  str,
    user_id:     str,
    search_mode: str,
    config:      dict,
) -> dict:
    """
    Execute the full RAG pipeline for one request.

    Steps:
      1. Input guard
      2. Session context lookup
      3. Route + Retrieve  ← cache applied on turn 1
      4. Rerank            ← cache applied on turn 1
      5. Generate (always fresh)
      6. Output guard
      7. Session update
      8. Log

    Returns a dict with all fields needed to build QueryResponse, plus query_id.
    sources_cited is returned as list[dict] (caller wraps in Pydantic model).
    """
    t_start    = time.time()
    model_used = config.get("groq_model", "llama-3.1-8b-instant")

    # ── 1. Input guard ────────────────────────────────────────────────────────
    guard = check_input(query, user_id=user_id)
    if not guard["is_safe"]:
        total_ms = round((time.time() - t_start) * 1000, 1)
        qid = str(uuid.uuid4())
        log_query(
            query_text             = query,
            query_id               = qid,
            session_id             = session_id,
            input_safe             = False,
            guardrail_block_reason = guard["reason"],
            search_mode            = search_mode,
            latency_ms             = total_ms,
            rate_limit_remaining   = guard["rate_limit_status"]["requests_remaining"],
            cache_hit              = False,
        )
        return dict(
            query_id          = qid,
            answer            = guard["reason"],
            sources_cited     = [],
            query_type        = "blocked",
            search_mode_used  = search_mode,
            is_safe           = False,
            blocked_reason    = guard["reason"],
            prompt_tokens     = 0,
            completion_tokens = 0,
            latency_ms        = total_ms,
        )

    query        = guard["sanitized_query"]
    rl_remaining = guard["rate_limit_status"]["requests_remaining"]

    # ── 2. Session context ────────────────────────────────────────────────────
    session_context  = get_context(session_id)
    session_turn_num = len(session_context) + 1

    # ── 3+4. Route + Retrieve + Rerank (with cache on turn 1) ─────────────────
    cache_hit    = False
    retriever_ms = 0.0
    reranker_ms  = 0.0

    if session_turn_num == 1:
        cached = query_cache.get(query, search_mode)
        if cached is not None:
            chunks, query_type, retrieval_query, reranked = cached
            cache_hit = True
            logger.info(
                "Cache HIT  — session=%s query='%s...' mode=%s",
                session_id, query[:50], search_mode,
            )

    rewrite_attempted = False

    if not cache_hit:
        t0 = time.time()
        chunks, query_type, retrieval_query = route_and_retrieve(
            query,
            search_mode     = search_mode,
            session_context = session_context,
        )
        retriever_ms = round((time.time() - t0) * 1000, 1)

        if query_type == "multi_part":
            # Already reranked per-sub-query inside route_and_retrieve
            reranked    = chunks
            reranker_ms = 0.0
        else:
            t0 = time.time()
            reranked = rerank(retrieval_query, chunks)
            reranker_ms = round((time.time() - t0) * 1000, 1)

        # ── Rewrite fallback: retry with informal rewrite if reranker score is low ──
        confidence_threshold = config.get("reranker_confidence_threshold", 1.5)
        top_score            = reranked[0]["reranker_score"] if reranked else 0.0

        if top_score < confidence_threshold and query_type != "multi_part":
            rewritten = _informal_rewrite(retrieval_query, model_used)
            if rewritten != retrieval_query:
                rewrite_attempted = True
                t0 = time.time()
                retry_chunks, _, _ = route_and_retrieve(
                    rewritten,
                    search_mode     = search_mode,
                    session_context = session_context,
                )
                retriever_ms += round((time.time() - t0) * 1000, 1)

                t0 = time.time()
                retry_reranked = rerank(rewritten, retry_chunks)
                reranker_ms   += round((time.time() - t0) * 1000, 1)

                retry_top = retry_reranked[0]["reranker_score"] if retry_reranked else 0.0
                if retry_top > top_score:
                    chunks          = retry_chunks
                    reranked        = retry_reranked
                    retrieval_query = rewritten
                    logger.info(
                        "Rewrite fallback improved score %.4f → %.4f | query='%s'",
                        top_score, retry_top, rewritten[:60],
                    )
                else:
                    logger.info(
                        "Rewrite fallback did not improve score (%.4f vs %.4f) — keeping original",
                        retry_top, top_score,
                    )

        if session_turn_num == 1:
            query_cache.set(query, search_mode, (chunks, query_type, retrieval_query, reranked))
            logger.info(
                "Cache MISS — stored. session=%s query='%s...' mode=%s",
                session_id, query[:50], search_mode,
            )
        else:
            logger.debug(
                "Cache SKIP (turn %d) — session=%s query='%s...'",
                session_turn_num, session_id, query[:50],
            )

    avg_ret_score      = round(sum(c["score"] for c in chunks) / len(chunks), 4) if chunks else 0.0
    top_chunk_score    = reranked[0]["reranker_score"]  if reranked else None
    min_reranker_score = reranked[-1]["reranker_score"] if reranked else None

    # ── 5. Generate ───────────────────────────────────────────────────────────
    t0 = time.time()
    if reranked:
        llm_result = generate_response(
            retrieval_query,
            reranked,
            session_context=session_context,
        )
    else:
        llm_result = {
            "answer":            "I could not find relevant information to answer your question.",
            "sources_cited":     [],
            "model":             model_used,
            "chunks_used":       0,
            "prompt_tokens":     0,
            "completion_tokens": 0,
        }
    llm_ms = round((time.time() - t0) * 1000, 1)

    answer            = llm_result["answer"]
    sources_cited     = llm_result["sources_cited"]
    prompt_tokens     = llm_result["prompt_tokens"]
    completion_tokens = llm_result["completion_tokens"]
    model_used        = llm_result.get("model", model_used)

    # ── 6. Output guard ───────────────────────────────────────────────────────
    if not sources_cited:
        out_guard   = {"is_safe": True, "reason": "OK", "filtered_response": answer}
        output_safe = True
    else:
        out_guard   = check_output(answer)
        output_safe = out_guard["is_safe"]
        if not output_safe:
            answer = out_guard["filtered_response"]

    # ── 7. Session update ─────────────────────────────────────────────────────
    add_turn(session_id, "user",      query)
    add_turn(session_id, "assistant", answer)

    total_ms = round((time.time() - t_start) * 1000, 1)

    # ── 8. Log ────────────────────────────────────────────────────────────────
    qid = log_query(
        query_text             = query,
        session_id             = session_id,
        input_safe             = True,
        output_safe            = output_safe,
        guardrail_block_reason = None if output_safe else out_guard["reason"],
        search_mode            = search_mode,
        chunks_retrieved       = len(chunks),
        chunks_reranked        = len(reranked),
        top_chunk_score        = top_chunk_score,
        avg_retriever_score    = avg_ret_score,
        min_reranker_score     = min_reranker_score,
        sources_cited          = sources_cited,
        citation_count         = len(sources_cited),
        response_text          = answer,
        model_used             = model_used,
        prompt_tokens          = prompt_tokens,
        completion_tokens      = completion_tokens,
        latency_ms             = total_ms,
        retriever_latency_ms   = retriever_ms,
        reranker_latency_ms    = reranker_ms,
        llm_latency_ms         = llm_ms,
        rate_limit_remaining   = rl_remaining,
        session_turn_number    = session_turn_num,
        cache_hit              = cache_hit,
        rewrite_attempted      = rewrite_attempted,
    )

    # ── 9. Async faithfulness (fire-and-forget, does not block response) ──────
    if (
        config.get("enable_async_faithfulness", False)
        and reranked
        and sources_cited          # only when LLM produced a grounded answer
        and not cache_hit          # skip: cached results were already evaluated on first call
    ):
        contexts = [c["text"] for c in reranked]
        compute_faithfulness_async(qid, answer, contexts, model_used)

    return dict(
        query_id          = qid,
        answer            = answer,
        sources_cited     = sources_cited,
        query_type        = query_type,
        search_mode_used  = search_mode,
        is_safe           = True,
        blocked_reason    = None if output_safe else out_guard["reason"],
        prompt_tokens     = prompt_tokens,
        completion_tokens = completion_tokens,
        latency_ms        = total_ms,
    )
