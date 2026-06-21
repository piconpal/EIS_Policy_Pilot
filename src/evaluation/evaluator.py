"""
evaluator.py — Step 12
Evaluates retrieval and answer quality against golden_dataset.json.

Metrics computed per query:
  Precision@k      — fraction of reranked chunks from a relevant doc
  Recall@k         — 1.0 if any reranked chunk is from a relevant doc, else 0.0
  Reciprocal Rank  — 1/rank of the first relevant chunk (for MRR)
  Keyword Hit Rate — fraction of expected_answer_keywords found in LLM answer

Fixes applied:
  - #6  config loaded once at module level (not per eval run)
  - #22 log_query() called with is_eval=True so eval rows are excluded from
        production stats in get_log_stats()
  - #25 logging.basicConfig removed from module level
"""

import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

import yaml

from src.retrieval.retriever      import retrieve
from src.retrieval.reranker       import rerank
from src.generation.llm_handler   import generate_response
from src.logging.retrieval_logger import log_query, update_eval_scores

logger = logging.getLogger(__name__)


# ── Module-level config (#6) ───────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

_config: dict = _load_config()


# ── Token-aware rate limiter ───────────────────────────────────────────────────

class _TokenPacer:
    """
    Ensures the evaluator stays within Groq's tokens-per-minute (TPM) limit.

    Tracks tokens consumed in a rolling 60-second window. Before each LLM call,
    checks whether the next query's estimated token usage would breach the limit.
    If it would, sleeps exactly long enough for the current window to expire and
    the bucket to reset — then proceeds immediately.

    Estimate strategy: running average of actual tokens used so far in the run.
    Converges toward the true per-query cost as more queries complete.
    Falls back to a conservative 1,000-token estimate before the first query.
    """

    _WINDOW = 60.0   # Groq TPM window in seconds

    def __init__(self, tpm_limit: int):
        self._tpm_limit      = tpm_limit
        self._window_start   = time.monotonic()
        self._tokens_in_window = 0
        self._total_tokens   = 0
        self._queries_done   = 0

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Call this immediately after each LLM response with actual token counts."""
        used = prompt_tokens + completion_tokens
        self._tokens_in_window += used
        self._total_tokens     += used
        self._queries_done     += 1

    def wait_if_needed(self) -> None:
        """
        Call this BEFORE each LLM call.
        Sleeps if the next query would push the window over the TPM limit.
        """
        # Roll the window if 60s have passed
        elapsed = time.monotonic() - self._window_start
        if elapsed >= self._WINDOW:
            self._window_start     = time.monotonic()
            self._tokens_in_window = 0
            return   # fresh window — no sleep needed

        # Estimate tokens for the next query
        estimated_next = (
            math.ceil(self._total_tokens / self._queries_done)
            if self._queries_done > 0
            else 1000   # conservative fallback before first query
        )

        if self._tokens_in_window + estimated_next > self._tpm_limit:
            sleep_for = self._WINDOW - elapsed
            logger.info(
                "TPM pacer: used %d/%d tokens in window, next est. ~%d tokens. "
                "Sleeping %.1fs until window resets.",
                self._tokens_in_window, self._tpm_limit, estimated_next, sleep_for,
            )
            print(
                f"\n  [Rate limiter] {self._tokens_in_window}/{self._tpm_limit} tokens used "
                f"in window — sleeping {sleep_for:.1f}s for bucket reset...",
                flush=True,
            )
            time.sleep(sleep_for)
            self._window_start     = time.monotonic()
            self._tokens_in_window = 0


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _precision_at_k(retrieved_sources: list[str], relevant_docs: list[str], k: int) -> float:
    top_k = retrieved_sources[:k]
    hits  = sum(1 for s in top_k if s in relevant_docs)
    return round(hits / k, 4) if k > 0 else 0.0


def _recall_at_k(retrieved_sources: list[str], relevant_docs: list[str], k: int) -> float:
    top_k = retrieved_sources[:k]
    return 1.0 if any(s in relevant_docs for s in top_k) else 0.0


def _reciprocal_rank(retrieved_sources: list[str], relevant_docs: list[str]) -> float:
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in relevant_docs:
            return round(1.0 / rank, 4)
    return 0.0


def _keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 4)


# ── Core evaluation loop ───────────────────────────────────────────────────────

def run_evaluation(
    golden_dataset_path: str | None = None,
    top_k:               int | None = None,
    log_to_db:           bool       = True,
) -> dict:
    """
    Run retrieval + generation evaluation against the golden dataset.

    Args:
        golden_dataset_path: Path to golden_dataset.json. Defaults to config.
        top_k:               Chunks to retrieve per query. Defaults to config.
        log_to_db:           If True, each query is logged with is_eval=True (#22).

    Returns:
        Full evaluation report dict (also saved to eval_report_path).
    """
    golden_path      = Path(golden_dataset_path or _config["golden_dataset_path"])
    eval_report_path = Path(_config["eval_report_path"])
    top_k            = top_k or _config["top_k"]
    model_used       = _config.get("groq_model", "llama-3.1-8b-instant")
    search_mode      = _config.get("search_mode", "hybrid")
    reranker_top_n   = _config.get("reranker_top_n", 3)

    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {golden_path}")

    with open(golden_path) as f:
        golden_dataset = json.load(f)

    tpm_limit = int(_config.get("groq_tpm_limit", 6000))
    pacer     = _TokenPacer(tpm_limit=tpm_limit)

    logger.info("Starting evaluation — %d queries | top_k=%d | mode=%s | tpm_limit=%d",
                len(golden_dataset), top_k, search_mode, tpm_limit)

    per_query_results = []
    eval_start        = time.time()

    for entry in golden_dataset:
        query_id   = entry["query_id"]
        query      = entry["query"]
        relevant   = entry["relevant_doc_ids"]
        keywords   = entry["expected_answer_keywords"]
        query_type = entry.get("query_type", "unknown")

        logger.info("Evaluating [%s] %s", query_id, query[:60])
        q_start = time.time()

        # ── Retrieve ──────────────────────────────────────────────────────────
        t0      = time.time()
        chunks  = retrieve(query, top_k=top_k, search_mode=search_mode)
        retriever_ms = round((time.time() - t0) * 1000, 1)

        retrieved_sources = [c["source_file"] for c in chunks]
        avg_ret_score     = round(sum(c["score"] for c in chunks) / len(chunks), 4) if chunks else 0.0

        # ── Rerank (apply_threshold=False during eval to always get results) ──
        t0       = time.time()
        reranked = rerank(query, chunks, apply_threshold=False)
        reranker_ms = round((time.time() - t0) * 1000, 1)

        reranked_sources   = [c["source_file"] for c in reranked]
        top_chunk_score    = reranked[0]["reranker_score"]  if reranked else None
        min_reranker_score = reranked[-1]["reranker_score"] if reranked else None

        # ── Generate ──────────────────────────────────────────────────────────
        t0 = time.time()
        if reranked:
            pacer.wait_if_needed()
            llm_result = generate_response(query, reranked)
            pacer.record(llm_result["prompt_tokens"], llm_result["completion_tokens"])
        else:
            llm_result = {
                "answer": "No chunks available after reranking.",
                "sources_cited": [], "model": model_used,
                "chunks_used": 0, "prompt_tokens": 0, "completion_tokens": 0,
            }
        llm_ms = round((time.time() - t0) * 1000, 1)

        answer            = llm_result["answer"]
        sources_cited     = llm_result["sources_cited"]
        prompt_tokens     = llm_result["prompt_tokens"]
        completion_tokens = llm_result["completion_tokens"]
        total_ms          = round((time.time() - q_start) * 1000, 1)

        # ── Compute metrics ───────────────────────────────────────────────────
        p_at_k  = _precision_at_k(reranked_sources, relevant, reranker_top_n)
        r_at_k  = _recall_at_k(reranked_sources,    relevant, reranker_top_n)
        rr      = _reciprocal_rank(reranked_sources, relevant)
        kw_rate = _keyword_hit_rate(answer, keywords)

        result = {
            "query_id":              query_id,
            "query":                 query,
            "query_type":            query_type,
            "relevant_docs":         relevant,
            "retrieved_sources":     retrieved_sources,
            "reranked_sources":      reranked_sources,
            "precision_at_k":        p_at_k,
            "recall_at_k":           r_at_k,
            "reciprocal_rank":       rr,
            "keyword_hit_rate":      kw_rate,
            "keywords_expected":     keywords,
            "keywords_found":        [kw for kw in keywords if kw.lower() in answer.lower()],
            "answer_preview":        answer[:200],
            "sources_cited":         sources_cited,
            "top_chunk_score":       round(top_chunk_score,    4) if top_chunk_score    is not None else None,
            "min_reranker_score":    round(min_reranker_score, 4) if min_reranker_score is not None else None,
            "avg_retriever_score":   avg_ret_score,
            "prompt_tokens":         prompt_tokens,
            "completion_tokens":     completion_tokens,
            "retriever_latency_ms":  retriever_ms,
            "reranker_latency_ms":   reranker_ms,
            "llm_latency_ms":        llm_ms,
            "total_latency_ms":      total_ms,
        }
        per_query_results.append(result)

        # ── Log to DB with is_eval=True (#22) ─────────────────────────────────
        if log_to_db:
            log_query(
                query_text           = query,
                session_id           = "eval_run",
                search_mode          = search_mode,
                chunks_retrieved     = len(chunks),
                chunks_reranked      = len(reranked),
                top_chunk_score      = top_chunk_score,
                avg_retriever_score  = avg_ret_score,
                min_reranker_score   = min_reranker_score,
                sources_cited        = sources_cited,
                citation_count       = len(sources_cited),
                response_text        = answer,
                input_safe           = True,
                output_safe          = True,
                model_used           = model_used,
                prompt_tokens        = prompt_tokens,
                completion_tokens    = completion_tokens,
                latency_ms           = total_ms,
                retriever_latency_ms = retriever_ms,
                reranker_latency_ms  = reranker_ms,
                llm_latency_ms       = llm_ms,
                golden_query_id      = query_id,
                precision_at_k       = p_at_k,
                recall_at_k          = r_at_k,
                is_eval              = True,   # #22 — keeps eval out of production stats
            )

        logger.info(
            "  [%s] P@k=%.2f R@k=%.2f RR=%.2f KW=%.2f | %dms",
            query_id, p_at_k, r_at_k, rr, kw_rate, total_ms,
        )

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    n = len(per_query_results)

    def _mean(key: str) -> float:
        return round(sum(r[key] for r in per_query_results) / n, 4) if n else 0.0

    by_type: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "precision": [], "recall": [], "rr": [], "kw": []
    })
    for r in per_query_results:
        qt = r["query_type"]
        by_type[qt]["count"]     += 1
        by_type[qt]["precision"].append(r["precision_at_k"])
        by_type[qt]["recall"].append(r["recall_at_k"])
        by_type[qt]["rr"].append(r["reciprocal_rank"])
        by_type[qt]["kw"].append(r["keyword_hit_rate"])

    type_summary = {}
    for qt, vals in sorted(by_type.items()):
        c = vals["count"]
        type_summary[qt] = {
            "count":                 c,
            "mean_precision_at_k":   round(sum(vals["precision"]) / c, 4),
            "mean_recall_at_k":      round(sum(vals["recall"])    / c, 4),
            "mrr":                   round(sum(vals["rr"])         / c, 4),
            "mean_keyword_hit_rate": round(sum(vals["kw"])         / c, 4),
        }

    total_eval_s = round(time.time() - eval_start, 1)

    report = {
        "eval_summary": {
            "total_queries":         n,
            "top_k":                 top_k,
            "reranker_top_n":        reranker_top_n,
            "search_mode":           search_mode,
            "model_used":            model_used,
            "mean_precision_at_k":   _mean("precision_at_k"),
            "mean_recall_at_k":      _mean("recall_at_k"),
            "mrr":                   _mean("reciprocal_rank"),
            "mean_keyword_hit_rate": _mean("keyword_hit_rate"),
            "avg_total_latency_ms":  _mean("total_latency_ms"),
            "avg_retriever_ms":      _mean("retriever_latency_ms"),
            "avg_reranker_ms":       _mean("reranker_latency_ms"),
            "avg_llm_ms":            _mean("llm_latency_ms"),
            "avg_prompt_tokens":     _mean("prompt_tokens"),
            "avg_completion_tokens": _mean("completion_tokens"),
            "total_eval_time_s":     total_eval_s,
        },
        "by_query_type": type_summary,
        "per_query":     per_query_results,
    }

    eval_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Evaluation complete | P@k=%.3f R@k=%.3f MRR=%.3f KW=%.3f | saved → %s",
        report["eval_summary"]["mean_precision_at_k"],
        report["eval_summary"]["mean_recall_at_k"],
        report["eval_summary"]["mrr"],
        report["eval_summary"]["mean_keyword_hit_rate"],
        eval_report_path,
    )
    return report


# ── Smoke-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    report = run_evaluation(log_to_db=True)
    s      = report["eval_summary"]

    print(f"\n{'='*65}\n  EVALUATION REPORT\n{'='*65}")
    print(f"  Queries   : {s['total_queries']}  |  mode={s['search_mode']}  |  top_k={s['top_k']}")
    print(f"  Model     : {s['model_used']}\n")
    print(f"  {'Metric':<28} {'Score':>8}")
    print(f"  {'-'*36}")
    print(f"  {'Mean Precision@k':<28} {s['mean_precision_at_k']:>8.4f}")
    print(f"  {'Mean Recall@k':<28} {s['mean_recall_at_k']:>8.4f}")
    print(f"  {'MRR':<28} {s['mrr']:>8.4f}")
    print(f"  {'Mean Keyword Hit Rate':<28} {s['mean_keyword_hit_rate']:>8.4f}")
    print(f"\n  {'Latency':}")
    print(f"  {'Retriever avg':<28} {s['avg_retriever_ms']:>7.1f}ms")
    print(f"  {'Reranker avg':<28} {s['avg_reranker_ms']:>7.1f}ms")
    print(f"  {'LLM avg':<28} {s['avg_llm_ms']:>7.1f}ms")
    print(f"  {'End-to-end avg':<28} {s['avg_total_latency_ms']:>7.1f}ms")
    print(f"\n{'='*65}\n")
