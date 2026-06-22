"""
canary.py
Canary Query Runner for the Enterprise RAG pipeline.

Runs a fixed set of canary queries through the retrieval + generation pipeline
and compares results against a stored baseline to detect regressions.

Behaviour:
  - First run (no baseline): executes all queries, saves baseline to
    logs/canary_baseline.json
  - Subsequent runs: executes all queries, compares against baseline,
    flags regressions where:
      * top_chunk_score drops more than 1.5 vs baseline
      * keyword_hit_rate drops more than 0.3 vs baseline
      * sources_cited is empty when baseline had citations

Canary runs are logged with is_eval=True so they do not pollute
production KPI metrics.

Run:
    python -m src.evaluation.canary

Output:
    logs/canary_baseline.json  (first run only)
    logs/canary_report.json
"""

import json
import logging
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.retrieval.retriever        import retrieve
from src.retrieval.reranker         import rerank
from src.generation.llm_handler     import generate_response
from src.logging.retrieval_logger   import log_query
from src.evaluation.evaluator       import _TokenPacer

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

_config      = _load_config()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 4)


# ── Core runner ─────────────────────────────────────────────────────────────────

def _run_canary_query(
    query:   str,
    session_id: str,
    top_k:   int,
    mode:    str,
    model:   str,
    pacer:   _TokenPacer,
) -> dict:
    """
    Run a single canary query through retrieve → rerank → generate.
    Returns a result dict with top_chunk_score, sources_cited, answer.
    """
    t_start = time.time()

    chunks   = retrieve(query, top_k=top_k, search_mode=mode)
    reranked = rerank(query, chunks)

    if reranked:
        pacer.wait_if_needed()
        llm_result = generate_response(query, reranked)
        pacer.record(llm_result["prompt_tokens"], llm_result["completion_tokens"])
    else:
        llm_result = {
            "answer": "I could not find relevant information to answer your question.",
            "sources_cited": [],
            "model": model,
            "chunks_used": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    latency_ms        = round((time.time() - t_start) * 1000, 1)
    top_chunk_score   = reranked[0]["reranker_score"] if reranked else None
    avg_ret_score     = round(sum(c["score"] for c in chunks) / len(chunks), 4) if chunks else 0.0
    min_reranker      = reranked[-1]["reranker_score"] if reranked else None

    # Log as eval so it doesn't affect production KPI metrics
    log_query(
        query_text          = query,
        session_id          = session_id,
        input_safe          = True,
        output_safe         = True,
        search_mode         = mode,
        chunks_retrieved    = len(chunks),
        chunks_reranked     = len(reranked),
        top_chunk_score     = top_chunk_score,
        avg_retriever_score = avg_ret_score,
        min_reranker_score  = min_reranker,
        sources_cited       = llm_result["sources_cited"],
        response_text       = llm_result["answer"],
        model_used          = model,
        prompt_tokens       = llm_result["prompt_tokens"],
        completion_tokens   = llm_result["completion_tokens"],
        latency_ms          = latency_ms,
        is_eval             = True,
    )

    return {
        "answer":          llm_result["answer"],
        "answer_preview":  llm_result["answer"][:200],
        "top_chunk_score": round(top_chunk_score, 4) if top_chunk_score is not None else None,
        "sources_cited":   llm_result["sources_cited"],
        "latency_ms":      latency_ms,
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def run_canary() -> dict:
    canary_path   = _PROJECT_ROOT / "data"  / "canary_queries.json"
    baseline_path = _PROJECT_ROOT / "logs"  / "canary_baseline.json"
    report_path   = _PROJECT_ROOT / "logs"  / "canary_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not canary_path.exists():
        raise FileNotFoundError(f"Canary queries not found: {canary_path}")

    canary_queries = json.loads(canary_path.read_text())
    top_k  = _config.get("top_k", 10)
    mode   = _config.get("search_mode", "hybrid")
    model  = _config.get("groq_model", "llama-3.1-8b-instant")
    pacer  = _TokenPacer(tpm_limit=int(_config.get("groq_tpm_limit", 6000)))

    is_first_run = not baseline_path.exists()

    logger.info(
        "Canary runner — %d queries | mode=%s | %s",
        len(canary_queries), mode,
        "FIRST RUN (building baseline)" if is_first_run else "comparing vs baseline",
    )

    baseline = {} if is_first_run else json.loads(baseline_path.read_text())

    results     = []
    regressions = []

    for i, item in enumerate(canary_queries, 1):
        qid   = item["query_id"]
        query = item["query"]
        keywords = item.get("expected_keywords", [])

        logger.info("[%d/%d] %s — %s", i, len(canary_queries), qid, query[:60])
        print(f"  [{i}/{len(canary_queries)}] {qid}: {query[:60]}…", flush=True)

        result = _run_canary_query(query, item.get("session_id", f"canary-{qid}"), top_k, mode, model, pacer)

        kw_rate = _keyword_hit_rate(result["answer"], keywords)
        has_citations = len(result["sources_cited"]) > 0

        entry = {
            "query_id":        qid,
            "query":           query,
            "domain":          item.get("domain", ""),
            "query_type":      item.get("query_type", ""),
            "answer_preview":  result["answer_preview"],
            "top_chunk_score": result["top_chunk_score"],
            "sources_cited":   result["sources_cited"],
            "keyword_hit_rate": kw_rate,
            "latency_ms":      result["latency_ms"],
        }
        results.append(entry)

        # Compare vs baseline on subsequent runs
        if not is_first_run and qid in baseline:
            b = baseline[qid]
            reg_reasons = []

            b_score = b.get("top_chunk_score")
            c_score = result["top_chunk_score"]
            if b_score is not None and c_score is not None:
                if (b_score - c_score) > 1.5:
                    reg_reasons.append(
                        f"top_chunk_score dropped {b_score - c_score:.3f} "
                        f"(baseline={b_score}, current={c_score})"
                    )

            b_kw = b.get("keyword_hit_rate", 0.0)
            if (b_kw - kw_rate) > 0.3:
                reg_reasons.append(
                    f"keyword_hit_rate dropped {b_kw - kw_rate:.3f} "
                    f"(baseline={b_kw:.3f}, current={kw_rate:.3f})"
                )

            b_cited = len(b.get("sources_cited", []))
            if b_cited > 0 and not has_citations:
                reg_reasons.append(
                    f"sources_cited is empty (baseline had {b_cited} citations)"
                )

            if reg_reasons:
                regressions.append({
                    "query_id": qid,
                    "query":    query,
                    "domain":   item.get("domain", ""),
                    "reasons":  reg_reasons,
                })
                logger.warning("REGRESSION [%s]: %s", qid, " | ".join(reg_reasons))

        logger.info(
            "  score=%.4f kw_rate=%.3f cited=%d latency=%.0fms",
            result["top_chunk_score"] or 0, kw_rate,
            len(result["sources_cited"]), result["latency_ms"],
        )

    # Save baseline on first run
    if is_first_run:
        baseline_data = {
            r["query_id"]: {
                "answer_preview":  r["answer_preview"],
                "top_chunk_score": r["top_chunk_score"],
                "sources_cited":   r["sources_cited"],
                "keyword_hit_rate": r["keyword_hit_rate"],
            }
            for r in results
        }
        baseline_path.write_text(json.dumps(baseline_data, indent=2))
        logger.info("Baseline saved → %s", baseline_path)

    report = {
        "run_type":    "baseline" if is_first_run else "comparison",
        "total":       len(results),
        "regressions": len(regressions),
        "results":     results,
        "regression_detail": regressions,
    }
    report_path.write_text(json.dumps(report, indent=2))

    # ── Print summary ─────────────────────────────────────────────────────────
    W = 62
    print(f"\n{'='*W}")
    print(f"  CANARY REPORT — {'BASELINE SAVED' if is_first_run else 'COMPARISON'}")
    print(f"{'='*W}")
    print(f"  Queries run   : {len(results)}")
    if not is_first_run:
        status = "PASS" if not regressions else f"FAIL — {len(regressions)} regression(s)"
        print(f"  Status        : {status}")
        for reg in regressions:
            print(f"\n  REGRESSION [{reg['query_id']}] {reg['domain']}")
            for r in reg["reasons"]:
                print(f"    ✗ {r}")
    else:
        print(f"  Baseline saved → {baseline_path}")
    print(f"  Report saved  → {report_path}")
    print(f"{'='*W}\n")

    return report


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_canary()
