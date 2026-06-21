"""
ragas_eval.py
RAGAS-style Faithfulness evaluation for the Enterprise RAG pipeline.

Faithfulness measures whether every claim in the LLM's answer is
actually supported by the retrieved context chunks.

Algorithm (mirrors RAGAS internals):
  Step 1 — Statement extraction:
    Ask LLM to decompose the answer into atomic, independently
    verifiable statements.
  Step 2 — NLI verification:
    For each statement, ask LLM: "Can this be inferred from the context?"
  Score:
    faithfulness = supported_statements / total_statements
    Range [0.0, 1.0]. 1.0 = fully grounded. 0.0 = fully hallucinated.

Run:
    python -m src.evaluation.ragas_eval

Output:
    logs/ragas_report.json
"""

import json
import logging
import math
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from groq import Groq

from src.retrieval.retriever        import retrieve
from src.retrieval.reranker         import rerank
from src.generation.llm_handler     import generate_response
from src.evaluation.faithfulness    import compute_faithfulness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

_config = _load_config()


# ── Token-aware rate limiter (same logic as evaluator.py) ─────────────────────

class _TokenPacer:
    _WINDOW = 60.0

    def __init__(self, tpm_limit: int):
        self._tpm_limit        = tpm_limit
        self._window_start     = time.monotonic()
        self._tokens_in_window = 0
        self._total_tokens     = 0
        self._queries_done     = 0

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        used = prompt_tokens + completion_tokens
        self._tokens_in_window += used
        self._total_tokens     += used
        self._queries_done     += 1

    def wait_if_needed(self) -> None:
        elapsed = time.monotonic() - self._window_start
        if elapsed >= self._WINDOW:
            self._window_start     = time.monotonic()
            self._tokens_in_window = 0
            return
        estimated_next = (
            math.ceil(self._total_tokens / self._queries_done)
            if self._queries_done > 0 else 800
        )
        if self._tokens_in_window + estimated_next > self._tpm_limit:
            sleep_for = self._WINDOW - elapsed
            logger.info("Rate limiter: sleeping %.1fs to reset TPM window.", sleep_for)
            time.sleep(sleep_for)
            self._window_start     = time.monotonic()
            self._tokens_in_window = 0


# ── Per-query faithfulness (wraps faithfulness.py with pacer accounting) ──────

def _faithfulness_for_query(
    answer:   str,
    contexts: list[str],
    client:   Groq,
    model:    str,
    pacer:    _TokenPacer,
) -> dict:
    """
    Wraps compute_faithfulness() with token-pacer accounting for batch eval.
    The pacer records tokens consumed so the eval stays within Groq TPM limits.
    """
    # Estimate token cost before calling (pacer gates on this)
    pacer.wait_if_needed()
    result = compute_faithfulness(answer, contexts, client, model)
    # Rough accounting: statement extraction + N verifications ≈ N*150 tokens
    # We record a conservative estimate since we don't have exact usage here
    n = result["total_statements"]
    pacer.record(prompt_tokens=300 + n * 100, completion_tokens=n * 5 + 20)
    return result


# ── Main evaluation loop ───────────────────────────────────────────────────────

def run_ragas_eval() -> None:
    dataset_path = Path(_config["golden_dataset_path"])
    report_path  = Path(_config.get("ragas_report_path", "logs/ragas_report.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    golden = json.loads(dataset_path.read_text())
    model  = _config.get("groq_model", "llama-3.1-8b-instant")
    top_k  = _config.get("top_k", 10)
    mode   = _config.get("search_mode", "hybrid")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")
    client = Groq(api_key=api_key)
    pacer  = _TokenPacer(tpm_limit=int(_config.get("groq_tpm_limit", 6000)))

    logger.info(
        "Starting RAGAS Faithfulness eval — %d queries | model=%s | mode=%s",
        len(golden), model, mode,
    )

    results = []
    scores  = []   # only non-None scores

    for i, item in enumerate(golden, 1):
        qid   = item["query_id"]
        query = item["query"]
        logger.info("[%d/%d] %s — %s", i, len(golden), qid, query[:70])

        # ── Retrieve + rerank ──────────────────────────────────────────────────
        chunks   = retrieve(query, top_k=top_k, search_mode=mode)
        reranked = rerank(query, chunks)

        if not reranked:
            logger.info("  No reranked chunks — skipping LLM calls.")
            results.append({
                "query_id": qid, "query": query,
                "answer": "", "contexts": [],
                "total_statements": 0, "supported_statements": 0,
                "faithfulness": None, "statements": [],
            })
            continue

        # ── Generate answer ────────────────────────────────────────────────────
        pacer.wait_if_needed()
        llm_result = generate_response(query, reranked)
        pacer.record(llm_result["prompt_tokens"], llm_result["completion_tokens"])
        answer   = llm_result["answer"]
        contexts = [c["text"] for c in reranked]

        logger.info(
            "  Answer (%d chars) | %d context chunks",
            len(answer), len(contexts),
        )

        # ── Faithfulness ──────────────────────────────────────────────────────
        faith = _faithfulness_for_query(answer, contexts, client, model, pacer)
        score = faith["faithfulness"]

        if score is not None:
            scores.append(score)
            logger.info(
                "  Faithfulness: %.4f (%d/%d statements supported)",
                score, faith["supported_statements"], faith["total_statements"],
            )
        else:
            logger.info("  Faithfulness: N/A (no verifiable statements in answer)")

        results.append({
            "query_id":             qid,
            "query":                query,
            "answer":               answer,
            "contexts":             contexts,
            **faith,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    mean_faith = round(sum(scores) / len(scores), 4) if scores else None

    fully_faithful   = sum(1 for s in scores if s == 1.0)
    partial          = sum(1 for s in scores if 0.0 < s < 1.0)
    unfaithful       = sum(1 for s in scores if s == 0.0)
    no_answer        = sum(1 for r in results if r["faithfulness"] is None)

    summary = {
        "queries":             len(golden),
        "evaluated":           len(scores),
        "mean_faithfulness":   mean_faith,
        "fully_faithful":      fully_faithful,
        "partially_faithful":  partial,
        "unfaithful":          unfaithful,
        "no_answer_or_empty":  no_answer,
        "model":               model,
        "search_mode":         mode,
    }

    report = {"summary": summary, "results": results}
    report_path.write_text(json.dumps(report, indent=2))

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RAGAS — FAITHFULNESS REPORT")
    print(f"{'='*60}")
    print(f"  Queries evaluated : {len(scores)} / {len(golden)}")
    print(f"  Mean Faithfulness : {mean_faith:.4f}" if mean_faith else "  Mean Faithfulness : N/A")
    print(f"  Fully faithful    : {fully_faithful}")
    print(f"  Partially faithful: {partial}")
    print(f"  Unfaithful (0.0)  : {unfaithful}")
    print(f"  No answer / N/A   : {no_answer}")
    print(f"\n  Per-query breakdown:")
    print(f"  {'ID':<8} {'Score':>8}  {'Sup/Tot':>8}  Query")
    print(f"  {'-'*70}")
    for r in results:
        sc  = f"{r['faithfulness']:.4f}" if r["faithfulness"] is not None else "  N/A "
        st  = f"{r['supported_statements']}/{r['total_statements']}" if r["total_statements"] else "  —  "
        print(f"  {r['query_id']:<8} {sc:>8}  {st:>8}  {r['query'][:45]}")
    print(f"\n  Report saved → {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_ragas_eval()
