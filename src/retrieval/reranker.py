"""
reranker.py — Step 6
Reranks retrieved chunks using a cross-encoder model
(cross-encoder/ms-marco-MiniLM-L-6-v2).

Fixes applied:
  - #6  config loaded once at module level (not per request)
  - #9  warm_up() pre-loads the cross-encoder to avoid cold-start on first request
  - #18 reranker_confidence_threshold from config is now enforced — returns []
        when the top chunk score is below threshold (caller should not invoke LLM)
  - #25 logging.basicConfig removed from module level
"""

import logging
from pathlib import Path

import yaml
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder_cache: CrossEncoder | None = None

# ── Module-level config (#6) ───────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

_config: dict = _load_config()


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder_cache
    if _cross_encoder_cache is None:
        logger.info("Loading cross-encoder model: %s", _CROSS_ENCODER_MODEL)
        _cross_encoder_cache = CrossEncoder(_CROSS_ENCODER_MODEL, max_length=512)
    return _cross_encoder_cache


# ── Public API ─────────────────────────────────────────────────────────────────

def warm_up() -> None:
    """
    Pre-load the cross-encoder model so the first real request has no cold-start lag.
    Call this from the FastAPI lifespan startup handler.
    """
    logger.info("Warming up cross-encoder model: %s", _CROSS_ENCODER_MODEL)
    _get_cross_encoder()


def rerank(
    query: str,
    chunks: list[dict],
    top_n: int | None = None,
    apply_threshold: bool = True,
) -> list[dict]:
    """
    Rerank chunks using a cross-encoder and return the top_n most relevant.

    Args:
        query:            The user query string.
        chunks:           Output of retriever.retrieve().
        top_n:            Chunks to keep after reranking. Defaults to config.
        apply_threshold:  If True (default), return [] when the top chunk score
                          is below reranker_confidence_threshold in config.
                          Pass False during evaluation to always get results.

    Returns:
        Top top_n chunks sorted by reranker_score descending, each extended with:
        {
            "retriever_score":  float — original score from retriever,
            "reranker_score":   float — cross-encoder relevance score (raw logit),
        }
        Returns [] if confidence threshold is not met and apply_threshold=True.

    Raises:
        ValueError: If query is empty or chunks is empty.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if not chunks:
        raise ValueError("No chunks provided to rerank.")

    top_n = top_n if top_n is not None else _config["reranker_top_n"]

    cross_encoder = _get_cross_encoder()
    pairs         = [(query.strip(), chunk["text"]) for chunk in chunks]
    scores        = cross_encoder.predict(pairs, show_progress_bar=False)

    scored_chunks = []
    for chunk, ce_score in zip(chunks, scores):
        enriched                   = dict(chunk)
        enriched["retriever_score"] = chunk.get("score", 0.0)
        enriched["reranker_score"]  = float(ce_score)
        scored_chunks.append(enriched)

    scored_chunks.sort(key=lambda c: c["reranker_score"], reverse=True)

    # ── Per-chunk floor — drop truly irrelevant chunks ─────────────────────────
    # Uses reranker_min_chunk_score (default 0.0) as a floor so borderline-relevant
    # chunks (e.g. 0.98) reach the LLM while clearly irrelevant ones (e.g. -7.0) don't.
    if apply_threshold:
        min_score = _config.get("reranker_min_chunk_score", 0.0)
        scored_chunks = [c for c in scored_chunks if c["reranker_score"] >= min_score]

    top_chunks = scored_chunks[:top_n]

    # ── Confidence gate — skip LLM if best chunk is still weak (#18) ──────────
    if apply_threshold and top_chunks:
        threshold = _config.get("reranker_confidence_threshold")
        if threshold is not None and top_chunks[0]["reranker_score"] < threshold:
            logger.warning(
                "Top reranker score %.4f is below confidence threshold %.4f — "
                "returning empty to suppress low-quality LLM response.",
                top_chunks[0]["reranker_score"], threshold,
            )
            return []

    logger.info(
        "Reranked %d chunks → kept top %d | best score: %.4f",
        len(chunks), len(top_chunks),
        top_chunks[0]["reranker_score"] if top_chunks else float("nan"),
    )
    return top_chunks


# ── Smoke-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.retrieval.retriever import retrieve

    warm_up()

    test_cases = [
        "What are the key principles of Role Based Access Control?",
        "How does UEBA detect insider threats?",
        "What CVSS score indicates a critical vulnerability?",
    ]
    for query in test_cases:
        print(f"\n{'='*65}\nQuery: {query}\n{'='*65}")
        candidates = retrieve(query)
        print(f"\nRetriever top {len(candidates)} candidates:")
        for i, c in enumerate(candidates, 1):
            print(f"  [{i}] retriever_score={c['score']:.4f} | {c['source_file']} p{c['page_number']}")
            print(f"       {c['text'][:100]}")

        reranked = rerank(query, candidates, apply_threshold=False)
        print(f"\nReranker top {len(reranked)} (after cross-encoder):")
        for i, c in enumerate(reranked, 1):
            print(
                f"  [{i}] reranker={c['reranker_score']:.4f} "
                f"(was retriever={c['retriever_score']:.4f}) | "
                f"{c['source_file']} p{c['page_number']}"
            )
            print(f"       {c['text'][:150]}")
