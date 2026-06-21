"""
retriever.py — Step 5
Embeds an incoming query and retrieves the top_k most relevant chunks from ChromaDB.
Supports three search modes: vector, bm25, hybrid (default).
Hybrid mode fuses both rankings via Reciprocal Rank Fusion (RRF).

Fixes applied:
  - #6  config loaded once at module level (not per request)
  - #8  vector and BM25 search run in parallel via ThreadPoolExecutor in hybrid mode
  - #10 fallback to cached BM25 when ChromaDB is unavailable
  - #25 logging.basicConfig removed from module level
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import yaml
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

COLLECTION_NAME = "enterprise_rag"

# ── Module-level config (#6) ───────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

_config: dict = _load_config()

# ── Module-level caches ────────────────────────────────────────────────────────

_model_cache: dict[str, SentenceTransformer] = {}

# BM25 cache: (vectorstore_path, corpus_size) → (BM25Okapi, ids, documents, metadatas)
_bm25_cache: dict[tuple, tuple] = {}

_RRF_K = int(_config.get("rrf_k", 60))


def _get_model(embedding_model: str) -> SentenceTransformer:
    if embedding_model not in _model_cache:
        logger.info("Loading embedding model: %s", embedding_model)
        _model_cache[embedding_model] = SentenceTransformer(embedding_model)
    return _model_cache[embedding_model]


def _get_collection(vectorstore_path: str) -> chromadb.Collection:
    project_root = Path(__file__).resolve().parents[2]
    persist_dir  = (project_root / vectorstore_path).resolve()
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name=COLLECTION_NAME)


def _get_bm25_index(collection: chromadb.Collection, vectorstore_path: str) -> tuple:
    """
    Build (or return cached) a BM25 index over the full ChromaDB corpus.
    Cache invalidates when collection count changes (new ingestion).
    Returns (BM25Okapi, ids, documents, metadatas).
    """
    cache_key = (vectorstore_path, collection.count())
    if cache_key not in _bm25_cache:
        logger.info("Building BM25 index over corpus (%d docs)…", collection.count())
        corpus    = collection.get(include=["documents", "metadatas"])
        ids       = corpus["ids"]
        documents = corpus["documents"]
        metadatas = corpus["metadatas"]
        tokenized = [doc.lower().split() for doc in documents]
        bm25      = BM25Okapi(tokenized)
        _bm25_cache[cache_key] = (bm25, ids, documents, metadatas)
        # Evict older entries for this path to prevent memory leak (#19 partial fix)
        stale = [k for k in _bm25_cache if k[0] == vectorstore_path and k != cache_key]
        for k in stale:
            del _bm25_cache[k]
    return _bm25_cache[cache_key]


def _get_cached_bm25_fallback(vectorstore_path: str) -> tuple | None:
    """Return the most recent cached BM25 for a path, regardless of count. Used when ChromaDB is down."""
    candidates = {k: v for k, v in _bm25_cache.items() if k[0] == vectorstore_path}
    if not candidates:
        return None
    best_key = max(candidates.keys(), key=lambda k: k[1])
    return candidates[best_key]


def _rrf_fuse(
    vector_ranked: list[tuple[str, float]],
    bm25_ranked:   list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Returns (chunk_id, rrf_score) list sorted best-first."""
    rrf_scores: dict[str, float] = {}
    for rank, (chunk_id, _) in enumerate(vector_ranked):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, (chunk_id, _) in enumerate(bm25_ranked):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ── Table-anchored query detection ────────────────────────────────────────────
# Matches queries that explicitly ask for exact values from a table/dashboard/matrix.

_TABLE_QUERY_RE = re.compile(
    r"\b("
    r"according to"
    r"|dashboard"
    r"|matrix"
    r"|schedule"
    r"|what is the (current|target|sla|value|rate|trend)"
    r"|what (are|is) the .{0,30}(sla|kpi|metric|threshold|period|value|rate)"
    r"|how many"
    r"|sla for"
    r")\b",
    re.IGNORECASE,
)


def _is_table_anchored(query: str) -> bool:
    return bool(_TABLE_QUERY_RE.search(query))


def _apply_table_boost(
    ranked: list[tuple[str, float]],
    id_to_doc: dict[str, tuple[str, dict]],
    boost: float,
) -> list[tuple[str, float]]:
    """
    Multiply the RRF score of has_table=True chunks by boost, then re-sort.
    Only applied when the query is detected as table-anchored.
    """
    boosted = [
        (
            cid,
            score * boost
            if cid in id_to_doc and id_to_doc[cid][1].get("has_table")
            else score,
        )
        for cid, score in ranked
    ]
    return sorted(boosted, key=lambda x: x[1], reverse=True)


# ── Maximal Marginal Relevance ─────────────────────────────────────────────────

def _mmr_select(
    ranked: list[tuple[str, float]],
    id_to_doc: dict[str, tuple[str, dict]],
    top_k: int,
    mmr_lambda: float,
) -> list[tuple[str, float]]:
    """
    Select top_k chunks using Maximal Marginal Relevance.

    At each step picks the candidate that maximises:
        mmr_lambda * normalised_relevance - (1 - mmr_lambda) * max_jaccard_with_selected

    mmr_lambda=1.0 → pure relevance (no diversity effect, same as ranked order).
    mmr_lambda=0.7 → 70% relevance / 30% diversity (default).

    Jaccard similarity on token sets is used as the inter-chunk similarity metric —
    fast, dependency-free, and effective at catching near-duplicate chunks from the
    same document section.
    """
    if not ranked:
        return []

    # Normalise scores to [0, 1] so relevance and similarity are on the same scale
    raw_scores = {cid: s for cid, s in ranked}
    max_s = max(raw_scores.values()) or 1.0
    norm  = {cid: s / max_s for cid, s in raw_scores.items()}

    selected: list[tuple[str, float]] = []
    sel_token_sets: list[set[str]]    = []
    remaining: list[str] = [cid for cid, _ in ranked if cid in id_to_doc]

    while len(selected) < top_k and remaining:
        best_cid  = None
        best_mmr  = float("-inf")

        for cid in remaining:
            rel = norm.get(cid, 0.0)
            if not sel_token_sets:
                mmr_score = rel
            else:
                tokens  = set(id_to_doc[cid][0].lower().split())
                union_non_empty = [s for s in sel_token_sets if tokens | s]
                if not union_non_empty:
                    max_sim = 0.0
                else:
                    max_sim = max(
                        len(tokens & s) / len(tokens | s)
                        for s in union_non_empty
                    )
                mmr_score = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_cid = cid

        if best_cid is None:
            break

        selected.append((best_cid, raw_scores[best_cid]))
        sel_token_sets.append(set(id_to_doc[best_cid][0].lower().split()))
        remaining = [c for c in remaining if c != best_cid]

    return selected


# ── Cosine-similarity deduplication ───────────────────────────────────────────

def _fetch_missing_embeddings(
    collection: chromadb.Collection,
    ids: list[str],
    existing: dict[str, list[float]],
) -> dict[str, list[float]]:
    """
    Batch-fetch embeddings from ChromaDB for any IDs not already in existing.
    Used to cover BM25-only candidates whose embeddings weren't returned by
    the vector search query.
    """
    missing = [cid for cid in ids if cid not in existing]
    if not missing:
        return existing
    result = collection.get(ids=missing, include=["embeddings"])
    out = dict(existing)
    for cid, emb in zip(result["ids"], result["embeddings"]):
        out[cid] = emb
    return out


def _dedup_by_cosine(
    ranked: list[tuple[str, float]],
    id_to_emb: dict[str, list[float]],
    threshold: float,
) -> list[tuple[str, float]]:
    """
    Greedy cosine-similarity deduplication (highest score first).

    For each candidate (in score order), compute cosine similarity against
    EVERY item already in the kept list. If similarity > threshold to ANY
    kept item, drop the candidate. Otherwise keep it.

    Since embeddings are unit-normalised at ingestion time, cosine similarity
    equals the dot product — no division needed.

    Candidates with no embedding available are kept unconditionally so that
    BM25-only results are never silently dropped.
    """
    kept: list[tuple[str, float]]    = []
    kept_vecs: list[np.ndarray]      = []

    for cid, score in ranked:
        raw = id_to_emb.get(cid)
        if raw is None:
            kept.append((cid, score))   # no embedding — keep conservatively
            continue

        vec = np.array(raw, dtype=np.float32)

        is_dup = any(float(np.dot(vec, kv)) > threshold for kv in kept_vecs)

        if not is_dup:
            kept.append((cid, score))
            kept_vecs.append(vec)

    return kept


# ── Search sub-tasks (called in parallel for hybrid mode) ──────────────────────

def _do_vector_search(
    collection: chromadb.Collection,
    embedding_model: str,
    query: str,
    fetch_n: int,
) -> tuple[list[tuple[str, float]], dict[str, tuple[str, dict]], dict[str, list[float]]]:
    """Embed query and run ChromaDB cosine similarity search.
    Returns (ranked, id_to_doc, id_to_emb) — embeddings are used downstream
    for cosine-similarity deduplication without a second model call.
    """
    model     = _get_model(embedding_model)
    query_vec = model.encode(query.strip(), normalize_embeddings=True).tolist()
    results   = collection.query(
        query_embeddings=[query_vec],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    ranked: list[tuple[str, float]]      = []
    id_to_doc: dict[str, tuple[str, dict]] = {}
    id_to_emb: dict[str, list[float]]    = {}
    for chunk_id, text, meta, dist, emb in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["embeddings"][0],
    ):
        score = round(1.0 - (dist / 2.0), 4)
        ranked.append((chunk_id, score))
        id_to_doc[chunk_id] = (text, meta)
        id_to_emb[chunk_id] = emb
    return ranked, id_to_doc, id_to_emb


def _do_bm25_search(
    collection: chromadb.Collection,
    vectorstore_path: str,
    query: str,
    fetch_n: int,
) -> tuple[list[tuple[str, float]], dict[str, tuple[str, dict]]]:
    """Score all documents via BM25Okapi and return top fetch_n."""
    bm25, all_ids, all_docs, all_metas = _get_bm25_index(collection, vectorstore_path)
    tokenized_query = query.strip().lower().split()
    bm25_scores     = bm25.get_scores(tokenized_query)
    scored = sorted(
        zip(all_ids, bm25_scores, all_docs, all_metas),
        key=lambda x: x[1],
        reverse=True,
    )
    ranked: list[tuple[str, float]] = []
    id_to_doc: dict[str, tuple[str, dict]] = {}
    for chunk_id, score, text, meta in scored[:fetch_n]:
        ranked.append((chunk_id, round(float(score), 4)))
        id_to_doc[chunk_id] = (text, meta)
    return ranked, id_to_doc


# ── Public API ─────────────────────────────────────────────────────────────────

def warm_up() -> None:
    """
    Pre-load the embedding model so the first real request has no cold-start lag.
    Call this from the FastAPI lifespan startup handler.
    """
    logger.info("Warming up embedding model: %s", _config["embedding_model"])
    _get_model(_config["embedding_model"])


def retrieve(
    query: str,
    top_k: int | None = None,
    embedding_model: str | None = None,
    vectorstore_path: str | None = None,
    search_mode: str | None = None,
) -> list[dict]:
    """
    Retrieve the top_k most relevant chunks for a query.

    Args:
        query:            User query string.
        top_k:            Chunks to return. Defaults to config.
        embedding_model:  Sentence-transformer model name. Defaults to config.
        vectorstore_path: Path to ChromaDB store. Defaults to config.
        search_mode:      "vector" | "bm25" | "hybrid". Defaults to config.

    Returns:
        List of dicts (best-first): {chunk_id, text, source_file, page_number,
                                     section_header, score}
    Raises:
        ValueError: If query is empty or search_mode is invalid.
        RuntimeError: If ChromaDB is unavailable and no BM25 cache exists.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    top_k            = top_k            or _config["top_k"]
    embedding_model  = embedding_model  or _config["embedding_model"]
    vectorstore_path = vectorstore_path or _config["vectorstore_path"]
    search_mode      = search_mode      or _config.get("search_mode", "hybrid")

    if search_mode not in ("vector", "bm25", "hybrid"):
        raise ValueError(f"Invalid search_mode '{search_mode}'. Choose: vector | bm25 | hybrid")

    fetch_n = top_k * 10   # oversample; trim after dedup

    # ── Open ChromaDB — fallback to cached BM25 if unavailable (#10) ──────────
    collection = None
    degraded_bm25 = None
    try:
        collection = _get_collection(vectorstore_path)
        if collection.count() == 0:
            raise RuntimeError("ChromaDB collection is empty. Run the embedder first.")
        fetch_n = min(fetch_n, collection.count())
    except Exception as chroma_err:
        if search_mode == "vector":
            raise RuntimeError(
                f"ChromaDB unavailable and search_mode='vector' requires it: {chroma_err}"
            )
        degraded_bm25 = _get_cached_bm25_fallback(vectorstore_path)
        if degraded_bm25 is None:
            raise RuntimeError(
                f"ChromaDB unavailable and no BM25 cache exists yet: {chroma_err}\n"
                "Run at least one successful query before ChromaDB goes down to build the cache."
            )
        logger.warning(
            "ChromaDB unavailable — degraded to cached BM25-only mode: %s", chroma_err
        )

    # ── Execute searches ───────────────────────────────────────────────────────
    vector_ranked: list[tuple[str, float]] = []
    bm25_ranked:   list[tuple[str, float]] = []
    id_to_doc:     dict[str, tuple[str, dict]] = {}
    vec_embeddings: dict[str, list[float]]    = {}   # populated by vector search

    if degraded_bm25 is not None:
        # Offline fallback: score from cache, no ChromaDB needed
        bm25, all_ids, all_docs, all_metas = degraded_bm25
        tokenized_query = query.strip().lower().split()
        bm25_scores     = bm25.get_scores(tokenized_query)
        scored = sorted(
            zip(all_ids, bm25_scores, all_docs, all_metas),
            key=lambda x: x[1], reverse=True,
        )
        for chunk_id, score, text, meta in scored[:fetch_n]:
            bm25_ranked.append((chunk_id, round(float(score), 4)))
            id_to_doc[chunk_id] = (text, meta)

    elif search_mode == "hybrid":
        # Parallel vector + BM25 (#8)
        with ThreadPoolExecutor(max_workers=2) as executor:
            vec_future  = executor.submit(_do_vector_search, collection, embedding_model, query, fetch_n)
            bm25_future = executor.submit(_do_bm25_search, collection, vectorstore_path, query, fetch_n)

            for future in as_completed([vec_future, bm25_future]):
                if future is vec_future:
                    vector_ranked, docs, vec_embeddings = future.result()
                else:
                    bm25_ranked, docs = future.result()
                id_to_doc.update(docs)

    elif search_mode == "vector":
        vector_ranked, id_to_doc, vec_embeddings = _do_vector_search(
            collection, embedding_model, query, fetch_n
        )

    else:  # bm25
        bm25_ranked, id_to_doc = _do_bm25_search(collection, vectorstore_path, query, fetch_n)

    # ── Rank fusion / selection ────────────────────────────────────────────────
    effective_mode = "bm25" if degraded_bm25 else search_mode
    if effective_mode == "hybrid":
        fused = _rrf_fuse(vector_ranked, bm25_ranked)
    elif effective_mode == "vector":
        fused = vector_ranked
    else:
        fused = bm25_ranked

    # ── Table boost — re-score has_table chunks for table-anchored queries ─────
    table_boost = float(_config.get("table_score_boost", 1.0))
    if table_boost > 1.0 and _is_table_anchored(query):
        fused = _apply_table_boost(fused, id_to_doc, table_boost)
        logger.debug("Table boost (x%.1f) applied | query: '%s'", table_boost, query[:60])

    # ── Cosine deduplication — collapse near-identical candidates ─────────────
    dedup_threshold = float(_config.get("cosine_dedup_threshold", 1.0))
    if dedup_threshold < 1.0 and collection is not None:
        candidate_ids = [cid for cid, _ in fused]
        id_to_emb     = _fetch_missing_embeddings(collection, candidate_ids, vec_embeddings)
        pre_dedup_len = len(fused)
        fused         = _dedup_by_cosine(fused, id_to_emb, dedup_threshold)
        logger.debug(
            "Cosine dedup (threshold=%.2f): %d → %d candidates | query: '%s'",
            dedup_threshold, pre_dedup_len, len(fused), query[:60],
        )

    # ── MMR selection — diverse top_k from the fused candidate pool ───────────
    mmr_lambda = float(_config.get("mmr_lambda", 1.0))
    selected   = _mmr_select(fused, id_to_doc, top_k, mmr_lambda)

    # ── Format (seen_texts guards against identical text under different IDs) ──
    chunks: list[dict] = []
    seen_texts: set[str] = set()

    for chunk_id, score in selected:
        if chunk_id not in id_to_doc:
            continue
        text, meta = id_to_doc[chunk_id]
        text_key   = " ".join(text.split())
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        chunks.append({
            "chunk_id":       chunk_id,
            "text":           text,
            "source_file":    meta.get("source_file", ""),
            "page_number":    meta.get("page_number", -1),
            "section_header": meta.get("section_header", ""),
            "score":          score,
            "degraded":       bool(degraded_bm25),
        })

    logger.info(
        "Retrieved %d chunk(s) [mode=%s%s] for query: '%s'",
        len(chunks),
        effective_mode,
        " (degraded)" if degraded_bm25 else "",
        query[:60] + ("..." if len(query) > 60 else ""),
    )
    return chunks


# ── Smoke-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    test_queries = [
        "What is RBAC and how is it implemented?",
        "How does UEBA detect insider threats?",
        "What CVSS score indicates a critical vulnerability?",
    ]
    for mode in ("vector", "bm25", "hybrid"):
        print(f"\n{'='*60}\nMODE: {mode}\n{'='*60}")
        for q in test_queries:
            print(f"\nQuery: {q}")
            results = retrieve(q, search_mode=mode)
            for i, r in enumerate(results, 1):
                print(f"  [{i}] score={r['score']:.4f} | {r['source_file']} | p{r['page_number']}")
                print(f"       {r['text'][:150]}")
