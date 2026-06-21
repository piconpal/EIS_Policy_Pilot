"""
test_cache.py
Tests for the in-memory query cache introduced in src/pipeline.py.

Test groups:
  A. _QueryCache unit tests  — no mocking, pure Python
       A1  Cache miss returns None
       A2  Cache hit returns stored value
       A3  Key normalisation (case + whitespace)
       A4  Different search_mode → different key
       A5  TTL expiry
       A6  LRU eviction at maxsize
       A7  Thread safety — concurrent reads/writes

  B. Pipeline integration tests  — mock expensive IO, test cache routing logic
       B1  Turn 1 miss: route_and_retrieve called, result stored
       B2  Turn 1 hit:  route_and_retrieve NOT called, cached result used
       B3  Turn 2+:     cache never consulted even if identical query
       B4  cache_hit flag logged correctly via log_query

  C. Metric-preservation test  — cached chunks/reranked are byte-identical to source
       C1  Stored and retrieved retrieval payload are identical

Usage:
    python test_cache.py
"""

import sys
import time
import threading
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import _QueryCache, query_cache

DIVIDER  = "─" * 68
BOLD_DIV = "═" * 68


def _section(title: str) -> None:
    print(f"\n{BOLD_DIV}")
    print(f"  {title}")
    print(BOLD_DIV)


def _ok(label: str) -> None:
    print(f"  [PASS] {label}")


def _fail(label: str, detail: str = "") -> None:
    print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))


# ── A. _QueryCache unit tests ──────────────────────────────────────────────────

def test_a1_miss_returns_none() -> bool:
    cache = _QueryCache(maxsize=10, ttl=60)
    result = cache.get("What is RBAC?", "hybrid")
    ok = result is None
    (print("  [PASS] A1: cache miss returns None") if ok
     else _fail("A1: cache miss should return None", f"got {result!r}"))
    return ok


def test_a2_hit_returns_value() -> bool:
    cache   = _QueryCache(maxsize=10, ttl=60)
    payload = (["chunk1", "chunk2"], "factual", "What is RBAC?", ["r1", "r2"])
    cache.set("What is RBAC?", "hybrid", payload)
    result = cache.get("What is RBAC?", "hybrid")
    ok = result == payload
    (print("  [PASS] A2: cache hit returns stored value") if ok
     else _fail("A2: hit should return stored value", f"got {result!r}"))
    return ok


def test_a3_key_normalisation() -> bool:
    cache   = _QueryCache(maxsize=10, ttl=60)
    payload = (["chunk"], "factual", "what is rbac?", [])

    # Store with lowercase, stripped form
    cache.set("what is rbac?", "hybrid", payload)

    # Retrieve with different casing and surrounding whitespace
    result = cache.get("  What Is RBAC?  ", "hybrid")
    ok = result == payload
    (print("  [PASS] A3: key normalised (case + whitespace)") if ok
     else _fail("A3: normalisation failed", f"got {result!r}"))
    return ok


def test_a4_different_mode_different_key() -> bool:
    cache    = _QueryCache(maxsize=10, ttl=60)
    payload1 = (["chunk_hybrid"], "factual", "q", [])
    payload2 = (["chunk_vector"], "factual", "q", [])

    cache.set("What is RBAC?", "hybrid", payload1)
    cache.set("What is RBAC?", "vector", payload2)

    r_hybrid = cache.get("What is RBAC?", "hybrid")
    r_vector = cache.get("What is RBAC?", "vector")
    ok = r_hybrid == payload1 and r_vector == payload2 and r_hybrid != r_vector
    (print("  [PASS] A4: different search_mode → different cache key") if ok
     else _fail("A4: mode isolation failed"))
    return ok


def test_a5_ttl_expiry() -> bool:
    cache   = _QueryCache(maxsize=10, ttl=0.1)   # 100 ms TTL
    payload = (["chunk"], "factual", "q", [])
    cache.set("What is RBAC?", "hybrid", payload)

    # Should hit immediately
    assert cache.get("What is RBAC?", "hybrid") == payload, "pre-expiry get failed"

    time.sleep(0.15)   # wait past TTL

    result = cache.get("What is RBAC?", "hybrid")
    ok = result is None
    (print("  [PASS] A5: TTL expiry — stale entry evicted") if ok
     else _fail("A5: TTL expiry", f"got {result!r} after TTL"))
    return ok


def test_a6_lru_eviction() -> bool:
    cache = _QueryCache(maxsize=3, ttl=60)
    for i in range(4):
        cache.set(f"query {i}", "hybrid", (f"val{i}",))

    # query 0 should have been evicted (oldest / least recently used)
    evicted = cache.get("query 0", "hybrid")
    kept    = [cache.get(f"query {i}", "hybrid") for i in range(1, 4)]
    ok = evicted is None and all(v is not None for v in kept)
    (print("  [PASS] A6: LRU eviction — oldest entry dropped at maxsize") if ok
     else _fail("A6: LRU eviction", f"evicted={evicted!r} kept={kept!r}"))
    return ok


def test_a7_thread_safety() -> bool:
    cache   = _QueryCache(maxsize=50, ttl=60)
    errors  = []

    def worker(thread_id: int) -> None:
        try:
            for i in range(20):
                cache.set(f"query {thread_id}-{i}", "hybrid", (thread_id, i))
                cache.get(f"query {thread_id}-{i}", "hybrid")
        except Exception as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = len(errors) == 0 and cache.size() <= 50
    (print(f"  [PASS] A7: thread safety — {cache.size()} entries, 0 errors") if ok
     else _fail("A7: thread safety", f"errors={errors}, size={cache.size()}"))
    return ok


# ── B. Pipeline integration tests (with mocking) ──────────────────────────────

_FAKE_CHUNKS   = [{"chunk_id": "c1", "text": "RBAC assigns roles", "score": 0.9,
                   "source_file": "iam.pdf", "page_number": 1, "section_header": "IAM"}]
_FAKE_RERANKED = [{"chunk_id": "c1", "text": "RBAC assigns roles", "score": 0.9,
                   "reranker_score": 0.95, "source_file": "iam.pdf",
                   "page_number": 1, "section_header": "IAM"}]
_FAKE_LLM      = {
    "answer":            "RBAC stands for Role-Based Access Control.",
    "sources_cited":     [{"source_file": "iam.pdf", "page_number": 1,
                           "section_header": "IAM"}],
    "model":             "llama-3.1-8b-instant",
    "chunks_used":       1,
    "prompt_tokens":     100,
    "completion_tokens": 50,
}
_FAKE_GUARD_OK = {
    "is_safe":            True,
    "reason":             "OK",
    "sanitized_query":    "What is RBAC?",
    "rate_limit_status":  {"requests_remaining": 99},
}
_FAKE_OUT_GUARD = {"is_safe": True, "reason": "OK", "filtered_response": _FAKE_LLM["answer"]}


def _pipeline_mocks(session_context=None):
    """Return a dict of patches that stub all IO in run_pipeline."""
    if session_context is None:
        session_context = []
    return {
        "src.pipeline.check_input":        mock.MagicMock(return_value=_FAKE_GUARD_OK),
        "src.pipeline.check_output":       mock.MagicMock(return_value=_FAKE_OUT_GUARD),
        "src.pipeline.get_context":        mock.MagicMock(return_value=session_context),
        "src.pipeline.add_turn":           mock.MagicMock(),
        "src.pipeline.route_and_retrieve": mock.MagicMock(
            return_value=(_FAKE_CHUNKS, "factual", "What is RBAC?")
        ),
        "src.pipeline.rerank":             mock.MagicMock(return_value=_FAKE_RERANKED),
        "src.pipeline.generate_response":  mock.MagicMock(return_value=_FAKE_LLM),
        "src.pipeline.log_query":          mock.MagicMock(return_value="mock-qid-001"),
    }


def _apply_patches(patches: dict):
    active = {}
    for target, mock_obj in patches.items():
        p = mock.patch(target, mock_obj)
        p.start()
        active[target] = p
    return active


def _stop_patches(active: dict) -> None:
    for p in active.values():
        p.stop()


def test_b1_turn1_miss_calls_retriever() -> bool:
    """Turn 1, cache empty → route_and_retrieve must be called and result stored."""
    from src.pipeline import run_pipeline, query_cache
    query_cache.clear()

    patches = _pipeline_mocks(session_context=[])   # empty = turn 1
    active  = _apply_patches(patches)
    try:
        run_pipeline("What is RBAC?", "sess-b1", "user1", "hybrid", {"groq_model": "test"})
        called = patches["src.pipeline.route_and_retrieve"].call_count == 1
        stored = query_cache.get("What is RBAC?", "hybrid") is not None
        ok = called and stored
    finally:
        _stop_patches(active)

    (print("  [PASS] B1: turn 1 miss — route_and_retrieve called, result cached") if ok
     else _fail("B1", f"called={called}, stored={stored}"))
    return ok


def test_b2_turn1_hit_skips_retriever() -> bool:
    """Turn 1, cache warm → route_and_retrieve must NOT be called."""
    from src.pipeline import run_pipeline, query_cache
    query_cache.clear()

    # Pre-warm the cache
    query_cache.set("What is RBAC?", "hybrid",
                    (_FAKE_CHUNKS, "factual", "What is RBAC?", _FAKE_RERANKED))

    patches = _pipeline_mocks(session_context=[])
    active  = _apply_patches(patches)
    try:
        run_pipeline("What is RBAC?", "sess-b2", "user1", "hybrid", {"groq_model": "test"})
        not_called = patches["src.pipeline.route_and_retrieve"].call_count == 0
        rerank_not_called = patches["src.pipeline.rerank"].call_count == 0
        ok = not_called and rerank_not_called
    finally:
        _stop_patches(active)

    (print("  [PASS] B2: turn 1 hit  — route_and_retrieve + rerank skipped") if ok
     else _fail("B2: retriever/reranker should be skipped on cache hit"))
    return ok


def test_b3_turn2_always_retrieves() -> bool:
    """Turn 2+ must bypass cache entirely even if the query is identical."""
    from src.pipeline import run_pipeline, query_cache
    query_cache.clear()

    # Pre-warm cache for the same query
    query_cache.set("What is RBAC?", "hybrid",
                    (_FAKE_CHUNKS, "factual", "What is RBAC?", _FAKE_RERANKED))

    # session_context non-empty → turn 2
    prior_turn = [{"role": "user", "content": "What is RBAC?", "tokens": 10}]
    patches    = _pipeline_mocks(session_context=prior_turn)
    active     = _apply_patches(patches)
    try:
        run_pipeline("What is RBAC?", "sess-b3", "user1", "hybrid", {"groq_model": "test"})
        called = patches["src.pipeline.route_and_retrieve"].call_count == 1
        ok     = called
    finally:
        _stop_patches(active)

    (print("  [PASS] B3: turn 2+ — cache bypassed, fresh retrieval always runs") if ok
     else _fail("B3: turn 2+ should always call route_and_retrieve"))
    return ok


def test_b4_cache_hit_logged() -> bool:
    """cache_hit=True must be passed to log_query on a cache hit."""
    from src.pipeline import run_pipeline, query_cache
    query_cache.clear()

    query_cache.set("What is RBAC?", "hybrid",
                    (_FAKE_CHUNKS, "factual", "What is RBAC?", _FAKE_RERANKED))

    patches = _pipeline_mocks(session_context=[])
    active  = _apply_patches(patches)
    try:
        run_pipeline("What is RBAC?", "sess-b4", "user1", "hybrid", {"groq_model": "test"})
        call_kwargs = patches["src.pipeline.log_query"].call_args
        logged_hit  = call_kwargs.kwargs.get("cache_hit", None)
        ok = logged_hit is True
    finally:
        _stop_patches(active)

    (print("  [PASS] B4: cache_hit=True logged to log_query on hit") if ok
     else _fail("B4: expected cache_hit=True in log_query call", f"got {logged_hit!r}"))
    return ok


def test_b5_cache_miss_logged() -> bool:
    """cache_hit=False must be passed to log_query on a cache miss."""
    from src.pipeline import run_pipeline, query_cache
    query_cache.clear()

    patches = _pipeline_mocks(session_context=[])
    active  = _apply_patches(patches)
    try:
        run_pipeline("What is RBAC miss?", "sess-b5", "user1", "hybrid", {"groq_model": "test"})
        call_kwargs = patches["src.pipeline.log_query"].call_args
        logged_hit  = call_kwargs.kwargs.get("cache_hit", None)
        ok = logged_hit is False
    finally:
        _stop_patches(active)

    (print("  [PASS] B5: cache_hit=False logged to log_query on miss") if ok
     else _fail("B5: expected cache_hit=False in log_query call", f"got {logged_hit!r}"))
    return ok


# ── C. Metric-preservation test ───────────────────────────────────────────────

def test_c1_cached_payload_is_identical() -> bool:
    """
    Retrieval payload stored in cache must be byte-identical to what was originally
    computed. Verifies the cache does not transform, truncate, or reorder results.
    """
    cache = _QueryCache(maxsize=10, ttl=60)

    # Simulate realistic retrieval output
    chunks = [
        {"chunk_id": f"c{i}", "text": f"chunk text {i}", "score": round(0.9 - i * 0.05, 4),
         "source_file": "iam.pdf", "page_number": i + 1, "section_header": "Section"}
        for i in range(5)
    ]
    reranked = [
        {**c, "reranker_score": round(0.95 - idx * 0.05, 4)}
        for idx, c in enumerate(chunks[:3])
    ]
    payload = (chunks, "factual", "What is RBAC?", reranked)

    cache.set("What is RBAC?", "hybrid", payload)
    retrieved = cache.get("What is RBAC?", "hybrid")

    ok = (
        retrieved is not None
        and retrieved[0] == chunks        # same chunks
        and retrieved[1] == "factual"     # same query_type
        and retrieved[2] == "What is RBAC?"   # same retrieval_query
        and retrieved[3] == reranked      # same reranked list
        and retrieved[3][0]["reranker_score"] == reranked[0]["reranker_score"]
    )
    (print("  [PASS] C1: cached payload identical to source — no data corruption") if ok
     else _fail("C1: payload mismatch", f"retrieved={retrieved!r}"))
    return ok


# ── Runner ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BOLD_DIV}")
    print("  QUERY CACHE — TEST SUITE")
    print(f"{BOLD_DIV}")

    results: dict[str, bool] = {}

    _section("A. _QueryCache unit tests")
    results["A1 miss returns None"]             = test_a1_miss_returns_none()
    results["A2 hit returns value"]             = test_a2_hit_returns_value()
    results["A3 key normalisation"]             = test_a3_key_normalisation()
    results["A4 different mode → diff key"]     = test_a4_different_mode_different_key()
    results["A5 TTL expiry"]                    = test_a5_ttl_expiry()
    results["A6 LRU eviction at maxsize"]       = test_a6_lru_eviction()
    results["A7 thread safety"]                 = test_a7_thread_safety()

    _section("B. Pipeline integration tests (mocked IO)")
    results["B1 turn1 miss calls retriever"]    = test_b1_turn1_miss_calls_retriever()
    results["B2 turn1 hit skips retriever"]     = test_b2_turn1_hit_skips_retriever()
    results["B3 turn2+ always retrieves"]       = test_b3_turn2_always_retrieves()
    results["B4 cache_hit=True logged on hit"]  = test_b4_cache_hit_logged()
    results["B5 cache_hit=False logged on miss"]= test_b5_cache_miss_logged()

    _section("C. Metric-preservation test")
    results["C1 cached payload identical"]      = test_c1_cached_payload_is_identical()

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(results.values())
    total  = len(results)
    print(f"\n{BOLD_DIV}")
    print(f"  SUMMARY: {passed}/{total} tests passed")
    if passed < total:
        print(f"\n  Failed:")
        for name, ok in results.items():
            if not ok:
                print(f"    - {name}")
    print(f"{BOLD_DIV}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
