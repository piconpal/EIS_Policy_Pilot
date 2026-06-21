"""
query_router.py
Classifies an incoming query and routes it through the appropriate retrieval strategy.

Normalization (runs first on every query):
  Step 1 — Conversational prefix stripping (regex, free):
    Removes filler phrases ("can you tell me", "show me", "explain me", etc.)
    that add embedding noise without carrying information.
  Step 2 — Grammatical normalization (regex, free):
    Fixes informal contractions ("whats"→"what is", "is we"→"is the organization").
    Also prevents "or not" from falsely triggering comparison routing.
  Step 3 — Multi-part query decomposition (regex detect + LLM decompose):
    Detects two independent question clauses joined by "and".
    Decomposes into sub-queries, retrieves + reranks each separately,
    merges results. Returns query_type="multi_part". Pipeline skips rerank.
  Step 4 — Informal rewrite (LLM, only when regex classification fails):
    Triggered when no regex pattern matches (query is non-standard/indirect).
    Reformulates the query as a direct, formal question before retrieval.

Classification (two-stage, runs after normalization):
  Stage 1 — Regex (0ms, free):
    Detects explicit comparison patterns: "difference between", "X vs Y", etc.
    If matched → query_type = "comparison", skip Stage 2.
  Stage 2 — LLM fallback (only if regex misses):
    Sends a minimal classification prompt to Groq.
    Returns: factual | comparison | metric | process

Routing:
  multi_part → decompose → retrieve each → rerank each → merge (rerank done inside)
  comparison → decompose → retrieve each → merge → rerank (rerank done in pipeline)
  all others → single retrieve() call as normal

Public API:
  classify_query(query)           → str
  route_and_retrieve(query, ...)  → (chunks, query_type, retrieval_query)
"""

import os
import re
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv
from groq import Groq

from src.retrieval.retriever import retrieve

# ── Logger ─────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Stage 1: Regex classifier ──────────────────────────────────────────────────

# Factual patterns checked first — highest priority to avoid "how does" false positives
# on queries that are primarily definitional ("What is X and how does it work?")
_FACTUAL_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^\s*what\s+is\b",
        r"^\s*what\s+are\b",
        r"^\s*define\b",
        r"^\s*explain\s+what\b",
        r"^\s*describe\s+what\b",
    ]
]

_COMPARISON_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bdifference\s+between\b",
        r"\bcompare\b",
        r"\bversus\b",
        r"\bvs\.?\s",
        r"\bhow\s+does\s+.+\s+differ\b",
        r"\bwhat\s+distinguishes\b",
        r"\bcontrast\b",
        r"\bsimilarities\s+and\s+differences\b",
    ]
]

_METRIC_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bkpi\b",
        r"\bmetric\b",
        r"\bmeasure\b",
        r"\bthreshold\b",
        r"\bscore\b",
        r"\bbenchmark\b",
        r"\bindicator\b",
    ]
]

_PROCESS_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bhow\s+does\b",
        r"\bhow\s+do\b",
        r"\bstep[s]?\b",
        r"\bprocess\b",
        r"\bworkflow\b",
        r"\bprocedure\b",
    ]
]


# ── Step 1+2: Query normalization ─────────────────────────────────────────────

# Conversational prefixes that add embedding noise without information
_PREFIX_STRIP = re.compile(
    r"^\s*(?:"
    r"can you (?:tell me|explain|show me|describe|list|give me)|"
    r"(?:please\s+)?(?:tell me|show me|explain(?:\s+me)?|describe|list|give me)|"
    r"i (?:want|need|would like) (?:to know|to understand|information about)|"
    r"(?:could you|would you|help me)\s+(?:explain|describe|tell me about|show me)"
    r")\s+",
    re.IGNORECASE,
)

# Grammar fixes: (pattern, replacement)
_GRAMMAR_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bwhats\b",       re.IGNORECASE), "what is"),
    (re.compile(r"\bhows\b",        re.IGNORECASE), "how is"),
    (re.compile(r"\bwhos\b",        re.IGNORECASE), "who is"),
    (re.compile(r"\bis\s+we\b",     re.IGNORECASE), "is the organization"),
    (re.compile(r"\bor\s+not\b",    re.IGNORECASE), ""),       # prevent false comparison hit
]


def _normalize_query(query: str) -> str:
    """Strip conversational prefixes (Step 1) and fix grammar (Step 2)."""
    q = _PREFIX_STRIP.sub("", query).strip()
    for pattern, replacement in _GRAMMAR_FIXES:
        q = pattern.sub(replacement, q)
    return q.strip()


# ── Step 3: Multi-part query detection + decomposition ────────────────────────

# Matches: "<clause of 3+ words> and <what/how/which/who/when + word>"
_MULTIPART_RE = re.compile(
    r"(?:\w+\s+){3,}and\s+(?:what(?:'?s)?\s+\w+|how\s+\w+|which\s+\w+|who\s+\w+|when\s+\w+)",
    re.IGNORECASE,
)


def _is_multipart(query: str) -> bool:
    """Return True if query contains two independent question clauses joined by 'and'."""
    return bool(_MULTIPART_RE.search(query))


_MULTIPART_DECOMPOSE_PROMPT = """A user asked one question that actually contains two independent factual questions joined by "and". Break it into exactly 2 standalone questions — one for each independent sub-question.

Original: "{query}"

Respond with exactly 2 lines. Line 1 is sub-question 1. Line 2 is sub-question 2.
No numbering, no bullets, no extra text — just the two questions:"""


def _decompose_multipart(query: str, groq_model: str) -> tuple[str, str]:
    """LLM-decompose a multi-part query into two independent factual sub-questions."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return query, query

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": _MULTIPART_DECOMPOSE_PROMPT.format(query=query)}],
            temperature=0.0,
            max_tokens=80,
        )
        lines = [
            l.strip() for l in
            response.choices[0].message.content.strip().splitlines()
            if l.strip()
        ]
        if len(lines) >= 2:
            logger.info("Multi-part decomposed:\n  [1] %s\n  [2] %s", lines[0], lines[1])
            return lines[0], lines[1]
    except Exception as e:
        logger.warning("Multi-part decomposition failed (%s) — using original.", e)

    return query, query


def _handle_multipart(
    query: str,
    sub_q1: str,
    sub_q2: str,
    retrieval_kwargs: dict,
    config: dict,
) -> list[dict]:
    """Retrieve and rerank each sub-query independently, then merge results."""
    from src.retrieval.reranker import rerank as _rerank_fn   # local import avoids circular

    chunks_1   = retrieve(sub_q1, **retrieval_kwargs)
    reranked_1 = _rerank_fn(sub_q1, chunks_1)

    chunks_2   = retrieve(sub_q2, **retrieval_kwargs)
    reranked_2 = _rerank_fn(sub_q2, chunks_2)

    # Interleave-merge, deduplicate by chunk_id and text
    merged:     list[dict] = []
    seen_ids:   set[str]   = set()
    seen_texts: set[str]   = set()

    for c1, c2 in zip(reranked_1, reranked_2):
        for chunk in (c1, c2):
            text_key = " ".join(chunk["text"].split())
            cid      = chunk.get("chunk_id") or chunk.get("id", text_key[:40])
            if cid not in seen_ids and text_key not in seen_texts:
                merged.append(chunk)
                seen_ids.add(cid)
                seen_texts.add(text_key)

    for chunk in reranked_1 + reranked_2:
        text_key = " ".join(chunk["text"].split())
        cid      = chunk.get("chunk_id") or chunk.get("id", text_key[:40])
        if cid not in seen_ids and text_key not in seen_texts:
            merged.append(chunk)
            seen_ids.add(cid)
            seen_texts.add(text_key)

    logger.info(
        "Multi-part merge: %d + %d reranked → %d unique chunks",
        len(reranked_1), len(reranked_2), len(merged),
    )
    return merged


# ── Step 4: Informal rewrite (triggered only when regex classification fails) ──

_INFORMAL_REWRITE_PROMPT = """You are a search query optimizer for an enterprise security knowledge base \
covering IAM, PAM, SIEM, UEBA, vulnerability management, data classification, fraud detection, and incident response.

Rewrite the following informal or indirect question as a precise retrieval query using formal security terminology.

Rules:
- Use technical terms: "credential" not "password", "privileged account" not "admin account",
  "credential rotation" not "password rotation", "account risk tier" not "account type",
  "vulnerability remediation SLA" not "fix time", "data classification tier" not "data type"
- Strip personal context ("I'm a sys admin", "we're about to", "our team") — keep only the information need
- Make it formal, specific, and concise
- Return ONE query only — no explanation, no prefix

Original: "{query}"

Rewritten:"""


def _informal_rewrite(query: str, groq_model: str) -> str:
    """LLM rewrite for informal/indirect queries that didn't match any regex pattern."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return query

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": _INFORMAL_REWRITE_PROMPT.format(query=query)}],
            temperature=0.0,
            max_tokens=60,
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten and len(rewritten.split()) <= 40:
            logger.info("Informal rewrite: '%s' → '%s'", query[:60], rewritten[:60])
            return rewritten
    except Exception as e:
        logger.warning("Informal rewrite failed (%s) — using original.", e)

    return query


def _regex_classify(query: str) -> str | None:
    """
    Stage 1: classify via regex patterns.
    Priority order:
      1. Comparison — strongest explicit signal ("difference between", "vs")
      2. Factual    — "What is / What are" at query start overrides process patterns
      3. Metric     — KPI / score / threshold keywords
      4. Process    — "how does / how do / steps" (checked last to avoid false positives)
    Returns query type string or None if no pattern matches.
    """
    if any(p.search(query) for p in _COMPARISON_PATTERNS):
        return "comparison"
    if any(p.search(query) for p in _FACTUAL_PATTERNS):
        return "factual"
    if any(p.search(query) for p in _METRIC_PATTERNS):
        return "metric"
    if any(p.search(query) for p in _PROCESS_PATTERNS):
        return "process"
    return None   # fall through to LLM


# ── Stage 2: LLM fallback classifier ──────────────────────────────────────────

_CLASSIFICATION_PROMPT = """Classify this security query into exactly one category.

Categories:
- factual    : asks what something IS (definition, explanation)
- comparison : asks the DIFFERENCE or contrast between two things
- metric     : asks for KPIs, scores, thresholds, measurements
- process    : asks HOW something works, steps, workflow

Query: "{query}"

Respond with one word only — the category name:"""


def _llm_classify(query: str, groq_model: str) -> str:
    """
    Stage 2: LLM-based classification via a minimal Groq call.
    Uses low max_tokens (5) — only needs one word back.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — defaulting to 'factual' for classification.")
        return "factual"

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=groq_model,
            messages=[{
                "role":    "user",
                "content": _CLASSIFICATION_PROMPT.format(query=query),
            }],
            temperature=0.0,
            max_tokens=5,
        )
        label = response.choices[0].message.content.strip().lower()
        if label not in ("factual", "comparison", "metric", "process"):
            logger.warning("LLM classifier returned unexpected label '%s' — using 'factual'.", label)
            return "factual"
        return label
    except Exception as e:
        logger.warning("LLM classifier failed (%s) — defaulting to 'factual'.", e)
        return "factual"


# ── Public: classify_query ─────────────────────────────────────────────────────

def classify_query(query: str, use_llm_fallback: bool = True) -> str:
    """
    Classify a query into: factual | comparison | metric | process.

    Stage 1: regex (instant).
    Stage 2: LLM call (only if regex returns None and use_llm_fallback=True).

    Args:
        query:            Raw user query string.
        use_llm_fallback: Whether to call LLM if regex finds no signal.

    Returns:
        Query type string.
    """
    regex_result = _regex_classify(query)
    if regex_result:
        logger.info("Query classified as '%s' (regex) — '%s'", regex_result, query[:60])
        return regex_result

    if use_llm_fallback:
        config = _load_config()
        label  = _llm_classify(query, config.get("groq_model", "llama-3.1-8b-instant"))
        logger.info("Query classified as '%s' (LLM) — '%s'", label, query[:60])
        return label

    logger.info("Query classified as 'factual' (default) — '%s'", query[:60])
    return "factual"


# ── Comparison decomposition ───────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """A user asked a comparison question. Break it into exactly 2
independent sub-questions — one for each concept being compared. Each sub-question
should be self-contained and answerable on its own.

Original question: "{query}"

Respond with exactly 2 lines. Line 1 is sub-question 1. Line 2 is sub-question 2.
No numbering, no bullets, no extra text — just the two questions:"""


def _decompose_comparison(query: str, groq_model: str) -> tuple[str, str]:
    """
    Use LLM to split a comparison query into two focused sub-queries.
    Falls back to returning the original query twice if decomposition fails.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return query, query

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=groq_model,
            messages=[{
                "role":    "user",
                "content": _DECOMPOSE_PROMPT.format(query=query),
            }],
            temperature=0.0,
            max_tokens=80,
        )
        lines = [
            l.strip() for l in
            response.choices[0].message.content.strip().splitlines()
            if l.strip()
        ]
        if len(lines) >= 2:
            logger.info("Decomposed into:\n  [1] %s\n  [2] %s", lines[0], lines[1])
            return lines[0], lines[1]
    except Exception as e:
        logger.warning("Decomposition failed (%s) — using original query.", e)

    return query, query


# ── Query rewriting ────────────────────────────────────────────────────────────

_REWRITE_PROMPT = """You are a search query rewriter.

Given a conversation history and a follow-up question, rewrite the follow-up so it is
fully self-contained and can be understood without the conversation history.

Rules:
- Replace pronouns (it, they, that, this, the document, the policy) with the specific
  entity mentioned in the conversation.
- If the follow-up is already self-contained, return it unchanged.
- Return ONE sentence only — the rewritten query. No explanation, no prefix.

Conversation history (most recent last):
{history}

Follow-up question: {query}

Rewritten question:"""

# Trigger rewriting when query is short OR contains reference pronouns
_REFERENCE_WORDS = re.compile(
    r"\b(it|its|they|their|them|that|this|the document|the policy|"
    r"the procedure|the report|the same|the above|the mentioned)\b",
    re.IGNORECASE,
)


def _needs_rewrite(query: str, session_context: list[dict]) -> bool:
    """True if the query likely needs context to be understood."""
    if not session_context:
        return False
    return bool(_REFERENCE_WORDS.search(query)) or len(query.split()) < 8


def _rewrite_query(query: str, session_context: list[dict], groq_model: str) -> str:
    """
    Rewrite a follow-up query to be self-contained using the conversation history.
    Falls back to the original query if the LLM call fails.
    """
    history_lines = "\n".join(
        f"{t['role'].capitalize()}: {t['content']}"
        for t in session_context[-4:]   # last 4 turns (2 pairs) is enough
    )
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return query

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=groq_model,
            messages=[{
                "role":    "user",
                "content": _REWRITE_PROMPT.format(history=history_lines, query=query),
            }],
            temperature=0.0,
            max_tokens=60,
        )
        rewritten = response.choices[0].message.content.strip()
        # Sanity check — if rewrite is empty or too long, use original
        if rewritten and len(rewritten.split()) <= 40:
            logger.info("Query rewritten: '%s' → '%s'", query[:60], rewritten[:60])
            return rewritten
    except Exception as e:
        logger.warning("Query rewrite failed (%s) — using original query.", e)

    return query


# ── Public: route_and_retrieve ─────────────────────────────────────────────────

def route_and_retrieve(
    query:            str,
    top_k:            int | None   = None,
    embedding_model:  str | None   = None,
    vectorstore_path: str | None   = None,
    search_mode:      str | None   = None,
    query_type:       str | None   = None,
    session_context:  list[dict]   = None,
) -> tuple[list[dict], str, str]:
    """
    Returns (chunks, query_type, retrieval_query).
    retrieval_query is the rewritten query used for retrieval and reranking —
    may differ from the original query when pronouns were resolved.
    """
    """
    Classify the query, decompose if comparison, retrieve and merge chunks.

    Args:
        query:            User query string.
        top_k:            Chunks to retrieve. Defaults to config.
        embedding_model:  Embedding model. Defaults to config.
        vectorstore_path: ChromaDB path. Defaults to config.
        search_mode:      vector | bm25 | hybrid. Defaults to config.
        query_type:       Pre-supplied type (skips classification). Optional.

    Returns:
        (chunks, query_type) — chunks ready for reranker, detected query type.
    """
    config     = _load_config()
    top_k      = top_k or config["top_k"]
    groq_model = config.get("groq_model", "llama-3.1-8b-instant")

    # ── Step 1+2: Normalize (strip prefix + grammar fix) ─────────────────────
    normalized = _normalize_query(query)
    if normalized != query:
        logger.debug("Normalized: '%s' → '%s'", query[:60], normalized[:60])

    # ── Rewrite query if it contains unresolved references (#multi-turn) ──────
    retrieval_query = normalized
    if _needs_rewrite(normalized, session_context or []):
        retrieval_query = _rewrite_query(normalized, session_context or [], groq_model)

    retrieval_kwargs = dict(
        top_k            = top_k,
        embedding_model  = embedding_model,
        vectorstore_path = vectorstore_path,
        search_mode      = search_mode,
    )

    # ── Step 3: Multi-part detection → retrieve + rerank-per-sub + merge ──────
    if _is_multipart(retrieval_query):
        sub_q1, sub_q2 = _decompose_multipart(retrieval_query, groq_model)
        merged = _handle_multipart(retrieval_query, sub_q1, sub_q2, retrieval_kwargs, config)

        if not merged:
            # Fallback: both sub-queries returned 0 chunks (KB terminology mismatch).
            # Collapse to a single informal rewrite on the original query and retry.
            logger.info(
                "Multi-part fallback: 0 chunks from sub-queries — retrying with unified rewrite."
            )
            rewritten = _informal_rewrite(retrieval_query, groq_model)
            if rewritten != retrieval_query:
                from src.retrieval.reranker import rerank as _rerank_fn
                fb_chunks   = retrieve(rewritten, **retrieval_kwargs)
                merged      = _rerank_fn(rewritten, fb_chunks)
                retrieval_query = rewritten
                logger.info(
                    "Multi-part fallback: %d chunks via unified rewrite '%s'",
                    len(merged), rewritten[:60],
                )

        return merged, "multi_part", retrieval_query

    # ── Classify (regex first, then optionally LLM) ───────────────────────────
    if query_type:
        qtype = query_type
    else:
        regex_type = _regex_classify(retrieval_query)
        if regex_type is None:
            # ── Step 4: Informal rewrite — query didn't match any known pattern ──
            retrieval_query = _informal_rewrite(retrieval_query, groq_model)
            regex_type      = _regex_classify(retrieval_query)
        qtype = regex_type or _llm_classify(retrieval_query, groq_model)

    logger.info("Query classified as '%s' — '%s'", qtype, retrieval_query[:60])

    # ── Comparison: decompose → retrieve both → merge ─────────────────────────
    if qtype == "comparison":
        sub_q1, sub_q2 = _decompose_comparison(retrieval_query, groq_model)

        logger.info("Comparison retrieval: sub-query 1 = '%s'", sub_q1[:60])
        chunks_1 = retrieve(sub_q1, **retrieval_kwargs)

        logger.info("Comparison retrieval: sub-query 2 = '%s'", sub_q2[:60])
        chunks_2 = retrieve(sub_q2, **retrieval_kwargs)

        # Merge: interleave both lists to balance both sides, deduplicate by chunk_id
        merged:     list[dict] = []
        seen_ids:   set[str]   = set()
        seen_texts: set[str]   = set()

        for c1, c2 in zip(chunks_1, chunks_2):
            for chunk in (c1, c2):
                text_key = " ".join(chunk["text"].split())
                if chunk["chunk_id"] not in seen_ids and text_key not in seen_texts:
                    merged.append(chunk)
                    seen_ids.add(chunk["chunk_id"])
                    seen_texts.add(text_key)

        # Append any leftovers from the longer list
        for chunk in (chunks_1 + chunks_2):
            text_key = " ".join(chunk["text"].split())
            if chunk["chunk_id"] not in seen_ids and text_key not in seen_texts:
                merged.append(chunk)
                seen_ids.add(chunk["chunk_id"])
                seen_texts.add(text_key)

        # Cap at top_k * 2 to give reranker more candidates for comparison queries
        merged = merged[: top_k * 2]
        logger.info(
            "Comparison merge: %d + %d → %d unique chunks",
            len(chunks_1), len(chunks_2), len(merged),
        )
        return merged, qtype, retrieval_query

    # ── All other types: single retrieve ──────────────────────────────────────
    chunks = retrieve(retrieval_query, **retrieval_kwargs)
    return chunks, qtype, retrieval_query


# ── Smoke-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        # (query, expected_type)
        ("What is RBAC and how does it work?",                                "factual"),
        ("What is the difference between RBAC and ABAC?",                     "comparison"),
        ("What KPIs measure the effectiveness of a DLP program?",             "metric"),
        ("How does a SIEM system aggregate logs from multiple sources?",       "process"),
        ("Compare UEBA and SIEM for insider threat detection",                 "comparison"),
        ("What CVSS score indicates a critical vulnerability?",                "metric"),
        ("How does PAM enforce just-in-time access?",                         "process"),
        ("Insider threat versus external threat in fraud investigations",       "comparison"),
    ]

    print(f"\n{'='*65}")
    print("  QUERY ROUTER SMOKE-TEST")
    print(f"{'='*65}\n")
    print(f"  {'Query':<52} {'Expected':<12} {'Got':<12} {'Stage'}")
    print(f"  {'-'*85}")

    passed = 0
    for query, expected in test_queries:
        regex_hit = _regex_classify(query)
        stage     = "regex" if regex_hit else "LLM"
        got       = classify_query(query)
        ok        = got == expected
        if ok:
            passed += 1
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {query[:50]:<52} {expected:<12} {got:<12} ({stage})")

    print(f"\n  {passed}/{len(test_queries)} classification tests passed\n")

    # ── Decomposition test ─────────────────────────────────────────────────
    print(f"  {'─'*65}")
    print("  DECOMPOSITION TEST")
    print(f"  {'─'*65}\n")
    config = _load_config()
    comp_query = "What is the difference between RBAC and ABAC in enterprise IAM?"
    q1, q2 = _decompose_comparison(comp_query, config.get("groq_model", "llama-3.1-8b-instant"))
    print(f"  Original : {comp_query}")
    print(f"  Sub-Q 1  : {q1}")
    print(f"  Sub-Q 2  : {q2}")

    # ── Route and retrieve test ────────────────────────────────────────────
    print(f"\n  {'─'*65}")
    print("  ROUTE & RETRIEVE TEST (comparison)")
    print(f"  {'─'*65}\n")
    chunks, qtype = route_and_retrieve(comp_query, top_k=3)
    print(f"  Query type : {qtype}")
    print(f"  Chunks     : {len(chunks)} (expect up to 6 from merged dual retrieval)")
    sources = set(c['source_file'] for c in chunks)
    print(f"  Sources    : {sources}")
    print(f"  Diversity  : {len(sources)} unique documents\n")

    print(f"{'='*65}\n")
