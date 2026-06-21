"""
test_normalization.py
Tests the 10 real user questions through the normalization + routing layer.
Shows what transform was applied and how many chunks came back.
Does NOT call LLM generation — only route_and_retrieve.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.WARNING)   # suppress noise; we'll print our own output

from src.retrieval.query_router import (
    _normalize_query,
    _is_multipart,
    _regex_classify,
    route_and_retrieve,
)

# ── The 10 original user questions ────────────────────────────────────────────
QUESTIONS = [
    ("Q01", "show me the mean time to remediate critical vulnerabilities and what's the target for it?"),
    ("Q02", "is we meeting the SLA or not?"),
    ("Q03", "what are the key indicators of a phishing attack?"),
    ("Q04", "can you tell me about the incident response process?"),
    ("Q05", "what is the CVSS score and how do we calculate risk?"),
    ("Q06", "explain me the vulnerability disclosure policy"),
    ("Q07", "hows our patch management performance?"),
    ("Q08", "tell me what sections exist in the third party risk document"),
    ("Q09", "what is the mean time to detect threats?"),
    ("Q10", "what is the mean time to remediate critical vulnerabilities and whats the target for it?"),
]

SEP = "─" * 72


def analyse(qid: str, raw_query: str) -> None:
    print(f"\n{'═'*72}")
    print(f"  {qid}: {raw_query}")
    print(SEP)

    # Step 1+2: normalization
    normalized = _normalize_query(raw_query)
    norm_changed = normalized != raw_query
    print(f"  Normalized : {'YES → ' + repr(normalized) if norm_changed else 'unchanged'}")

    # Multi-part check (on normalized)
    mp = _is_multipart(normalized)
    print(f"  Multi-part : {'YES' if mp else 'no'}")

    # Regex classification (on normalized, without informal rewrite)
    regex_type = _regex_classify(normalized)
    print(f"  Regex type : {regex_type or '(none — informal rewrite will fire)'}")

    # Full route_and_retrieve (includes LLM calls where needed)
    print(f"  Routing... ", end="", flush=True)
    try:
        chunks, qtype, retrieval_query = route_and_retrieve(
            raw_query,
            search_mode="hybrid",
        )
        print(f"done")
        print(f"  Query type : {qtype}")
        if retrieval_query != normalized:
            print(f"  Retrieval Q: {retrieval_query!r}")
        print(f"  Chunks     : {len(chunks)} retrieved")
        if chunks:
            top_score = chunks[0].get("reranker_score") or chunks[0].get("score")
            sources   = sorted({c.get("source_file", "?") for c in chunks})
            print(f"  Top score  : {top_score:.4f}" if top_score else "  Top score  : n/a")
            print(f"  Sources    : {sources}")
        else:
            print("  ⚠ No chunks returned")
    except Exception as exc:
        print(f"ERROR: {exc}")


def main() -> None:
    print(f"\n{'═'*72}")
    print("  QUERY NORMALIZATION LAYER — 10 QUESTION ANALYSIS")
    print(f"{'═'*72}")
    print("  Tests: prefix strip | grammar fix | multi-part | informal rewrite")

    for qid, question in QUESTIONS:
        analyse(qid, question)

    print(f"\n{'═'*72}\n")


if __name__ == "__main__":
    main()
