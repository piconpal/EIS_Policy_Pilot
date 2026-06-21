"""
faithfulness.py
Core RAGAS-style Faithfulness logic, shared by:
  - ragas_eval.py  (batch offline evaluation on golden dataset)
  - pipeline.py    (async per-query check on live production traffic)

Algorithm:
  Step 1 — Statement extraction:
    LLM decomposes the answer into atomic, independently verifiable claims.
  Step 2 — NLI verification:
    For each claim, LLM checks: "Is this supported by the retrieved context?"
  Score:
    faithfulness = supported_statements / total_statements  ∈ [0.0, 1.0]
    Returns None when the answer has no verifiable claims (abstention / no-answer).

Public API:
    compute_faithfulness(answer, contexts, client, model)  → float | None
    compute_faithfulness_async(query_id, answer, contexts, groq_model) → None
"""

import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)


# ── Prompts ────────────────────────────────────────────────────────────────────

_STATEMENT_EXTRACTION_PROMPT = """Given an answer, extract every atomic factual statement it makes.
An atomic statement is one single fact that can be independently verified.

Rules:
- One statement per line
- No numbering, no bullets, no extra text
- If the answer says it has no information or cannot answer, output exactly: NO_STATEMENTS

Answer: "{answer}"

Statements:"""


_NLI_VERIFICATION_PROMPT = """Does the following statement logically follow from the context below?
Answer YES if the statement is fully supported or can be directly inferred.
Answer NO if it contradicts the context or introduces information not present in it.
One word only — YES or NO.

Context:
{context}

Statement: "{statement}"

Answer:"""


# ── Core LLM calls ─────────────────────────────────────────────────────────────

def _extract_statements(answer: str, client: Groq, model: str) -> list[str]:
    """Step 1: Decompose the LLM answer into atomic statements."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role":    "user",
                "content": _STATEMENT_EXTRACTION_PROMPT.format(answer=answer),
            }],
            temperature=0.0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        if not raw or raw == "NO_STATEMENTS":
            return []
        return [s.strip() for s in raw.splitlines() if s.strip()]
    except Exception as e:
        logger.warning("Statement extraction failed: %s", e)
        return []


def _verify_statement(statement: str, context: str, client: Groq, model: str) -> bool:
    """Step 2: Check whether a single statement is supported by the context."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role":    "user",
                "content": _NLI_VERIFICATION_PROMPT.format(
                    context=context, statement=statement
                ),
            }],
            temperature=0.0,
            max_tokens=5,
        )
        verdict = resp.choices[0].message.content.strip().upper()
        return verdict.startswith("YES")
    except Exception as e:
        logger.warning("NLI verification failed: %s", e)
        return False


# ── Public: synchronous ────────────────────────────────────────────────────────

def compute_faithfulness(
    answer:   str,
    contexts: list[str],
    client:   Groq,
    model:    str,
) -> dict:
    """
    Compute faithfulness for one (answer, contexts) pair.

    Returns a dict:
        {
            "faithfulness":         float | None,
            "total_statements":     int,
            "supported_statements": int,
            "statements":           list[{"statement": str, "supported": bool}],
        }
    """
    if not answer or not contexts:
        return {"faithfulness": None, "total_statements": 0,
                "supported_statements": 0, "statements": []}

    combined_context = "\n\n".join(contexts)
    statements       = _extract_statements(answer, client, model)

    if not statements:
        return {"faithfulness": None, "total_statements": 0,
                "supported_statements": 0, "statements": []}

    verified = []
    for stmt in statements:
        supported = _verify_statement(stmt, combined_context, client, model)
        verified.append({"statement": stmt, "supported": supported})

    supported_count = sum(1 for v in verified if v["supported"])
    score           = round(supported_count / len(verified), 4)

    return {
        "faithfulness":         score,
        "total_statements":     len(verified),
        "supported_statements": supported_count,
        "statements":           verified,
    }


# ── Public: async (fire-and-forget for production pipeline) ───────────────────

def compute_faithfulness_async(
    query_id:    str,
    answer:      str,
    contexts:    list[str],
    groq_model:  str,
) -> None:
    """
    Fire-and-forget: compute faithfulness in a background daemon thread.
    Updates the retrieval_logger row identified by query_id when done.
    Safe to call without awaiting — does not block the response path.
    """
    def _run() -> None:
        # Import here to avoid circular import at module load time
        from src.logging.retrieval_logger import update_faithfulness_score

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("Async faithfulness skipped — GROQ_API_KEY not set.")
            return

        client = Groq(api_key=api_key)
        try:
            result = compute_faithfulness(answer, contexts, client, groq_model)
            score  = result["faithfulness"]
            if score is not None:
                update_faithfulness_score(query_id, score)
                logger.info(
                    "Async faithfulness [%s]: %.4f (%d/%d statements supported)",
                    query_id[:8], score,
                    result["supported_statements"], result["total_statements"],
                )
            else:
                logger.debug(
                    "Async faithfulness [%s]: N/A — no verifiable statements.",
                    query_id[:8],
                )
        except Exception as exc:
            logger.warning("Async faithfulness failed [%s]: %s", query_id[:8], exc)

    thread = threading.Thread(target=_run, daemon=True, name=f"faith-{query_id[:8]}")
    thread.start()
