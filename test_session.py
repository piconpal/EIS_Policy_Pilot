"""
test_session.py
Demonstrates session_manager with a live 6-turn conversation.

Turns 1-3 : real queries through the full RAG pipeline (retrieve→rerank→generate).
Turns 4-6 : additional queries that push the session to its 5-turn cap and
            trigger summarization of the evicted turn before it is dropped.

After every turn prints:
  - Full session history (role + first 80 chars of content + token count)
  - Total tokens in context window vs budget

Summarization:
  When the session is at max capacity and a new turn arrives, the oldest turn
  is about to be evicted. We intercept it, summarise with Groq (cheap, fast),
  and prepend the summary as a "system context" note so long-run memory is not
  completely lost.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from groq import Groq
from src.retrieval.retriever  import retrieve
from src.retrieval.reranker   import rerank
from src.generation.llm_handler import generate_response
from src.context.session_manager import (
    add_turn, get_context, get_history,
    clear_session, session_stats,
    _MAX_TURNS, _TOKEN_BUDGET,
)

# ── Config ─────────────────────────────────────────────────────────────────────
SESSION_ID  = "test_session_001"
GROQ_MODEL  = "llama-3.1-8b-instant"

_groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

DIVIDER     = "─" * 68
BOLD_DIV    = "═" * 68

# ── Conversation turns (6 total) ───────────────────────────────────────────────
QUERIES = [
    "What is RBAC and how does it work?",
    "How does RBAC relate to Segregation of Duties (SoD)?",
    "What are the KPIs to measure RBAC effectiveness?",
    "What are common RBAC implementation pitfalls?",
    "How does RBAC integrate with Identity Governance frameworks?",
    "What is the difference between RBAC and Zero Trust architecture?",
]


# ── Summariser ─────────────────────────────────────────────────────────────────

def _summarise_turns(turns: list[dict]) -> str:
    """
    Call Groq to produce a compact summary of a list of conversation turns.
    Used to preserve memory of turns that are about to be evicted from the
    hot session store.
    """
    if not turns:
        return ""

    convo_text = "\n".join(
        f"{t['role'].upper()}: {t['content']}" for t in turns
    )
    prompt = (
        "Summarise the following conversation turns in 2-3 concise sentences, "
        "preserving the key topics, facts, and any conclusions reached. "
        "Write in third person ('The user asked…').\n\n"
        f"{convo_text}"
    )
    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


# ── Session-aware add with eviction interception ───────────────────────────────

_eviction_summaries: list[str] = []   # accumulated summaries of evicted pairs

def _add_with_eviction_guard(session_id: str, role: str, content: str) -> None:
    """
    Eviction fires ONLY when adding an assistant turn (completing a pair).
    This guarantees we always evict and summarise a full user+assistant
    exchange — never an orphaned user-only message.

    Pair-eviction logic:
      - deque maxlen=5 stores up to 5 individual messages
      - a full exchange = 2 messages (user + assistant)
      - when assistant turn would push us to 6 messages (> maxlen),
        the deque will auto-drop the oldest message — but we pre-emptively
        grab the oldest PAIR (indices 0 and 1) before that happens,
        summarise them together, then let the deque handle the drop.
    """
    if role == "assistant":
        history = get_history(session_id)
        # After adding this assistant turn we'd have len+1 messages.
        # If that exceeds _MAX_TURNS the deque will evict index 0.
        # Check that index 0 is a user turn and index 1 is its assistant
        # response — i.e. we have a complete pair to summarise.
        if len(history) >= _MAX_TURNS and len(history) >= 2:
            oldest_pair = history[0:2]
            if oldest_pair[0]["role"] == "user" and oldest_pair[1]["role"] == "assistant":
                print(
                    f"\n  ⚠  Turn cap reached ({_MAX_TURNS}). "
                    f"Evicting complete pair:\n"
                    f"     USER : {oldest_pair[0]['content'][:60]}...\n"
                    f"     ASST : {oldest_pair[1]['content'][:60]}..."
                )
                print("  Summarising pair via Groq before eviction...")
                summary = _summarise_turns(oldest_pair)
                _eviction_summaries.append(summary)
                print(f"  Summary stored: \"{summary[:120]}{'...' if len(summary)>120 else ''}\"")

    add_turn(session_id, role, content)


# ── Display helpers ────────────────────────────────────────────────────────────

def _print_session_state(turn_num: int) -> None:
    history = get_history(SESSION_ID)
    stats   = session_stats(SESSION_ID)
    ctx     = get_context(SESSION_ID)
    ctx_tokens = sum(t["tokens"] for t in ctx)

    print(f"\n  {DIVIDER}")
    print(f"  SESSION STATE after Turn {turn_num}")
    print(f"  {DIVIDER}")
    print(f"  Stored turns : {stats['turns_stored']}/{stats['max_turns']}")
    print(f"  Context window: {ctx_tokens} / {_TOKEN_BUDGET} tokens "
          f"({len(ctx)} turns fit in budget)")

    print(f"\n  History:")
    for i, t in enumerate(history, 1):
        preview = t['content'][:80].replace('\n', ' ')
        print(f"    [{i}] {t['role']:9s} | {t['tokens']:4d} tok | {preview}{'…' if len(t['content'])>80 else ''}")

    if _eviction_summaries:
        print(f"\n  Warm-memory summaries ({len(_eviction_summaries)} evicted turns captured):")
        for j, s in enumerate(_eviction_summaries, 1):
            print(f"    [{j}] {s[:120]}{'…' if len(s)>120 else ''}")

    budget_bar_filled = int((ctx_tokens / _TOKEN_BUDGET) * 30)
    budget_bar = "█" * budget_bar_filled + "░" * (30 - budget_bar_filled)
    pct = round(ctx_tokens / _TOKEN_BUDGET * 100, 1)
    print(f"\n  Token budget: [{budget_bar}] {pct}%")


# ── Full RAG pipeline ──────────────────────────────────────────────────────────

def _run_pipeline(query: str) -> str:
    """Retrieve → rerank → generate. Returns the answer string."""
    chunks   = retrieve(query)
    reranked = rerank(query, chunks)
    result   = generate_response(query, reranked)
    return result["answer"]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    clear_session(SESSION_ID)

    print(f"\n{BOLD_DIV}")
    print(f"  SESSION CONVERSATION TEST — {len(QUERIES)} turns")
    print(f"  session_id : {SESSION_ID}")
    print(f"  max_turns  : {_MAX_TURNS}  |  token_budget: {_TOKEN_BUDGET}")
    print(f"{BOLD_DIV}")

    for turn_num, query in enumerate(QUERIES, 1):
        print(f"\n{BOLD_DIV}")
        print(f"  TURN {turn_num}: {query}")
        print(BOLD_DIV)

        # 1. Store user query
        _add_with_eviction_guard(SESSION_ID, "user", query)

        # 2. Build context-aware prompt (inject prior conversation)
        ctx_turns = get_context(SESSION_ID)
        ctx_prefix = ""
        if len(ctx_turns) > 1:          # more than just the current user turn
            prior = ctx_turns[:-1]      # exclude the turn we just added
            ctx_prefix = "Previous conversation:\n" + "\n".join(
                f"  {t['role'].upper()}: {t['content']}" for t in prior
            ) + "\n\n"

        # Include any warm-memory summaries
        if _eviction_summaries:
            ctx_prefix = (
                "Earlier conversation summary (from warm memory):\n"
                + "\n".join(f"  - {s}" for s in _eviction_summaries)
                + "\n\n" + ctx_prefix
            )

        augmented_query = ctx_prefix + f"Current question: {query}"

        # 3. Run RAG pipeline
        print(f"\n  Retrieving and generating answer...")
        answer = _run_pipeline(augmented_query)

        # 4. Store assistant answer
        _add_with_eviction_guard(SESSION_ID, "assistant", answer)

        # 5. Print answer
        print(f"\n  Answer:")
        for line in answer.splitlines():
            print(f"    {line}")

        # 6. Print session state
        _print_session_state(turn_num)

    print(f"\n{BOLD_DIV}")
    print(f"  TEST COMPLETE — {len(QUERIES)} turns processed")
    final = session_stats(SESSION_ID)
    print(f"  Final history : {final['turns_stored']} turns stored")
    print(f"  Warm summaries: {len(_eviction_summaries)} evicted turns captured")
    print(f"{BOLD_DIV}\n")


if __name__ == "__main__":
    main()
