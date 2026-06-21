"""
session_manager.py — Step 10
In-memory per-session conversation history with token budget enforcement.

Fixes applied:
  - #12 Sessions idle for longer than SESSION_TTL_SECONDS (30 min) are evicted
        by a background daemon thread that runs every 5 minutes.
        clear_session() now also removes the lock entry to prevent lock leak.
  - #25 logging.basicConfig removed from module level.

Design:
  - Each session_id maps to a deque of turn dicts capped at session_history_turns.
  - Tokens are estimated as len(content) // 4.
  - get_context() returns turns newest-first until token_budget is exhausted,
    then reverses back to chronological order for prompt assembly.
  - Thread-safe: one Lock per session (fine-grained locking).

Public API:
    add_turn(session_id, role, content)  → None
    get_context(session_id)              → list[dict]   chronological
    get_history(session_id)              → list[dict]   all stored turns
    clear_session(session_id)            → None
    session_stats(session_id)            → dict
"""

import time
import threading
import logging
from collections import deque
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

_config       = _load_config()
_MAX_TURNS    = int(_config.get("session_history_turns", 5))
_TOKEN_BUDGET = int(_config.get("token_budget", 1500))

# Sessions idle longer than this are evicted by the GC thread (#12)
_SESSION_TTL_SECONDS = int(_config.get("session_ttl_seconds", 1800))
_GC_INTERVAL_SECONDS = int(_config.get("session_gc_interval_seconds", 300))


# ── In-memory store ────────────────────────────────────────────────────────────

_sessions:      dict[str, deque]            = {}
_session_locks: dict[str, threading.Lock]   = {}
_last_accessed: dict[str, float]            = {}   # session_id → monotonic timestamp
_store_lock = threading.Lock()   # guards creation/deletion of session entries


def _get_lock(session_id: str) -> threading.Lock:
    with _store_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


def _touch(session_id: str) -> None:
    """Update the last-accessed timestamp for a session."""
    with _store_lock:
        _last_accessed[session_id] = time.monotonic()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── Background GC thread (#12) ─────────────────────────────────────────────────

def _cleanup_expired_sessions() -> None:
    """Evict sessions that have been idle for longer than _SESSION_TTL_SECONDS."""
    now = time.monotonic()
    with _store_lock:
        expired = [
            sid for sid, t in _last_accessed.items()
            if now - t > _SESSION_TTL_SECONDS
        ]
        for sid in expired:
            _sessions.pop(sid, None)
            _session_locks.pop(sid, None)
            _last_accessed.pop(sid, None)
    if expired:
        logger.info("Session GC: evicted %d expired session(s) (idle > %ds).",
                    len(expired), _SESSION_TTL_SECONDS)


def _start_gc_thread() -> None:
    def _run() -> None:
        while True:
            time.sleep(_GC_INTERVAL_SECONDS)
            try:
                _cleanup_expired_sessions()
            except Exception as exc:
                logger.warning("Session GC error: %s", exc)

    t = threading.Thread(target=_run, daemon=True, name="session-gc")
    t.start()
    logger.debug("Session GC thread started (interval=%ds, TTL=%ds).",
                 _GC_INTERVAL_SECONDS, _SESSION_TTL_SECONDS)

_start_gc_thread()


# ── Public API ─────────────────────────────────────────────────────────────────

def add_turn(session_id: str, role: str, content: str) -> None:
    """
    Append a turn to the session history.

    Args:
        session_id: Unique identifier for the conversation session.
        role:       "user" or "assistant".
        content:    Text content of the turn.

    Raises:
        ValueError: If role is not "user" or "assistant".
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', got '{role}'")

    turn = {"role": role, "content": content, "tokens": _estimate_tokens(content)}

    # Ensure deque exists, then append — avoid holding _store_lock during append
    with _store_lock:
        if session_id not in _sessions:
            _sessions[session_id] = deque(maxlen=_MAX_TURNS)
        _last_accessed[session_id] = time.monotonic()

    lock = _get_lock(session_id)
    with lock:
        _sessions[session_id].append(turn)

    logger.debug(
        "Session '%s' — added %s turn (%d tokens). History: %d/%d turns.",
        session_id, role, turn["tokens"], len(_sessions[session_id]), _MAX_TURNS,
    )


def get_context(session_id: str) -> list[dict]:
    """
    Return conversation turns that fit within token_budget, oldest-first.
    Turns are selected newest-first so recent context is always included.
    """
    _touch(session_id)
    lock = _get_lock(session_id)
    with lock:
        turns = list(_sessions.get(session_id, []))

    if not turns:
        return []

    selected: list[dict] = []
    budget_remaining = _TOKEN_BUDGET
    for turn in reversed(turns):
        if turn["tokens"] > budget_remaining:
            break
        selected.append(turn)
        budget_remaining -= turn["tokens"]

    selected.reverse()
    logger.debug(
        "Session '%s' — context: %d/%d turns, ~%d tokens used.",
        session_id, len(selected), len(turns), _TOKEN_BUDGET - budget_remaining,
    )
    return selected


def get_history(session_id: str) -> list[dict]:
    """Return all stored turns for a session, chronological. No token filtering."""
    lock = _get_lock(session_id)
    with lock:
        return list(_sessions.get(session_id, []))


def clear_session(session_id: str) -> None:
    """
    Remove all history for a session and release its lock entry (#12).
    """
    with _store_lock:
        _sessions.pop(session_id, None)
        _session_locks.pop(session_id, None)
        _last_accessed.pop(session_id, None)
    logger.info("Session '%s' cleared.", session_id)


def session_stats(session_id: str) -> dict:
    """Return usage stats for a session without modifying history."""
    history      = get_history(session_id)
    total_tokens = sum(t["tokens"] for t in history)
    return {
        "session_id":        session_id,
        "turns_stored":      len(history),
        "max_turns":         _MAX_TURNS,
        "tokens_in_history": total_tokens,
        "token_budget":      _TOKEN_BUDGET,
        "tokens_remaining":  max(0, _TOKEN_BUDGET - total_tokens),
    }


# ── Smoke-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    SID = "test-session-001"

    print(f"\n{'='*60}\n  SESSION MANAGER SMOKE-TEST\n{'='*60}\n")

    turns = [
        ("user",      "What is RBAC?"),
        ("assistant", "RBAC stands for Role-Based Access Control. It assigns permissions to roles "
                      "rather than individuals, and users inherit permissions through role membership."),
        ("user",      "How does it differ from ABAC?"),
        ("assistant", "ABAC makes access decisions based on attributes of the user, resource, and "
                      "environment rather than predefined roles."),
        ("user",      "Which is better for large enterprises?"),
        ("assistant", "RBAC is generally preferred for large enterprises due to its simplicity and "
                      "scalability, while ABAC suits fine-grained dynamic policies."),
    ]

    for role, content in turns:
        add_turn(SID, role, content)

    history = get_history(SID)
    print(f"[Test 1] Added 6 turns, max_turns={_MAX_TURNS}")
    print(f"  Stored turns: {len(history)} (expected {_MAX_TURNS})")
    assert len(history) == _MAX_TURNS, "FAIL: deque should cap at max_turns"
    print(f"  PASS: deque capped correctly\n")

    ctx = get_context(SID)
    ctx_tokens = sum(t["tokens"] for t in ctx)
    print(f"[Test 2] get_context() with budget={_TOKEN_BUDGET}")
    print(f"  Context turns: {len(ctx)}, tokens used: ~{ctx_tokens}")
    assert ctx_tokens <= _TOKEN_BUDGET, "FAIL: context exceeds token budget"
    print(f"  PASS: context fits within budget\n")

    stats = session_stats(SID)
    print(f"[Test 3] session_stats():")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    assert stats["turns_stored"] == _MAX_TURNS
    assert stats["tokens_remaining"] >= 0
    print(f"  PASS\n")

    clear_session(SID)
    assert get_history(SID) == [], "FAIL: history should be empty after clear"
    assert SID not in _session_locks, "FAIL: lock entry should be removed after clear"
    print(f"[Test 4] clear_session() — PASS: history and lock entry removed\n")

    try:
        add_turn(SID, "system", "You are a helpful assistant.")
        print("[Test 5] FAIL: should have raised ValueError")
    except ValueError as e:
        print(f"[Test 5] PASS: ValueError raised — {e}\n")

    print(f"{'='*60}\n  All tests passed.\n{'='*60}\n")
