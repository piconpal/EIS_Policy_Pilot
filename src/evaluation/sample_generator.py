"""
sample_generator.py
Stratified Weekly Sample Generator for Human Review.

Pulls 20 queries from the last 7 days of production traffic
(is_eval=0, input_safe=1) using stratified selection:
  - Stratum A — 5 hardest retrievals: bottom quartile top_chunk_score
  - Stratum B — 5 rewrite cases: rewrite_attempted = 1
  - Stratum C — 5 lowest faithfulness: non-null faithfulness_score, ascending
  - Stratum D — 5 random: remaining rows not already selected

Each stratum is filled to its quota first; if a stratum has fewer
rows than its quota the remainder rolls over to Stratum D random pool.

Run:
    python -m src.evaluation.sample_generator

Output:
    logs/weekly_sample.json
"""

import json
import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.logging.retrieval_logger import _engine

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

_config       = _load_config()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _row_to_dict(row) -> dict:
    """Convert a SQLAlchemy Row to a plain dict."""
    return dict(row._mapping)


def _pick(pool: list[dict], n: int, seen_ids: set) -> list[dict]:
    """Pick up to n rows from pool that are not already in seen_ids."""
    chosen = []
    for row in pool:
        if len(chosen) >= n:
            break
        if row["id"] not in seen_ids:
            chosen.append(row)
            seen_ids.add(row["id"])
    return chosen


# ── Main ────────────────────────────────────────────────────────────────────────

def run_sample_generator(
    total: int = 20,
    days: int = 7,
) -> dict:
    sample_path = _PROJECT_ROOT / _config.get("weekly_sample_path", "logs/weekly_sample.json")
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # ── 1. Load all production rows from last 7 days ──────────────────────────
    with Session(_engine) as db:
        rows = db.execute(text("""
            SELECT
                id,
                query_text,
                session_id,
                search_mode,
                chunks_retrieved,
                chunks_reranked,
                top_chunk_score,
                avg_retriever_score,
                rewrite_attempted,
                faithfulness_score,
                sources_cited,
                response_text,
                latency_ms,
                created_at
            FROM retrieval_logs
            WHERE is_eval = 0
              AND input_safe = 1
              AND query_text IS NOT NULL
              AND LENGTH(TRIM(query_text)) > 0
              AND created_at >= :cutoff
            ORDER BY created_at DESC
        """), {"cutoff": cutoff_ts}).fetchall()

    all_rows = [_row_to_dict(r) for r in rows]

    if not all_rows:
        logger.warning("No production queries found in the last %d days.", days)
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_days":  days,
            "total_pool":   0,
            "sample_size":  0,
            "skipped":      True,
            "reason":       "No production queries in the last 7 days",
            "sample":       [],
        }
        sample_path.write_text(json.dumps(report, indent=2))
        print(f"\n  Sample skipped — no queries in last {days} days.\n")
        return report

    logger.info(
        "Sample generator: %d rows in last %d days → selecting %d",
        len(all_rows), days, total,
    )

    per_stratum = total // 4          # 5 each
    extra       = total - per_stratum * 4   # handle non-divisible totals

    seen_ids: set = set()
    sample:   list[dict] = []

    # ── Stratum A: bottom quartile top_chunk_score ────────────────────────────
    scored = [r for r in all_rows if r.get("top_chunk_score") is not None]
    scored.sort(key=lambda r: r["top_chunk_score"])          # ascending = worst first
    q25_cutoff_idx = max(1, len(scored) // 4)
    bottom_q = scored[:q25_cutoff_idx]
    stratum_a = _pick(bottom_q, per_stratum, seen_ids)
    for r in stratum_a:
        r["_stratum"] = "A_low_chunk_score"
    sample.extend(stratum_a)

    # ── Stratum B: rewrite_attempted = 1 ──────────────────────────────────────
    rewrites = [r for r in all_rows if r.get("rewrite_attempted") == 1]
    stratum_b = _pick(rewrites, per_stratum, seen_ids)
    for r in stratum_b:
        r["_stratum"] = "B_rewrite_attempted"
    sample.extend(stratum_b)

    # ── Stratum C: lowest faithfulness_score (non-null) ───────────────────────
    faithful = [r for r in all_rows if r.get("faithfulness_score") is not None]
    faithful.sort(key=lambda r: r["faithfulness_score"])     # ascending = worst first
    stratum_c = _pick(faithful, per_stratum, seen_ids)
    for r in stratum_c:
        r["_stratum"] = "C_low_faithfulness"
    sample.extend(stratum_c)

    # ── Stratum D: random from remaining ──────────────────────────────────────
    remaining = [r for r in all_rows if r["id"] not in seen_ids]
    random.shuffle(remaining)
    quota_d   = per_stratum + extra + (total - len(sample) - per_stratum - extra)
    # Simpler: just fill up to the total
    quota_d   = total - len(sample)
    stratum_d = _pick(remaining, quota_d, seen_ids)
    for r in stratum_d:
        r["_stratum"] = "D_random"
    sample.extend(stratum_d)

    # ── Serialise sources_cited (stored as JSON string in DB) ─────────────────
    for r in sample:
        sc = r.get("sources_cited")
        if isinstance(sc, str):
            try:
                r["sources_cited"] = json.loads(sc)
            except (json.JSONDecodeError, TypeError):
                r["sources_cited"] = []

    # ── Build counts by stratum ────────────────────────────────────────────────
    stratum_counts = {}
    for r in sample:
        s = r["_stratum"]
        stratum_counts[s] = stratum_counts.get(s, 0) + 1

    report = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "window_days":    days,
        "total_pool":     len(all_rows),
        "sample_size":    len(sample),
        "skipped":        False,
        "stratum_counts": stratum_counts,
        "sample":         sample,
    }
    sample_path.write_text(json.dumps(report, indent=2, default=str))

    # ── Print summary ──────────────────────────────────────────────────────────
    W = 62
    print(f"\n{'='*W}")
    print(f"  WEEKLY SAMPLE REPORT — {len(sample)} queries selected")
    print(f"{'='*W}")
    print(f"  Pool (last {days} days) : {len(all_rows)} production queries")
    for stratum, count in sorted(stratum_counts.items()):
        label = {
            "A_low_chunk_score":   "Low chunk score (bottom quartile)",
            "B_rewrite_attempted": "Rewrite attempted",
            "C_low_faithfulness":  "Low faithfulness score",
            "D_random":            "Random remaining",
        }.get(stratum, stratum)
        print(f"  {stratum[0]}: {count:2d}  {label}")
    print(f"\n  Sample saved → {sample_path}")
    print(f"{'='*W}\n")

    return report


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_sample_generator()
