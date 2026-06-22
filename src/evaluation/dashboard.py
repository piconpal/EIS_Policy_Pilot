"""
dashboard.py
KPI Dashboard for the Enterprise RAG pipeline.

Reads from retrieval_log.db (production rows only: is_eval=0) and computes:
  - User & volume metrics
  - Latency percentiles (p50, p95, p99)
  - Quality metrics (denial rate, rewrite rate, faithfulness score)
  - Drift signals (week-over-week changes in key quality metrics)

Flags are raised when drift thresholds are breached:
  - top_chunk_score drops >10% vs prior 4-week average
  - denial rate increases >15% vs prior 4 weeks
  - faithfulness_score drops >0.05 vs prior 4 weeks

Run:
    python -m src.evaluation.dashboard

Output:
    logs/kpi_report.json
"""

import json
import logging
import statistics
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

_config = _load_config()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _percentile(values: list[float], p: float) -> float | None:
    """Return the p-th percentile (0-100) of a sorted list. Returns None if empty."""
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = (p / 100) * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return round(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]), 1)


def _utc_days_ago(n: int) -> str:
    """Return ISO timestamp string for N days ago in UTC."""
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


def _fetch_scalars(sql: str, params: dict | None = None) -> list:
    """Execute SQL and return list of first-column values."""
    with Session(_engine) as db:
        rows = db.execute(text(sql), params or {}).fetchall()
    return [r[0] for r in rows if r[0] is not None]


def _fetch_one(sql: str, params: dict | None = None):
    with Session(_engine) as db:
        row = db.execute(text(sql), params or {}).fetchone()
    return row


# ── KPI Computation ─────────────────────────────────────────────────────────────

def _volume_metrics() -> dict:
    row = _fetch_one("""
        SELECT
            COUNT(*)                    AS total_queries,
            COUNT(DISTINCT session_id)  AS unique_sessions,
            SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) AS cache_hits
        FROM retrieval_logs
        WHERE is_eval = 0
    """)
    total     = row[0] or 0
    cache_hits = row[2] or 0

    last7  = _fetch_one(
        "SELECT COUNT(*) FROM retrieval_logs WHERE is_eval=0 AND timestamp >= :ts",
        {"ts": _utc_days_ago(7)},
    )[0] or 0

    last30 = _fetch_one(
        "SELECT COUNT(*) FROM retrieval_logs WHERE is_eval=0 AND timestamp >= :ts",
        {"ts": _utc_days_ago(30)},
    )[0] or 0

    return {
        "total_queries":     total,
        "unique_sessions":   row[1] or 0,
        "queries_last_7d":   last7,
        "queries_last_30d":  last30,
        "cache_hit_rate":    round(cache_hits / total, 4) if total > 0 else 0.0,
    }


def _latency_metrics() -> dict:
    def _lat_stats(col: str) -> dict:
        vals = _fetch_scalars(
            f"SELECT {col} FROM retrieval_logs WHERE is_eval=0 AND {col} IS NOT NULL"
        )
        if not vals:
            return {"mean": None, "p50": None, "p95": None, "p99": None}
        return {
            "mean": round(statistics.mean(vals), 1),
            "p50":  _percentile(vals, 50),
            "p95":  _percentile(vals, 95),
            "p99":  _percentile(vals, 99),
        }

    total_lat = _lat_stats("latency_ms")
    return {
        "total_latency":     total_lat,
        "retriever_latency": _lat_stats("retriever_latency_ms"),
        "reranker_latency":  _lat_stats("reranker_latency_ms"),
        "llm_latency":       _lat_stats("llm_latency_ms"),
        "p95_total_flag":    (
            total_lat["p95"] is not None and total_lat["p95"] > 10_000
        ),
    }


def _quality_metrics() -> dict:
    # Denial rate
    denial_row = _fetch_one("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN response_text LIKE '%could not find relevant information%' THEN 1 ELSE 0 END) AS denials
        FROM retrieval_logs
        WHERE is_eval=0 AND input_safe=1
    """)
    total   = denial_row[0] or 0
    denials = denial_row[1] or 0
    denial_rate = round(denials / total, 4) if total > 0 else 0.0

    # Mean top_chunk_score — rolling 4-week average
    chunk_scores = _fetch_scalars("""
        SELECT top_chunk_score FROM retrieval_logs
        WHERE is_eval=0 AND top_chunk_score IS NOT NULL
          AND timestamp >= :ts
    """, {"ts": _utc_days_ago(28)})
    mean_top_score_4w = round(statistics.mean(chunk_scores), 4) if chunk_scores else None

    # Rewrite rate
    rewrite_row = _fetch_one("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN rewrite_attempted = 1 THEN 1 ELSE 0 END) AS rewrites
        FROM retrieval_logs WHERE is_eval=0
    """)
    rw_total   = rewrite_row[0] or 0
    rw_count   = rewrite_row[1] or 0
    rewrite_rate = round(rw_count / rw_total, 4) if rw_total > 0 else 0.0

    # Mean faithfulness
    faith_row = _fetch_one("""
        SELECT AVG(faithfulness_score)
        FROM retrieval_logs
        WHERE is_eval=0 AND faithfulness_score IS NOT NULL
    """)
    mean_faith = round(faith_row[0], 4) if faith_row and faith_row[0] is not None else None

    # Feedback breakdown
    fb_row = _fetch_one("""
        SELECT
            SUM(CASE WHEN feedback_score =  1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN feedback_score = -1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN feedback_score IS NULL THEN 1 ELSE 0 END)
        FROM retrieval_logs WHERE is_eval=0
    """)
    return {
        "denial_rate":             denial_rate,
        "mean_top_chunk_score_4w": mean_top_score_4w,
        "rewrite_rate":            rewrite_rate,
        "mean_faithfulness_score": mean_faith,
        "feedback": {
            "positive":    int(fb_row[0] or 0),
            "negative":    int(fb_row[1] or 0),
            "no_feedback": int(fb_row[2] or 0),
        },
    }


def _drift_signals() -> tuple[dict, list[str]]:
    """
    Compute week-over-week drift signals.
    Returns (drift_metrics dict, flags list).
    """
    now_ts      = _utc_days_ago(0)
    week_ago    = _utc_days_ago(7)
    four_weeks  = _utc_days_ago(35)

    # Current week: last 7 days
    # Prior period: days 7–35 (4-week baseline)

    # ── top_chunk_score drift ─────────────────────────────────────────────────
    cur_scores = _fetch_scalars("""
        SELECT top_chunk_score FROM retrieval_logs
        WHERE is_eval=0 AND top_chunk_score IS NOT NULL
          AND timestamp >= :w AND timestamp < :n
    """, {"w": week_ago, "n": now_ts})

    prior_scores = _fetch_scalars("""
        SELECT top_chunk_score FROM retrieval_logs
        WHERE is_eval=0 AND top_chunk_score IS NOT NULL
          AND timestamp >= :f AND timestamp < :w
    """, {"f": four_weeks, "w": week_ago})

    cur_score_avg   = round(statistics.mean(cur_scores),   4) if cur_scores   else None
    prior_score_avg = round(statistics.mean(prior_scores), 4) if prior_scores else None

    # ── denial rate drift ─────────────────────────────────────────────────────
    def _denial_rate_for(ts_from: str, ts_to: str) -> float | None:
        row = _fetch_one("""
            SELECT COUNT(*),
                   SUM(CASE WHEN response_text LIKE '%could not find relevant information%' THEN 1 ELSE 0 END)
            FROM retrieval_logs
            WHERE is_eval=0 AND input_safe=1
              AND timestamp >= :f AND timestamp < :t
        """, {"f": ts_from, "t": ts_to})
        if not row or not row[0]:
            return None
        return round((row[1] or 0) / row[0], 4)

    cur_denial   = _denial_rate_for(week_ago, now_ts)
    prior_denial = _denial_rate_for(four_weeks, week_ago)

    # ── faithfulness drift ────────────────────────────────────────────────────
    cur_faith_row = _fetch_one("""
        SELECT AVG(faithfulness_score) FROM retrieval_logs
        WHERE is_eval=0 AND faithfulness_score IS NOT NULL
          AND timestamp >= :w AND timestamp < :n
    """, {"w": week_ago, "n": now_ts})
    cur_faith = round(cur_faith_row[0], 4) if cur_faith_row and cur_faith_row[0] is not None else None

    prior_faith_row = _fetch_one("""
        SELECT AVG(faithfulness_score) FROM retrieval_logs
        WHERE is_eval=0 AND faithfulness_score IS NOT NULL
          AND timestamp >= :f AND timestamp < :w
    """, {"f": four_weeks, "w": week_ago})
    prior_faith = round(prior_faith_row[0], 4) if prior_faith_row and prior_faith_row[0] is not None else None

    # ── Evaluate flags ────────────────────────────────────────────────────────
    flags: list[str] = []

    score_drop = None
    if cur_score_avg is not None and prior_score_avg is not None and prior_score_avg > 0:
        score_drop = round((prior_score_avg - cur_score_avg) / prior_score_avg, 4)
        if score_drop > 0.10:
            flags.append(
                f"DRIFT: top_chunk_score dropped {score_drop*100:.1f}% "
                f"(current={cur_score_avg}, prior_4w={prior_score_avg})"
            )

    denial_increase = None
    if cur_denial is not None and prior_denial is not None and prior_denial > 0:
        denial_increase = round((cur_denial - prior_denial) / prior_denial, 4)
        if denial_increase > 0.15:
            flags.append(
                f"DRIFT: denial rate increased {denial_increase*100:.1f}% "
                f"(current={cur_denial:.3f}, prior_4w={prior_denial:.3f})"
            )

    faith_drop = None
    if cur_faith is not None and prior_faith is not None:
        faith_drop = round(prior_faith - cur_faith, 4)
        if faith_drop > 0.05:
            flags.append(
                f"DRIFT: faithfulness_score dropped {faith_drop:.4f} "
                f"(current={cur_faith}, prior_4w={prior_faith})"
            )

    drift = {
        "current_week_top_chunk_score":   cur_score_avg,
        "prior_4w_top_chunk_score":       prior_score_avg,
        "score_drop_fraction":            score_drop,
        "current_week_denial_rate":       cur_denial,
        "prior_4w_denial_rate":           prior_denial,
        "denial_rate_increase_fraction":  denial_increase,
        "current_week_faithfulness":      cur_faith,
        "prior_4w_faithfulness":          prior_faith,
        "faithfulness_drop_absolute":     faith_drop,
    }
    return drift, flags


# ── Main ────────────────────────────────────────────────────────────────────────

def run_dashboard() -> dict:
    logger.info("Computing KPI dashboard from retrieval_log.db …")

    volume   = _volume_metrics()
    latency  = _latency_metrics()
    quality  = _quality_metrics()
    drift, flags = _drift_signals()

    # Add p95 latency flag to flags list
    if latency["p95_total_flag"]:
        flags.append(
            f"LATENCY: p95 total latency is {latency['total_latency']['p95']}ms (threshold: 10,000ms)"
        )

    report = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "volume":        volume,
        "latency":       latency,
        "quality":       quality,
        "drift":         drift,
        "flags":         flags,
    }

    report_path = _PROJECT_ROOT / _config.get("kpi_report_path", "logs/kpi_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    # ── Print formatted report ────────────────────────────────────────────────
    W = 62
    print(f"\n{'='*W}")
    print(f"  KPI DASHBOARD — {report['generated_at'][:19]} UTC")
    print(f"{'='*W}")

    print(f"\n  VOLUME")
    print(f"  {'Total queries (prod)':<35} {volume['total_queries']:>8}")
    print(f"  {'Unique sessions':<35} {volume['unique_sessions']:>8}")
    print(f"  {'Queries last 7 days':<35} {volume['queries_last_7d']:>8}")
    print(f"  {'Queries last 30 days':<35} {volume['queries_last_30d']:>8}")
    print(f"  {'Cache hit rate':<35} {volume['cache_hit_rate']:>8.2%}")

    print(f"\n  LATENCY (ms)")
    for label, key in [
        ("Total",     "total_latency"),
        ("Retriever", "retriever_latency"),
        ("Reranker",  "reranker_latency"),
        ("LLM",       "llm_latency"),
    ]:
        s = latency[key]
        mean = f"{s['mean']:.1f}" if s['mean'] is not None else "N/A"
        p50  = f"{s['p50']:.1f}"  if s['p50']  is not None else "N/A"
        p95  = f"{s['p95']:.1f}"  if s['p95']  is not None else "N/A"
        p99  = f"{s['p99']:.1f}"  if s['p99']  is not None else "N/A"
        print(f"  {label:<12} mean={mean:>9}  p50={p50:>9}  p95={p95:>9}  p99={p99:>9}")

    print(f"\n  QUALITY")
    print(f"  {'Denial rate':<35} {quality['denial_rate']:>8.2%}")
    print(f"  {'Rewrite rate':<35} {quality['rewrite_rate']:>8.2%}")
    mts = quality['mean_top_chunk_score_4w']
    print(f"  {'Mean top_chunk_score (4w)':<35} {f'{mts:.4f}' if mts is not None else 'N/A':>8}")
    mf = quality['mean_faithfulness_score']
    print(f"  {'Mean faithfulness score':<35} {f'{mf:.4f}' if mf is not None else 'N/A':>8}")
    fb = quality['feedback']
    print(f"  {'Feedback +1 / -1 / none':<35} {fb['positive']:>3} / {fb['negative']:>3} / {fb['no_feedback']:>5}")

    print(f"\n  DRIFT SIGNALS (current week vs prior 4 weeks)")
    cws  = drift['current_week_top_chunk_score']
    p4ws = drift['prior_4w_top_chunk_score']
    print(f"  top_chunk_score  curr={f'{cws:.4f}' if cws is not None else 'N/A'}  prior={f'{p4ws:.4f}' if p4ws is not None else 'N/A'}")
    cwd  = drift['current_week_denial_rate']
    p4wd = drift['prior_4w_denial_rate']
    print(f"  denial_rate      curr={f'{cwd:.3f}' if cwd is not None else 'N/A'}   prior={f'{p4wd:.3f}' if p4wd is not None else 'N/A'}")
    cwf  = drift['current_week_faithfulness']
    p4wf = drift['prior_4w_faithfulness']
    print(f"  faithfulness     curr={f'{cwf:.4f}' if cwf is not None else 'N/A'}  prior={f'{p4wf:.4f}' if p4wf is not None else 'N/A'}")

    if flags:
        print(f"\n  FLAGS ({len(flags)})")
        for f in flags:
            print(f"  ⚑  {f}")
    else:
        print(f"\n  FLAGS  None — all metrics within thresholds.")

    print(f"\n  Report saved → {report_path}")
    print(f"{'='*W}\n")

    return report


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_dashboard()
