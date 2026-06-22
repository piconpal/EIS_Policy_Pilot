"""
run_monitoring.py
Continuous Evaluation & Drift Monitoring — Orchestration Runner.

Runs all four monitoring components in sequence and prints a
consolidated summary with all actionable flags.

Components:
  1. dashboard.py       — KPI metrics + drift signals
  2. canary.py          — Regression detection vs baseline
  3. query_clusters.py  — Topic clustering + growth flags
  4. sample_generator.py — Stratified weekly review sample

Run:
    python run_monitoring.py

All output files are written to logs/:
    logs/kpi_report.json
    logs/canary_report.json   (+ logs/canary_baseline.json on first run)
    logs/cluster_report.json
    logs/weekly_sample.json
"""

import logging
import sys
import traceback
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.WARNING,          # suppress verbose INFO from sub-modules
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

W = 64


def _section(title: str) -> None:
    print(f"\n{'─'*W}")
    print(f"  {title}")
    print(f"{'─'*W}")


def _run_step(name: str, fn) -> dict | None:
    _section(f"STEP: {name}")
    try:
        return fn()
    except Exception as exc:
        print(f"\n  [ERROR] {name} failed: {exc}")
        traceback.print_exc()
        return None


def main() -> int:
    print(f"\n{'='*W}")
    print(f"  ENTERPRISE RAG — MONITORING RUN")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*W}")

    # ── Step 1: KPI Dashboard ─────────────────────────────────────────────────
    from src.evaluation.dashboard import run_dashboard
    kpi = _run_step("KPI Dashboard", run_dashboard)

    # ── Step 2: Canary Runner ─────────────────────────────────────────────────
    from src.evaluation.canary import run_canary
    canary = _run_step("Canary Runner", run_canary)

    # ── Step 3: Query Clusters ────────────────────────────────────────────────
    from src.evaluation.query_clusters import run_query_clusters
    clusters = _run_step("Query Clusters", run_query_clusters)

    # ── Step 4: Weekly Sample ─────────────────────────────────────────────────
    from src.evaluation.sample_generator import run_sample_generator
    sample = _run_step("Weekly Sample Generator", run_sample_generator)

    # ── Consolidated Summary ──────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("  CONSOLIDATED MONITORING SUMMARY")
    print(f"{'='*W}")

    exit_code = 0

    # KPI drift flags
    if kpi:
        drift_flags = kpi.get("drift_signals", {}).get("flags", [])
        if drift_flags:
            print(f"\n  KPI DRIFT FLAGS ({len(drift_flags)})")
            for flag in drift_flags:
                print(f"    ⚑  {flag}")
            exit_code = 1
        else:
            print("\n  KPI Drift          : CLEAR")

        vol = kpi.get("volume", {})
        qual = kpi.get("quality", {})
        lat  = kpi.get("latency", {})
        print(f"  Queries (7d)       : {vol.get('total_queries_7d', 'n/a')}")
        p95 = lat.get("latency_ms", {}).get("p95")
        print(f"  Latency p95        : {f'{p95:.0f}ms' if p95 else 'n/a'}")
        print(f"  Denial rate        : {qual.get('denial_rate_pct', 'n/a')}%")
        print(f"  Faithfulness (4w)  : {qual.get('faithfulness_4w_avg', 'n/a')}")
    else:
        print("\n  KPI Dashboard      : FAILED")
        exit_code = 1

    # Canary regressions
    if canary:
        n_reg = canary.get("regressions", 0)
        run_type = canary.get("run_type", "")
        if run_type == "baseline":
            print(f"  Canary             : BASELINE SAVED ({canary.get('total', 0)} queries)")
        elif n_reg:
            print(f"\n  CANARY REGRESSIONS ({n_reg})")
            for reg in canary.get("regression_detail", []):
                print(f"    [{reg['query_id']}] {reg['domain']}")
                for reason in reg["reasons"]:
                    print(f"      ✗ {reason}")
            exit_code = 1
        else:
            print(f"  Canary             : PASS ({canary.get('total', 0)} queries)")
    else:
        print("  Canary             : FAILED")
        exit_code = 1

    # Cluster flags
    if clusters and not clusters.get("skipped"):
        cluster_flags = clusters.get("flags", [])
        if cluster_flags:
            print(f"\n  CLUSTER FLAGS ({len(cluster_flags)})")
            for flag in cluster_flags:
                print(f"    ⚑  {flag}")
        else:
            n = clusters.get("n_clusters", 0)
            print(f"  Query Clusters     : {n} clusters, no dominant/growing topics")
    elif clusters and clusters.get("skipped"):
        print(f"  Query Clusters     : SKIPPED — {clusters.get('reason', '')}")
    else:
        print("  Query Clusters     : FAILED")

    # Sample size
    if sample:
        if sample.get("skipped"):
            print(f"  Weekly Sample      : SKIPPED — {sample.get('reason', '')}")
        else:
            n_s = sample.get("sample_size", 0)
            pool = sample.get("total_pool", 0)
            print(f"  Weekly Sample      : {n_s} queries selected from {pool}-query pool")
    else:
        print("  Weekly Sample      : FAILED")

    print(f"\n{'='*W}")
    status = "ALL CLEAR" if exit_code == 0 else "ACTION REQUIRED — see flags above"
    print(f"  Status: {status}")
    print(f"{'='*W}\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
