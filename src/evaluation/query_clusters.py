"""
query_clusters.py
Query Clustering for Topic Drift Detection.

Embeds all real-user queries from retrieval_log.db and clusters them
using KMeans (n=7, matching the 7 KB domains) to surface emerging
topics and KB coverage gaps.

Steps:
  1. Load query_text from DB (is_eval=0, input_safe=1). Deduplicate.
  2. Embed using the same SentenceTransformer singleton as retriever.py.
  3. Cluster with KMeans (n_clusters=7).
  4. Extract 5 most representative queries per cluster (closest to centroid).
  5. Compute cluster size and % of total volume.
  6. Flag clusters with >25% of total volume (dominant topics).
  7. Compare cluster sizes week-over-week if a prior report exists.
     Flag clusters that grew >20% in share vs last run.

Run:
    python -m src.evaluation.query_clusters

Output:
    logs/cluster_report.json
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.logging.retrieval_logger import _engine
from src.retrieval.retriever      import _get_model

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

_config       = _load_config()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Main ────────────────────────────────────────────────────────────────────────

def run_query_clusters(n_clusters: int = 7) -> dict:
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize as sk_normalize
    except ImportError:
        raise RuntimeError(
            "scikit-learn is required. Install it with:\n"
            "  pip install scikit-learn"
        )

    report_path = _PROJECT_ROOT / _config.get("cluster_report_path", "logs/cluster_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load deduplicated queries from DB ──────────────────────────────────
    with Session(_engine) as db:
        rows = db.execute(text("""
            SELECT DISTINCT query_text
            FROM retrieval_logs
            WHERE is_eval = 0 AND input_safe = 1
              AND query_text IS NOT NULL
              AND LENGTH(TRIM(query_text)) > 0
        """)).fetchall()

    queries = [r[0].strip() for r in rows if r[0] and r[0].strip()]

    if len(queries) < n_clusters:
        logger.warning(
            "Only %d unique queries in DB — need at least %d for %d clusters. "
            "Skipping clustering.",
            len(queries), n_clusters, n_clusters,
        )
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_queries": len(queries),
            "n_clusters": n_clusters,
            "skipped": True,
            "reason": f"Insufficient queries ({len(queries)} < {n_clusters})",
            "clusters": [],
            "flags": [],
        }
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\n  Clustering skipped — only {len(queries)} unique queries in DB "
              f"(need >= {n_clusters}).\n")
        return report

    logger.info("Embedding %d unique queries for clustering …", len(queries))
    print(f"\n  Embedding {len(queries)} queries …", flush=True)

    # ── 2. Embed using the same singleton as retriever.py ────────────────────
    embedding_model = _config.get("embedding_model", "all-MiniLM-L6-v2")
    model = _get_model(embedding_model)
    embeddings = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    # embeddings shape: (n_queries, embed_dim)

    # ── 3. KMeans clustering ──────────────────────────────────────────────────
    logger.info("Running KMeans (n_clusters=%d) …", n_clusters)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_          # shape: (n_clusters, embed_dim)

    # ── 4. Per-cluster analysis ───────────────────────────────────────────────
    total_queries = len(queries)
    clusters = []

    for cluster_id in range(n_clusters):
        # Indices of queries in this cluster
        member_idxs = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        if not member_idxs:
            continue

        cluster_size   = len(member_idxs)
        volume_pct     = round(cluster_size / total_queries * 100, 2)
        is_dominant    = volume_pct > 25.0

        # 5 queries closest to centroid (by cosine distance)
        centroid = centroids[cluster_id]
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)

        member_embeds  = embeddings[member_idxs]           # (k, d)
        cos_sims       = member_embeds @ centroid_norm     # (k,)
        top5_local     = np.argsort(cos_sims)[::-1][:5]   # closest 5 by cosine sim
        representative = [queries[member_idxs[i]] for i in top5_local]

        clusters.append({
            "cluster_id":       cluster_id,
            "size":             cluster_size,
            "volume_pct":       volume_pct,
            "is_dominant":      is_dominant,
            "representative_queries": representative,
            "growth_flag":      False,     # filled in during WoW comparison below
            "growth_pct":       None,
        })

    # ── 5. Week-over-week comparison ──────────────────────────────────────────
    flags: list[str] = []
    prior_cluster_sizes: dict[int, float] = {}

    if report_path.exists():
        try:
            prior = json.loads(report_path.read_text())
            for c in prior.get("clusters", []):
                prior_cluster_sizes[c["cluster_id"]] = c.get("volume_pct", 0.0)
        except (json.JSONDecodeError, KeyError):
            pass

    for c in clusters:
        prior_pct = prior_cluster_sizes.get(c["cluster_id"])
        if prior_pct is not None and prior_pct > 0:
            growth = (c["volume_pct"] - prior_pct) / prior_pct
            c["growth_pct"] = round(growth * 100, 2)
            if growth > 0.20:
                c["growth_flag"] = True
                flags.append(
                    f"TOPIC GROWTH: cluster {c['cluster_id']} grew "
                    f"{c['growth_pct']:.1f}% in share "
                    f"({prior_pct:.1f}% → {c['volume_pct']:.1f}%)\n"
                    f"    Top query: \"{c['representative_queries'][0]}\""
                )

        if c["is_dominant"]:
            flags.append(
                f"DOMINANT TOPIC: cluster {c['cluster_id']} represents "
                f"{c['volume_pct']:.1f}% of query volume\n"
                f"    Top query: \"{c['representative_queries'][0]}\""
            )

    report = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "total_queries": total_queries,
        "n_clusters":    n_clusters,
        "embedding_model": embedding_model,
        "skipped":       False,
        "clusters":      clusters,
        "flags":         flags,
    }
    report_path.write_text(json.dumps(report, indent=2))

    # ── Print summary ─────────────────────────────────────────────────────────
    W = 62
    print(f"\n{'='*W}")
    print(f"  QUERY CLUSTER REPORT — {total_queries} queries → {n_clusters} clusters")
    print(f"{'='*W}")
    for c in sorted(clusters, key=lambda x: x["volume_pct"], reverse=True):
        dom = " [DOMINANT]" if c["is_dominant"] else ""
        gro = f" [+{c['growth_pct']:.0f}% growth]" if c.get("growth_flag") else ""
        print(f"\n  Cluster {c['cluster_id']} — {c['size']} queries ({c['volume_pct']:.1f}%){dom}{gro}")
        for q in c["representative_queries"]:
            print(f"    • {q[:75]}")
    if flags:
        print(f"\n  FLAGS ({len(flags)})")
        for f in flags:
            print(f"  ⚑  {f}")
    else:
        print(f"\n  FLAGS  None.")
    print(f"\n  Report saved → {report_path}")
    print(f"{'='*W}\n")

    return report


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_query_clusters()
