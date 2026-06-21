"""
test_chunking.py
Verifies the full ingestion pipeline (pdf_loader → chunker) on a single PDF.

Usage:
    python test_chunking.py                        # defaults to iam_rbac_policy.pdf
    python test_chunking.py fraud_insider_threat.pdf

Steps verified:
  1. PDF loads and pages are extracted
  2. Hybrid chunking (section-aware + recursive fallback) runs
  3. Chunks are saved to data/processed/<stem>_chunks.json
  4. Chunk metadata fields are all present and valid
"""

import sys
import json
import logging
from pathlib import Path

# ── Make src/ importable from project root ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.chunker import chunk_pages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Required metadata fields every chunk must have ────────────────────────────
REQUIRED_FIELDS = {
    "chunk_id",
    "source_file",
    "page_number",
    "section_header",
    "text",
    "chunk_size_actual",
    "ingestion_timestamp",
}


def run_test(pdf_filename: str = "iam_rbac_policy.pdf") -> None:
    pdf_path = PROJECT_ROOT / "data" / "raw" / pdf_filename

    print("\n" + "=" * 60)
    print(f"  CHUNKING TEST — {pdf_filename}")
    print("=" * 60)

    # ── Step 1: Verify PDF exists ──────────────────────────────────────────────
    if not pdf_path.exists():
        print(f"\n[FAIL] PDF not found: {pdf_path}")
        print(f"       Available PDFs in data/raw/:")
        for p in sorted((PROJECT_ROOT / "data" / "raw").glob("*.pdf")):
            print(f"         - {p.name}")
        sys.exit(1)

    print(f"\n[OK] PDF found: {pdf_path}")

    # ── Step 2: Load pages ─────────────────────────────────────────────────────
    print("\n--- Stage 1: PDF Loading ---")
    pages = load_pdfs(str(PROJECT_ROOT / "data" / "raw"))

    # Filter to only the target PDF for focused testing
    pages = [p for p in pages if p["source_file"] == pdf_filename]

    if not pages:
        print(f"\n[FAIL] No pages extracted from {pdf_filename}. "
              "The PDF may be scanned/image-only.")
        sys.exit(1)

    print(f"[OK] Pages extracted : {len(pages)}")
    print(f"     First page preview ({pages[0]['source_file']}, "
          f"p{pages[0]['page_number']}):")
    print(f"     {pages[0]['text'][:200].strip()!r}")

    # ── Step 3: Chunk pages ────────────────────────────────────────────────────
    print("\n--- Stage 2: Hybrid Chunking ---")
    chunks = chunk_pages(pages)

    if not chunks:
        print("\n[FAIL] No chunks produced.")
        sys.exit(1)

    print(f"[OK] Chunks produced : {len(chunks)}")

    # ── Step 4: Verify output file ─────────────────────────────────────────────
    print("\n--- Stage 3: Output File Verification ---")
    stem         = Path(pdf_filename).stem
    output_path  = PROJECT_ROOT / "data" / "processed" / f"{stem}_chunks.json"

    if not output_path.exists():
        print(f"\n[FAIL] Output file not found: {output_path}")
        sys.exit(1)

    print(f"[OK] Output file     : {output_path}")
    print(f"     File size       : {output_path.stat().st_size / 1024:.1f} KB")

    # ── Step 5: Validate chunk contents ───────────────────────────────────────
    print("\n--- Stage 4: Chunk Metadata Validation ---")
    with open(output_path, "r", encoding="utf-8") as f:
        saved_chunks = json.load(f)

    errors = []
    for i, chunk in enumerate(saved_chunks):
        missing = REQUIRED_FIELDS - set(chunk.keys())
        if missing:
            errors.append(f"  Chunk {i} missing fields: {missing}")

        if not chunk.get("text", "").strip():
            errors.append(f"  Chunk {i} has empty text")

        if chunk.get("chunk_size_actual", 0) != len(chunk.get("text", "")):
            errors.append(
                f"  Chunk {i} chunk_size_actual mismatch: "
                f"stored={chunk.get('chunk_size_actual')} "
                f"actual={len(chunk.get('text', ''))}"
            )

    if errors:
        print(f"[FAIL] {len(errors)} validation error(s):")
        for e in errors:
            print(e)
        sys.exit(1)

    print(f"[OK] All {len(saved_chunks)} chunks pass metadata validation")

    # ── Step 6: Summary stats ──────────────────────────────────────────────────
    print("\n--- Summary ---")
    sizes      = [c["chunk_size_actual"] for c in saved_chunks]
    sections   = set(c["section_header"] for c in saved_chunks)
    pages_seen = sorted(set(c["page_number"] for c in saved_chunks))

    print(f"  PDF              : {pdf_filename}")
    print(f"  Pages processed  : {pages_seen}")
    print(f"  Total chunks     : {len(saved_chunks)}")
    print(f"  Avg chunk size   : {sum(sizes) / len(sizes):.0f} chars")
    print(f"  Min chunk size   : {min(sizes)} chars")
    print(f"  Max chunk size   : {max(sizes)} chars")
    print(f"  Unique sections  : {len(sections)}")

    if sections - {""}:
        print(f"  Section headers detected:")
        for s in sorted(sections - {""}):
            print(f"    · {s}")
    else:
        print("  (No section headers detected — all chunks used recursive fallback)")

    # ── Step 7: Print sample chunks ───────────────────────────────────────────
    print("\n--- Sample Chunks (first 3) ---")
    for c in saved_chunks[:3]:
        print(
            f"\n  chunk_id     : {c['chunk_id']}\n"
            f"  source_file  : {c['source_file']}\n"
            f"  page_number  : {c['page_number']}\n"
            f"  section      : '{c['section_header']}'\n"
            f"  size (chars) : {c['chunk_size_actual']}\n"
            f"  timestamp    : {c['ingestion_timestamp']}\n"
            f"  text preview : {c['text'][:120]!r}"
        )

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    target_pdf = sys.argv[1] if len(sys.argv) > 1 else "iam_rbac_policy.pdf"
    run_test(target_pdf)
