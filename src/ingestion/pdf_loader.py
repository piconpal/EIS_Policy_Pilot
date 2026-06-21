"""
pdf_loader.py — Step 2 (table-aware)
Loads all PDFs from data/raw/ and extracts text page by page using pdfplumber.

Tables are detected per page and converted to pipe-delimited structured text
(Header1: val1 | Header2: val2 | ...) to preserve their semantic meaning
(e.g. score thresholds, policy matrices, CVSS rating tables).

Non-table body text is extracted from the remaining page area to avoid
duplicating table cell content in the output.
"""

import logging
from pathlib import Path

import yaml
import pdfplumber

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

_config: dict = _load_config()


# ── Table conversion ───────────────────────────────────────────────────────────

def _table_to_structured_text(table_data: list[list[str | None]]) -> str:
    """
    Convert a pdfplumber table (list of rows, each a list of cell strings)
    to pipe-delimited key:value text.

    First row is treated as column headers. Each subsequent data row becomes
    one line:  Header1: val1 | Header2: val2 | ...

    Empty rows and fully-empty cells are skipped.
    If there are no headers (all empty first row), values are joined directly.
    """
    if not table_data:
        return ""

    rows = [[cell.strip() if cell else "" for cell in row] for row in table_data]
    headers = rows[0]
    has_headers = any(h for h in headers)

    result_lines = []
    for row in rows[1:]:
        if not any(cell for cell in row):
            continue  # skip blank rows

        if has_headers:
            pairs = [
                f"{h}: {v}"
                for h, v in zip(headers, row)
                if h or v
            ]
            line = " | ".join(pairs)
        else:
            line = " | ".join(v for v in row if v)

        if line:
            result_lines.append(line)

    return "\n".join(result_lines)


# ── Page content extraction ────────────────────────────────────────────────────

def _extract_page_content(page) -> tuple[str, list[str]]:
    """
    Extract content from a pdfplumber page, separating body text from tables.

    - Tables are detected and converted to structured pipe-delimited text.
    - Non-table body text is extracted from regions *outside* table bounding
      boxes to avoid duplicating table cell values.

    Returns:
        (body_text, table_texts)
        body_text:   Non-table text from the page (may be empty).
        table_texts: List of structured table strings, one per table found.
                     Each string is prefixed with its table caption (the text
                     in the 50pt zone immediately above the table bounding box)
                     so that BM25 and vector retrieval can match table-specific
                     section headers like "Table 4.1 – Risk Score Action Thresholds".
                     Empty list if no tables detected.
    """
    found_tables = page.find_tables()

    if not found_tables:
        text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
        return text.strip(), []

    table_bboxes = [t.bbox for t in found_tables]

    def _not_in_any_table(obj) -> bool:
        ox0 = obj.get("x0", 0)
        oy0 = obj.get("top", 0)
        ox1 = obj.get("x1", 0)
        oy1 = obj.get("bottom", 0)
        for (tx0, ty0, tx1, ty1) in table_bboxes:
            if ox1 > tx0 and ox0 < tx1 and oy1 > ty0 and oy0 < ty1:
                return False
        return True

    body_text = (
        page.filter(_not_in_any_table).extract_text(x_tolerance=3, y_tolerance=3) or ""
    ).strip()

    table_texts: list[str] = []
    for tbl in found_tables:
        data = tbl.extract()
        if not data:
            continue
        structured = _table_to_structured_text(data)
        if not structured:
            continue

        # Extract caption: text in the 50pt zone immediately above the table
        tx0, ty0, tx1, ty1 = tbl.bbox
        caption_zone = (0, max(0, ty0 - 50), page.width, ty0)
        caption = (page.crop(caption_zone).extract_text() or "").strip()

        # Prefix the structured rows with the caption so retrieval can match
        # the table title (e.g. "Risk Score Action Thresholds") to the data
        full_table_text = f"{caption}\n{structured}" if caption else structured
        table_texts.append(full_table_text)

    return body_text, table_texts


# ── Public API ─────────────────────────────────────────────────────────────────

def load_pdfs(raw_data_path: str | None = None) -> list[dict]:
    """
    Load all PDFs from raw_data_path and extract text page by page.

    Tables are detected per page and converted to pipe-delimited structured
    text to preserve semantic meaning (score thresholds, policy matrices, etc.).
    Non-table body text is extracted from the remaining page region.

    Args:
        raw_data_path: Directory containing PDF files.
                       Defaults to raw_data_path from config.yaml.

    Returns:
        List of dicts, each representing one extractable unit (page or table):
        {
            "text":        str  — extracted text,
            "page_number": int  — 1-based page number,
            "source_file": str  — filename (not full path),
            "has_table":   bool — True for table-content entries,
        }

        Pages that contain tables produce TWO types of entries:
          1. A body-text entry  (has_table=False) for the non-table text.
          2. One table entry    (has_table=True)  per detected table, containing
             pipe-delimited structured rows only (no body text mixed in).
        This separation ensures table rows always form their own chunks
        and are never interleaved with recursive-split body text overflow.

    Raises:
        FileNotFoundError: If raw_data_path does not exist.
        ValueError: If no PDF files are found in the directory.
    """
    if raw_data_path is None:
        raw_data_path = _config["raw_data_path"]

    project_root = Path(__file__).resolve().parents[2]
    data_dir = (project_root / raw_data_path).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {data_dir}")

    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {data_dir}")

    logger.info("Found %d PDF file(s) in %s", len(pdf_files), data_dir)

    pages: list[dict] = []

    for pdf_path in pdf_files:
        logger.info("Loading: %s", pdf_path.name)
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                num_pages = len(pdf.pages)
                table_pages = 0

                for page_index, page in enumerate(pdf.pages):
                    page_number = page_index + 1
                    body_text, table_texts = _extract_page_content(page)

                    if not body_text and not table_texts:
                        logger.warning(
                            "  Skipping blank page %d/%d in %s",
                            page_number, num_pages, pdf_path.name,
                        )
                        continue

                    # Emit body text as a regular page entry
                    if body_text:
                        pages.append({
                            "text":        body_text,
                            "page_number": page_number,
                            "source_file": pdf_path.name,
                            "has_table":   False,
                        })

                    # Emit each table as a separate entry so it never mixes
                    # with body text during chunking
                    for table_text in table_texts:
                        pages.append({
                            "text":        table_text,
                            "page_number": page_number,
                            "source_file": pdf_path.name,
                            "has_table":   True,
                        })
                        table_pages += 1

                logger.info(
                    "  Extracted %d page(s) (%d tables) from %s",
                    num_pages, table_pages, pdf_path.name,
                )

        except Exception as e:
            logger.error("  Failed to read %s: %s", pdf_path.name, e)
            continue

    logger.info("Total entries extracted: %d", len(pages))
    return pages


# ── Smoke-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    results = load_pdfs()
    table_entries = [r for r in results if r.get("has_table")]
    print(f"\nTotal entries: {len(results)} | Table entries: {len(table_entries)}\n")

    for r in table_entries[:3]:
        print(f"\n{'='*65}")
        print(f"[{r['source_file']} | Page {r['page_number']} | has_table=True]")
        print(f"{'='*65}")
        print(r["text"][:600])
        print("---")
